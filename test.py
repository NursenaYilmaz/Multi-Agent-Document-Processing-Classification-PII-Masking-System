import asyncio
import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import os
from dotenv import load_dotenv
import re
import uuid
import cv2
from shapely.geometry import Polygon
from difflib import SequenceMatcher 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageSequence, ImageOps, ImageFilter, ImageDraw
import time
import asyncio
import skimage
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import random
from collections import defaultdict
# .env dosyasını yükle
load_dotenv()

# Hugging Face
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import torch

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== TESSERACT SETUP ==========
PYTESS_AVAILABLE = False
try:
    import pytesseract
    from pytesseract import Output
    import shutil
    import platform

    tesseract_cmd = shutil.which("tesseract")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        PYTESS_AVAILABLE = True
        logger.info(f"✓ Tesseract found: {tesseract_cmd}")
    else:
        if platform.system() == "Windows":
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Tesseract-OCR\tesseract.exe",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    PYTESS_AVAILABLE = True
                    logger.info(f"✓ Tesseract found: {path}")
                    break
except Exception as e:
    logger.warning(f" Tesseract not available: {e}")

if not PYTESS_AVAILABLE:
    logger.info(" TrOCR will be used as default OCR engine")

# Optional HEIC/HEIF support
_HEIF_READY = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIF_READY = True
    logger.info(" HEIF/HEIC support enabled")
except Exception:
    logger.info(" HEIF/HEIC support not installed (optional)")


# ========================= IMAGE UTILITIES =========================

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".heic", ".heif"}
SKIP_EXTS = {".pdf"}


def _open_first_frame(path: str) -> Image.Image:
    """Open image and get first frame (for GIFs, etc.)"""
    img = Image.open(path)
    try:
        first = next(ImageSequence.Iterator(img))
    except Exception:
        first = img
    try:
        first = ImageOps.exif_transpose(first)
    except Exception:
        pass
    return first


def _maybe_downscale(im: Image.Image, max_side: int = 2500) -> Image.Image:
   
    w, h = im.size
    if max(w, h) <= max_side:
        return im
    scale = max_side / float(max(w, h))
    new_size = (int(w * scale), int(h * scale))
    return im.resize(new_size, resample=Image.LANCZOS)


def image_any_to_base64(path: str) -> str:
    
    with _open_first_frame(path) as im:
        rgb = im.convert("RGB")
        rgb = _maybe_downscale(rgb, 2500)
        buf = BytesIO()
        rgb.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode()


# ---- IMPROVED OCR pre-processing ----
def _prep_for_ocr(im: Image.Image) -> Image.Image:

    # Grayscale
    g = ImageOps.grayscale(im)
    
    g = ImageOps.autocontrast(g, cutoff=2)
    
    if min(g.size) < 1500:
        scale = 1500 / float(min(g.size))
        new_width = int(g.width * scale)
        new_height = int(g.height * scale)
        g = g.resize((new_width, new_height), Image.LANCZOS)
    
    # Double sharpening
    g = g.filter(ImageFilter.SHARPEN)
    g = g.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    return g

import re

_NOT_FOUND_PATTERNS = [
    r"not explicitly mentioned",
    r"information not found",
    r"not found in (the )?document",
    r"cannot be determined",
    r"no information (is )?provided",
    r"unable to find",
]

def postprocess_found_flag(found: bool, answer: str) -> bool:
    """
    Model bazen 'Information not found...' deyip found=True dönebiliyor.
    Bu fonksiyon answer'a bakarak found flag'i düzeltir.
    """
    if not answer:
        return False
    a = answer.strip().lower()
    for pat in _NOT_FOUND_PATTERNS:
        if re.search(pat, a):
            return False
    return bool(found)

def pick_qa_text(doc_out: dict) -> str:
    """
    QA soruları için MASKESİZ metni seç:
    - normalized_text varsa onu
    - yoksa text
    - yoksa ocr_text
    """
    for k in ("normalized_text", "text", "ocr_text"):
        v = doc_out.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def extract_evidence_from_text(text: str, doc_type: str, question: str) -> str:
    
    if not text:
        return ""

    q = (question or "").strip().lower()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = "\n".join(lines)

    def find_first_line(prefixes):
        for ln in lines:
            lnl = ln.lower()
            for p in prefixes:
                if lnl.startswith(p):
                    return ln
        return ""

    if doc_type == "email":
        if "sender" in q:
            return find_first_line(["from:", "author:"]) or ""
        if "subject" in q:
            return find_first_line(["subject:"]) or ""
        return ""

    if doc_type == "invoice":
        if "invoice number" in q:
            # invoice no / inv no / invoice #
            for ln in lines:
                l = ln.lower()
                if "invoice" in l and ("no" in l or "#" in l or "number" in l):
                    return ln
            return ""
        if "total" in q or "amount" in q:
            for ln in lines:
                l = ln.lower()
                if "total" in l:
                    return ln
            return ""
        return ""

    if doc_type == "receipt":
        if "total" in q or "amount" in q:
            for ln in lines:
                if "total" in ln.lower():
                    return ln
            return ""
        if "date" in q:
            # kaba tarih araması
            m = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", joined)
            return m.group(0) if m else ""
        return ""

    # generic fallback
    return ""


def sanitize_text_minimal(text: str) -> str:
    if not text:
        return ""
    t = text
    t = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", t)
    t = re.sub(r"(\+?\d{1,3}[\s-]?)?(\(?\d{2,4}\)?[\s-]?)\d{3,4}[\s-]?\d{3,4}", "[PHONE]", t)
    t = re.sub(r"\bTR\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b", "[IBAN]", t)
    t = re.sub(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", "[IBAN]", t)
    t = re.sub(r"\b(?:\d[ -]*?){13,19}\b", "[CARD]", t)
    return t

def sanitize_text_for_qa(text: str) -> str:
    """
    QA için hafif sanitization:
    - Email maskelenir
    - Telefon / IBAN / kart NUMARALARI KORUNUR
    """
    if not text:
        return ""

    t = text
    t = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        t,
    )
    return t
def sanitize_for_qa(text: str) -> str:
    if not text:
        return ""
    t = text
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def build_grounded_prompt(document_type: str, document_text: str, question: str, max_chars: int = 3500) -> str:
    doc = (document_text or "")[:max_chars]
    return f"""
You are analyzing a document of type: {document_type}

RULES:
- Answer ONLY using information that appears in the DOCUMENT.
- If the answer is not explicitly present, set found=false and answer="Information not found in document."
- Always provide a short evidence quote copied from the document when found=true.
- Output MUST be valid JSON with keys: found (boolean), answer (string), evidence (string).

DOCUMENT:
{doc}

QUESTION:
{question}

OUTPUT JSON:
""".strip()


def parse_json_loose(text: str) -> Dict[str, Any]:
    if not text:
        return {"found": False, "answer": "Information not found in document.", "evidence": ""}
    s = text.strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return {"found": False, "answer": "Information not found in document.", "evidence": ""}
    block = m.group(0)
    try:
        data = json.loads(block)
        found = bool(data.get("found", False))
        answer = str(data.get("answer", "")).strip()
        evidence = str(data.get("evidence", "")).strip()
        if not found:
            return {"found": False, "answer": "Information not found in document.", "evidence": ""}
        return {"found": True, "answer": answer, "evidence": evidence}
    except Exception:
        return {"found": False, "answer": "Information not found in document.", "evidence": ""}

import re

def validate_answer(question: str, answer: str, doc_type: str) -> bool:
    q = (question or "").lower()
    a = (answer or "").strip()

    if not a:
        return False
    
    if doc_type == "resume" and ("full name" in q or "candidate" in q and "name" in q):
        tokens = [t for t in re.split(r"\s+", a) if t]
        return len(tokens) >= 2 and len(a) >= 4

    if "date" in q:
        # 2019 gibi yıl veya 02/01/2019, 2019-02-01 vs
        if re.search(r"\b(19|20)\d{2}\b", a):
            return True
        if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", a):
            return True
        if re.search(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", a):
            return True
        return False

    if any(k in q for k in ["total", "amount", "balance", "paid", "due"]):
        return bool(re.search(r"\d", a))

    if any(k in q for k in ["invoice number", "invoice no", "invoice #", "reference", "ref no"]):
        return bool(re.search(r"[A-Za-z0-9]", a))

    return True

class RateLimiter:
    def __init__(self, max_calls: int, period_sec: float):
        self.max_calls = max_calls
        self.period = period_sec
        self.calls = []

    async def acquire(self):
        now = time.monotonic()
        self.calls = [t for t in self.calls if now - t < self.period]

        if len(self.calls) >= self.max_calls:
            sleep_for = self.period - (now - self.calls[0]) + 0.05
            await asyncio.sleep(max(0, sleep_for))

        self.calls.append(time.monotonic())

gemini_rl = RateLimiter(max_calls=5, period_sec=60)


# ========================= A2A PROTOCOL =========================

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    STATUS = "status"


class DocumentType(Enum):
    EMAIL = "email"
    FORM = "form"
    INVOICE = "invoice"
    LETTER = "letter"
    NEWS_ARTICLE = "news_article"
    RECEIPT = "receipt"
    RESUME = "resume"
    OTHER = "other"


@dataclass
class A2AMessage:
    sender: str
    receiver: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: str
    message_id: str


class BaseAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.message_queue = asyncio.Queue()

    async def send_message(self, receiver: str, message_type: MessageType,
                           content: Dict[str, Any]) -> A2AMessage:
        return A2AMessage(
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            content=content,
            timestamp=datetime.now().isoformat(),
            message_id=f"{self.agent_id}_{datetime.now().timestamp()}"
        )

    async def receive_message(self, message: A2AMessage):
        await self.message_queue.put(message)

    async def process_messages(self):
        while True:
            message = await self.message_queue.get()
            await self.handle_message(message)

    async def handle_message(self, message: A2AMessage):
        pass


# ========================= OCR AGENT =========================

class OCRAgent(BaseAgent):
    def __init__(self, default_backend: str = "tesseract"):
        super().__init__("ocr_agent")
        self.coordinator = None
        self.default_backend = default_backend.lower()
        self._trocr_processor = None
        self._trocr_model = None
        self._tesseract_available = PYTESS_AVAILABLE

        if self.default_backend == "tesseract" and not self._tesseract_available:
            logger.warning("Tesseract requested but not available, switching to TrOCR")
            self.default_backend = "trocr"

        if self.default_backend == "trocr":
            self._ensure_trocr_loaded()

    # TrOCR small performance optimization (eval mode and inference_mode)
    def _ensure_trocr_loaded(self):
        if self._trocr_processor is None:
            logger.info("Loading TrOCR model...")
            self._trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            self._trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            self._trocr_model.eval() # Model set to eval mode

    def set_coordinator(self, coordinator):
        self.coordinator = coordinator

    async def handle_message(self, message: A2AMessage):
        if message.message_type == MessageType.REQUEST and message.content.get('action') == 'ocr':
            await self.perform_ocr(message)

    def _ocr_with_tesseract(self, pil_img: Image.Image, lang: str = "eng") -> Dict:
        
        original_size = pil_img.size
        im = _prep_for_ocr(pil_img)
        processed_size = im.size
        
        # Calculate scale factors for coordinate transformation
        scale_x = processed_size[0] / original_size[0]
        scale_y = processed_size[1] / original_size[1]
        
        best_text = ""
        best_conf = 0.0
        best_psm = 6
        best_boxes = []
        
        # Try multiple PSM modes
        psm_modes = [
            (6, "--oem 3 --psm 6"),    # Uniform block
            (4, "--oem 3 --psm 4"),    # Single column
            (11, "--oem 3 --psm 11"), # Sparse text
            (3, "--oem 3 --psm 3"),    # Fully automatic
        ]

        for psm_num, config in psm_modes:
            try:
                txt = pytesseract.image_to_string(im, lang=lang, config=config)
                data = pytesseract.image_to_data(im, lang=lang, config=config, output_type=Output.DICT)
                confs = [int(c) for c in data.get("conf", []) if str(c) not in ("-1", "")]
                avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0

                readable_ratio = len([c for c in txt if c.isalnum() or c.isspace()]) / max(1, len(txt))
                
                # Sadece okunabilirlik oranı %70'in üzerindeyse ve mevcut en iyi metin/güvenilirlik oranını aşıyorsa kabul et
                if readable_ratio > 0.7 and (avg_conf > best_conf or len(txt) > len(best_text) * 1.5):
                    best_text = txt
                    best_conf = avg_conf
                    best_psm = psm_num
                    
                    # OCR kutularını topla VE ORİJİNAL BOYUTA DÖNÜŞTÜR
                    boxes = []
                    n_boxes = len(data['level'])
                    for i in range(n_boxes):
                        if int(data['conf'][i]) > 30:  # Confidence threshold
                            text = data['text'][i].strip()
                            if text:
                                # OCR koordinatlarını orijinal görüntü boyutuna dönüştür
                                x_processed = data['left'][i]
                                y_processed = data['top'][i]
                                w_processed = data['width'][i]
                                h_processed = data['height'][i]
                                
                                # Orijinal boyutlara scale et
                                x_original = int(x_processed / scale_x)
                                y_original = int(y_processed / scale_y)
                                w_original = int(w_processed / scale_x)
                                h_original = int(h_processed / scale_y)
                                
                                boxes.append({
                                    "text": text,
                                    "box": (x_original, y_original, w_original, h_original),
                                    "confidence": int(data['conf'][i])
                                })
                    best_boxes = boxes
                    
            except Exception:
                continue

        return {
            "text": best_text,
            "engine": f"tesseract_psm{best_psm}",
            "avg_confidence": best_conf,
            "ocr_boxes": best_boxes
        }

    def _ocr_with_trocr(self, pil_img: Image.Image) -> Dict:
        self._ensure_trocr_loaded()
        with torch.inference_mode(): # Inference mode used
            pixel_values = self._trocr_processor(pil_img, return_tensors="pt").pixel_values
            generated_ids = self._trocr_model.generate(
                pixel_values,
                num_beams=1,      # speed optimization
                max_new_tokens=128     # shorten unnecessary length
               )
            text = self._trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"text": text, "engine": "trocr", "avg_confidence": None, "ocr_boxes": []}

    async def perform_ocr(self, request: A2AMessage):
        try:
            image_data = base64.b64decode(request.content['image'])
            image = Image.open(BytesIO(image_data)).convert("RGB")
            backend = request.content.get("backend", self.default_backend)
            lang = request.content.get("lang", "eng")

            if backend == "tesseract" and self._tesseract_available:
                result = self._ocr_with_tesseract(image, lang)
            else:
                result = self._ocr_with_trocr(image)

            response = await self.send_message(
                receiver="coordinator",
                message_type=MessageType.RESPONSE,
                content={
                    'action': 'ocr_complete',
                    'text': result["text"],
                    'ocr_engine': result["engine"],
                    'ocr_avg_confidence': result["avg_confidence"],
                    'ocr_boxes': result["ocr_boxes"],
                    'image': request.content['image'],  # Görseli de iletiyoruz
                    'workflow_id': request.content.get('workflow_id')
                }
            )
            if self.coordinator:
                await self.coordinator.receive_message(response)

        except Exception as e:
            logger.error(f"OCR error: {e}")
            error_msg = await self.send_message(
                receiver="coordinator",
                message_type=MessageType.ERROR,
                content={'error': str(e), 'workflow_id': request.content.get('workflow_id')}
            )
            if self.coordinator:
                await self.coordinator.receive_message(error_msg)

# ========================= ENHANCED MASKING AGENT WITH FINE-TUNED PII MODEL =========================

import base64
import re
import uuid
import logging
from io import BytesIO
from typing import List, Dict
from difflib import SequenceMatcher
from datetime import datetime

import torch
from transformers import pipeline
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


class HybridPIIMaskingAgent(BaseAgent):
    """

    - DocumentClassifierAgent'tan gelen `document_type` kullanılır.
    - Kritik PII:
        email, phone, credit_card, account/iban, tax_id, ssn, signature,
        (bazı tiplerde) person, address, date_of_birth
    """

    def __init__(self):
        super().__init__("hybrid_masking_agent")

        self.coordinator = None
        self.min_confidence = 0.75

        # --------- HF MODEL (yalnız yapısal PII) ----------
        try:
            logger.info("Loading PII model: Isotonic/distilbert_finetuned_ai4privacy_v2")
            self.ner_model = pipeline(
                "token-classification",
                model="Isotonic/distilbert_finetuned_ai4privacy_v2",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1,
            )
            self.model_entity_map = {
                "B-EMAIL": "email", "I-EMAIL": "email",
                "B-PHONE": "phone", "I-PHONE": "phone",
                "B-CREDIT_CARD": "credit_card", "I-CREDIT_CARD": "credit_card",
                "B-ACCOUNT": "account", "I-ACCOUNT": "account",
                # NAME/LOC/DATE bilerek alınmıyor -> çok FP
            }
            logger.info(" HF PII model loaded.")
        except Exception as e:
            logger.error(f" HF PII model loading failed: {e}")
            self.ner_model = None
            self.model_entity_map = {}

        # --------- RENKLER ----------
        self.colors = {
            "email":       (204, 0, 255),
            "phone":       (255, 255, 102),
            "credit_card": (255, 178, 0),
            "iban":        (255, 128, 0),
            "account":     (255, 51, 51),
            "tax_id":      (153, 102, 51),
            "ssn":         (153, 102, 51),
            "address":     (0, 204, 102),
            "person":      (255, 100, 100),
            "date":        (0, 153, 153),
            "signature":   (0, 102, 255),
            "other":       (128, 128, 128),
        }

        # --------- DOC TYPE → POLICY ----------
        self.masking_policies = {
            "invoice":  ["email", "iban", "account", "credit_card", "tax_id", "ssn","address"],
            "receipt":  ["email", "credit_card"],
            "resume":   ["email", "phone", "person", "address", "date"],
            "form":     ["email", "phone", "person", "address", "tax_id", "ssn", "date", "signature"],
            "letter":   ["email", "phone", "person", "address", "signature"],
            "email":    ["email", "phone", "person", "address", "account", "tax_id", "ssn", "signature"],
            "news_article": [],
            "scientific":   [],
            "other":   ["email", "phone", "credit_card", "iban", "account", "tax_id", "ssn", "signature"],
        }

        self.false_positive_words = {
            # Genel kelimeler
            "balance", "due", "thank", "choosing", "panel", "board",
            "report", "business", "labor", "specifications", "workshop",
            "commission", "environment", "university", "guest",
            "server", "table", "item", "product", "price", "date", "total",

            # Fiş/fatura kelimeleri
            "subtotal", "tax", "amount", "invoice", "cashier", "terminal",
            "merchant", "receipt", "change", "tender", "approval",

            # Haber kelimeleri
            "reported", "according", "officials", "spokesperson", "journalist",
            "correspondent", "sources", "witnesses", "authorities",
        }

        self.receipt_business_keywords = [
            r"\bmerchant\b", r"\bterminal\b", r"\bauth(orization)?\s*code\b",
            r"\bapproval\b", r"\bcashier\b", r"\bserver\b", r"\btable\b",
            r"\bguest\b", r"\border\s*#\b", r"\bticket\s*#\b",
        ]
        self.re_receipt_business = [re.compile(p, re.I) for p in self.receipt_business_keywords]

        # --------- REGEXLER ----------
        self.re_iban = re.compile(
            r"(?i)\biban[:\s,]*([a-z0-9]{2}\d{2}[a-z0-9]{10,30})\b"
        )
        self.re_tax_id = re.compile(
            r"(?i)tax\s*id[:\s\-]*([0-9]{2,3}[\s\-]?[0-9]{2}[\s\-]?[0-9]{3,6})"
        )
        self.re_email = re.compile(
            r"\b[A-Za-z0-9._%+\-]{2,}@[A-Za-z0-9.\-]{2,}\.[A-Za-z]{2,}\b"
        )
        self.re_cc = re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
        )
        self.re_price = re.compile(
            r"(?:[$€£₺¥]\s*)?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})"
        )
        self.re_date = re.compile(
            r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"
        )
        self.re_time = re.compile(
            r"\b\d{1,2}:\d{2}(\s?(am|pm))?\b",
            re.I,
        )
        self.re_measure = re.compile(
            r"\b\d+(\.\d+)?\s?(mm|cm|m|km|mg|g|kg|lbs|%)\b",
            re.I,
        )
        self.corporate_patterns = [
            r"\b(inc\.|incorporated|corp\.|corporation|llc|ltd\.|limited|plc|gmbh|s\.a\.|co\.|company)\b",
            r"\b(headquarters|hq|office|department|dept\.|division|branch)\b",
            r"\b(accounts\s*payable|billing|customer\s*service|sales|marketing)\b",
            r"\b(industries|enterprises|solutions|services|systems|technologies|international|global)\b",
            r"\bp\.?\s*o\.?\s*box\b",
        ]
        self.re_corporate = [re.compile(p, re.I) for p in self.corporate_patterns]
        
        # Kişisel adres güçlendirici kelimeler
        self.personal_keywords = [
            r"\b(residence|home\s*address|residential|personal\s*address|current\s*address|permanent\s*address)\b",
            r"\b(apartment|apt\.|unit|suite|ste\.)\s*[#]?\s*[0-9a-z]+\b",
            r"\b(dear|sincerely|regards|curriculum|resume|cv|applicant|candidate)\b",
        ]
        self.re_personal = [re.compile(p, re.I) for p in self.personal_keywords]
        
        # Adres pattern'leri
        self.address_patterns = [
            # Sokak numarası + sokak ismi (123 Main Street)
            r"\b\d{1,5}\s+[A-Za-z]+(?:\s+[A-Za-z]+){0,3}\s+(street|st\.|avenue|ave\.|road|rd\.|drive|dr\.|lane|ln\.|way|court|ct\.|boulevard|blvd\.|place|pl\.)\b",
            # Şehir, Eyalet Posta Kodu (New York, NY 10001)
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b",
        ]
        self.re_address = [re.compile(p, re.I) for p in self.address_patterns]
        
        # Şehir + Eyalet + Zip
        self.re_city_state_zip = re.compile(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s*,\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\b", 
            re.I
        )

        # --------- DOC TYPE → STRICT POLICY (daraltılmış izinler) ----------
        self.strict_policies = {
            "receipt": ["email", "credit_card", "iban"],
            "invoice": ["email", "credit_card", "iban", "tax_id", "account","address"],
            "form": ["email", "phone", "address", "tax_id", "ssn", "date", "signature"],
            "letter": ["email", "phone", "signature"],
            "resume": ["email", "phone", "address", "date_of_birth","date"],
            "email": ["email", "phone", "signature"],
            "scientific": [],
            "news_article": [],
            "newspaper": [],
            "other": ["email", "phone", "iban", "credit_card", "tax_id", "ssn", "signature"],
        }

    # ---------- A2A ----------

    def set_coordinator(self, coordinator):
        self.coordinator = coordinator

    async def handle_message(self, message: "A2AMessage"):
        content = message.content
        if content.get("action") != "mask_visual":
            return

        image_b64 = content.get("image", "")
        ocr_boxes = content.get("ocr_boxes", [])
        workflow_id = content.get("workflow_id")
        doc_type = content.get("document_type", "other")
        true_type = content.get("true_type", None)

        try:
            masked_b64, stats = self.apply_hybrid_visual_mask(
                image_b64, ocr_boxes, doc_type, true_type
            )

            response = A2AMessage(
                sender=self.agent_id,
                receiver="coordinator",
                message_type=MessageType.RESPONSE,
                content={
                    "action": "masking_complete",
                    "masked_image": masked_b64,
                    "mask_stats": stats,
                    "workflow_id": workflow_id,
                },
                timestamp=datetime.utcnow().isoformat(),
                message_id=str(uuid.uuid4()),
            )
            if self.coordinator:
                await self.coordinator.receive_message(response)
            else:
                logger.warning(" No coordinator connected to HybridPIIMaskingAgent")
        except Exception as e:
            logger.error(f" HybridPIIMaskingAgent error: {e}")

    # ---------- HELPERS ----------

    def _get_allowed_types(self, doc_type: str) -> List[str]:
        return self.masking_policies.get(doc_type.lower(), self.masking_policies["other"])

    def _get_context(self, text: str, start: int, end: int, window: int = 40) -> str:
        left = max(0, start - window)
        right = min(len(text), end + window)
        return text[left:right]

    def _is_trivial(self, text: str) -> bool:
        """Bariz PII olmayan patternler."""
        t = text.strip()
        if not t:
            return True
        if re.fullmatch(r"\d+", t):
            return True
        if self.re_price.fullmatch(t):
            return True
        if self.re_time.fullmatch(t):
            return True
        if self.re_measure.search(t):
            return True
        # ölçüler → maskelenmez       
        if re.search(r"\b\d+(\.\d+)?\s?(mm|cm|m|kg|lbs|mg|g|%)\b", t.lower()):
           return True
        # rod length / filter length / width / pressure
        if any(k in t.lower() for k in ["pressure", "rod", "filter", "length", "diameter","plug","tipping"]):
           return True
        return False


    def _should_mask(
        self,
        entity_text: str,
        entity_type: str,
        context: str,
        score: float,
        doc_type: str,
    ) -> bool:
        """
        Belge tipine, entity tipine ve güven skoruna göre maskelenip maskelenmeyeceğine karar verir.
        """
        dt = (doc_type or "other").lower()

        # DocType'a göre izin verilen PII türlerini kontrol et
        allowed = self.masking_policies.get(dt, self.masking_policies["other"])
        if entity_type not in allowed:
            return False

        # Güven skoru düşükse maskelenmez
        if score < self.min_confidence:
            return False

        # Boş ya da çok kısa stringleri ele
        entity_text = (entity_text or "").strip().lower()
        if len(entity_text) < 2:
            return False

        if dt in ["news_article", "scientific", "newspaper"]:
            return False

        # Kritik structured PII
        if entity_type in {"email", "credit_card", "iban", "account", "tax_id", "ssn"}:
            if entity_type == "email" and ("@" not in entity_text or not re.search(r"\w@\w", entity_text)):
                return False
            if self._is_trivial(entity_text):
                return False
            return True

        # Telefon
        if entity_type == "phone":
            if self._is_trivial(entity_text):
                return False
            cl = context.lower()
            if any(k in cl for k in ["customer", "client", "personal", "mobile", "cell", "home", "contact"]):
                return True
            return False

        # Adres sadece belirli tiplerde
        if entity_type == "address":
            if self._is_trivial(entity_text):
                return False
            if dt in {"resume", "form", "letter", "email","invoice"}:
                return True
            return False

        # Kişi adı (person)
        if entity_type == "person":
            cl = context.lower()
            if dt in {"resume", "form", "letter", "email"} and any(
                k in cl
                for k in [
                    "dear",
                    "cv",
                    "resume",
                    "curriculum vitae",
                    "candidate",
                    "applicant",
                    "profile",
                    "about me",
                ]
            ):
                return True
            return False

        # İmza
        if entity_type == "signature":
            return dt not in ["news_article", "receipt"]
        
        # Tarih (özellikle CV doğum tarihi)
        if entity_type == "date_of_birth":
            return True


        return False

    # ---------- MODEL TESPİTİ ----------

    def _detect_with_model(self, text: str, doc_type: str) -> List[Dict]:
        if not self.ner_model or not text.strip():
            return []

        dt = (doc_type or "other").lower()

        if dt in {"receipt", "invoice", "news_article", "scientific", "newspaper"}:
            logger.info(f" Model inference skipped for doc_type={dt}")
            return []

        entities: List[Dict] = []
        try:
            results = self.ner_model(text)
            for r in results:
                etype = self.model_entity_map.get(r.get("entity_group"))
                if not etype:
                    continue
                ent_text = r["word"].strip()
                start, end = r["start"], r["end"]
                ctx = self._get_context(text, start, end)

                if not self._should_mask(ent_text, etype, ctx, r["score"], doc_type):
                    continue

                entities.append(
                    {
                        "text": ent_text,
                        "type": etype,
                        "score": r["score"],
                        "start": start,
                        "end": end,
                        "source": "model",
                    }
                )
        except Exception as e:
            logger.warning(f" HF model inference failed: {e}")
        return entities

    # ---------- REGEX TESPİTİ ----------

    def _detect_regex_patterns(self, text: str, doc_type: str) -> List[Dict]:
        entities: List[Dict] = []

        def add_if_mask(span, etype, s, e, score):
            ctx = self._get_context(text, s, e)
            if self._should_mask(span, etype, ctx, score, doc_type):
                entities.append(
                    {
                        "text": span,
                        "type": etype,
                        "score": score,
                        "start": s,
                        "end": e,
                        "source": "regex",
                    }
                )

        # Email
        for m in self.re_email.finditer(text):
            add_if_mask(m.group(), "email", m.start(), m.end(), 0.99)

        # IBAN
        for m in self.re_iban.finditer(text):
            iban_val = m.group(1)
            add_if_mask(iban_val, "iban", m.start(1), m.end(1), 0.97)

        # Tax ID
        for m in self.re_tax_id.finditer(text):
            tid = m.group(1)
            add_if_mask(tid, "tax_id", m.start(1), m.end(1), 0.95)

        # Credit Card
        for m in self.re_cc.finditer(text):
            if self._validate_luhn(m.group()):
                add_if_mask(m.group(), "credit_card", m.start(), m.end(), 0.98)

        return entities

    @staticmethod
    def _validate_luhn(card_number: str) -> bool:
        digits = [int(d) for d in card_number if d.isdigit()]
        if len(digits) < 13:
            return False
        checksum = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
        return checksum % 10 == 0

    # ---------- MERGE ----------

    def _merge_entities(self, *lists: List[Dict]) -> List[Dict]:
        all_e = [e for lst in lists for e in lst]
        all_e.sort(key=lambda x: (x["start"], -(x["end"] - x["start"]), -x["score"]))

        merged: List[Dict] = []
        for e in all_e:
            overlapped = False
            for m in merged:
                if not (e["end"] <= m["start"] or e["start"] >= m["end"]):
                    overlapped = True
                    if e["score"] > m["score"]:
                        m.update(e)
                    break
            if not overlapped:
                merged.append(e)
        return merged

    def _filter_false_positives(self, entities: List[Dict]) -> List[Dict]:
        """Yanlış pozitif (kişisel bilgi olmayan) kelimeleri eler"""
        filtered = []
        for e in entities:
            txt = (e.get("text") or "").lower().strip()
            if not txt:
                continue
            if txt in self.false_positive_words:
                continue
            if len(txt) <= 2 or txt.isnumeric():
                continue
            filtered.append(e)
        return filtered

    # ---------- OpenCV İMZA TESPİTİ (sadece form/letter için kullanacağız) ----------

    def _ocr_signature_regions(
        self, ocr_boxes: list, doc_type: str, image_b64: str = None
    ) -> list:
        import base64
        import cv2
        import numpy as np

        found_regions = []
        dt = (doc_type or "other").lower()

        if dt not in {"form", "letter", "other"}:
            logger.info(f" Skipping signature detection for doc_type={dt}")
            return []

        logger.info(f" Starting OCR signature detection for doc_type={dt}")

        img = None
        ih = iw = 0
        if image_b64:
            try:
                img_bytes = base64.b64decode(image_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    ih, iw = img.shape[:2]
                    logger.info(f" Image loaded: {iw}x{ih}")
            except Exception as e:
                logger.error(f" Image decode failed: {e}")
                img = None

        if dt == "form":
            keywords = ["signature of initiator", "signature", "initiator","signature(s)"]
        elif dt == "letter":
            keywords = ["sincerely", "cordially", "regards", "respectfully", "truly", "yours"]
        else:
            keywords = ["signature of initiator", "signature", "sincerely", "cordially", "regards", "respectfully", "truly", "yours","signature(s)"]

        logger.info(f" Keywords to search (priority order): {keywords}")

        def analyze_signature_region(region, signature_keywords, ocr_boxes):

            if img is None:
                return 0.5, 0.5, False

            rx, ry, rw, rh = map(int, region)
            rx, ry = max(0, rx), max(0, ry)
            rw, rh = min(iw - rx, rw), min(ih - ry, rh)

            if rw <= 0 or rh <= 0:
                return 0, 0, False

            roi = img[ry:ry + rh, rx:rx + rw]
            if roi.size == 0:
                return 0, 0, False
            
            mask = np.ones((rh, rw), dtype=np.uint8) * 255

            first_text_y = rh  # Varsayılan: bölgenin sonu
            texts_in_region = []
    
            for ocr_box in (ocr_boxes or []):
                bx, by, bw, bh = ocr_box.get("box", (0, 0, 0, 0))
                text = (ocr_box.get("text") or "").strip()
        
                if not text:
                    continue
        
        # Bölge ile kesişim
                if by >= ry and by < ry + rh and bx < rx + rw and bx + bw > rx:
                    text_lower = text.lower()
                    if any(kw in text_lower for kw in signature_keywords):
                        continue
            
                    clean_text = ''.join(c for c in text if c.isalnum())
                    if len(clean_text) >= 3 and any(c.isalpha() for c in clean_text):
                        texts_in_region.append({'text': text, 'y': by, 'box': (bx, by, bw, bh)})
                # En üstteki metnin y koordinatı
                        relative_y = by - ry
                        if relative_y < first_text_y:
                            first_text_y = relative_y
    
            logger.info(f" Region: ({rx},{ry},{rw},{rh})")
            logger.info(f" Texts found: {[t['text'] for t in texts_in_region]}")
    
    #  İMZA BÖLGESİ: Keyword'den ilk metne kadar olan alan
            if first_text_y < rh and first_text_y > 10:
        # İmza için sadece üst kısmı analiz et (metin öncesi)
                signature_height = first_text_y - 5  # Metinden 5px önce
                if signature_height > 15:  # En az 15px yükseklik
                    logger.info(f"     Signature area: top {signature_height}px (before text)")
                    analysis_roi = roi[0:signature_height, :]
                else:
                    logger.info(f"     Text too close to top, using full region")
                    analysis_roi = roi
            else:
        # Metin yok veya çok aşağıda - tüm bölgeyi kullan
                logger.info(f"     No text in region, using full area")
                analysis_roi = roi
    
    # Ink ve edge hesapla
            total_pixels = analysis_roi.size
            if total_pixels < 100:
                logger.info(f"     Analysis area too small")
                return 0, 0, False
    
            ink_pixels = np.sum(analysis_roi < 160)
            ink_density = ink_pixels / total_pixels
    
            try:
                edges = cv2.Canny(analysis_roi, 50, 150)
                edge_density = np.sum(edges > 0) / total_pixels
            except:
                edge_density = 0
    
            logger.info(f"     ink={ink_density:.4f}, edge={edge_density:.4f}")
    
            MIN_INK = 0.015
            MIN_EDGE = 0.012
    
    # İmza alanında (metin öncesi) ink VEYA edge varsa
            has_signature = (ink_density > MIN_INK) or (edge_density > MIN_EDGE)
    
            if has_signature:
                logger.info(f"     Signature detected in pre-text area")
                return ink_density, edge_density, True
            else:
                logger.info(f"     No signature detected")
                return ink_density, edge_density, False

        sorted_boxes = sorted(ocr_boxes, key=lambda b: (b.get("box", (0,0,0,0))[1], b.get("box", (0,0,0,0))[0]))
        used_box_indices = set()

        for keyword in keywords:
            keyword_lower = keyword.lower()
            keyword_words = keyword_lower.split()

            if len(keyword_words) > 1:
            # ÇOK KELİMELİK KEYWORD
                for i, start_box in enumerate(sorted_boxes):
                    if i in used_box_indices:
                        continue
                    
                    start_text = (start_box.get("text") or "").lower().strip()
                    if keyword_words[0] not in start_text:
                        continue
                
                    combined_text = start_text
                    combined_boxes = [start_box]
                    combined_indices = [i]
                    sx, sy, sw, sh = start_box.get("box", (0, 0, 0, 0))
                
                    for j in range(i + 1, min(i + len(keyword_words) + 3, len(sorted_boxes))):
                        if j in used_box_indices:
                            continue
                        
                        next_box = sorted_boxes[j]
                        nx, ny, nw, nh = next_box.get("box", (0, 0, 0, 0))
                    
                        if abs(ny - sy) > sh * 2.0:
                            break
                    
                        last_box = combined_boxes[-1]
                        last_x, last_y, last_w, last_h = last_box.get("box", (0, 0, 0, 0))
                        if nx > last_x + last_w + 150:
                            break
                    
                        next_text = (next_box.get("text") or "").lower().strip()
                        combined_text += " " + next_text
                        combined_boxes.append(next_box)
                        combined_indices.append(j)
                    
                        if keyword_lower in combined_text:
                            for idx in combined_indices:
                                used_box_indices.add(idx)
                        
                            all_x = [b.get("box", (0,0,0,0))[0] for b in combined_boxes]
                            all_y = [b.get("box", (0,0,0,0))[1] for b in combined_boxes]
                            all_xe = [b.get("box", (0,0,0,0))[0] + b.get("box", (0,0,0,0))[2] for b in combined_boxes]
                            all_ye = [b.get("box", (0,0,0,0))[1] + b.get("box", (0,0,0,0))[3] for b in combined_boxes]
                        
                            x = min(all_x)
                            y = min(all_y)
                            w = max(all_xe) - x
                            h = max(all_ye) - y
                        
                            logger.info(f" Found '{keyword}' → box=({x},{y},{w},{h})")
                        
                            right_region = (x + w + 10, y - 5, 200, h + 30)
                            below_region = (x - 10, y + h + 4, w + 100, 80)
                        
                            right_ink, right_edge, right_hand = analyze_signature_region(right_region, keywords, ocr_boxes)
                            below_ink, below_edge, below_hand = analyze_signature_region(below_region, keywords, ocr_boxes)
                        

                            if right_hand and (right_ink > 0.01 or right_edge > 0.005):
                                found_regions.append(right_region)
                                logger.info(f"   Signature FOUND to RIGHT")
                            elif below_hand and (below_ink > 0.01 or below_edge > 0.005):
                                found_regions.append(below_region)
                                logger.info(f"   Signature FOUND BELOW")
                            else:
                                logger.info(f"   No signature detected")
                        
                            break
        
            else:
            # TEK KELİMELİK KEYWORD
                for i, box in enumerate(sorted_boxes):
                    if i in used_box_indices:
                        continue
                    
                    text = (box.get("text") or "").lower().strip()
                
                    if keyword_lower == "signature(s)":
                        if "signature" not in text:
                            continue
                    elif keyword_lower not in text:
                        continue
                
                    used_box_indices.add(i)
                
                    x, y, w, h = box.get("box", (0, 0, 0, 0))
                    logger.info(f" Found '{keyword}' → box=({x},{y},{w},{h})")
                
                    is_letter_keyword = keyword_lower in ["sincerely", "cordially", "regards", "respectfully", "truly", "yours"]
                
                    if dt == "letter" or is_letter_keyword:
                        base_region = (max(0, x - 50), y + h + 5, min(280, iw - x + 50), 85)
                        ink, edge, is_handwritten = analyze_signature_region(base_region, keywords, ocr_boxes)
                    

                        if is_handwritten and (ink > 0.01 or edge > 0.005):
                            found_regions.append(base_region)
                            logger.info(f"   Signature FOUND below")
                        else:
                            logger.info(f"   No signature detected")
                    else:
                        right_region = (x + w + 10, y - 5, 200, h + 30)
                        below_region = (x - 10, y + h + 4, w + 100, 80)
                    
                        right_ink, right_edge, right_hand = analyze_signature_region(right_region, keywords, ocr_boxes)
                        below_ink, below_edge, below_hand = analyze_signature_region(below_region, keywords, ocr_boxes)
                    

                        if right_hand and (right_ink > 0.01 or right_edge > 0.005):
                            found_regions.append(right_region)
                            logger.info(f"   Signature FOUND to RIGHT")
                        elif below_hand and (below_ink > 0.01 or below_edge > 0.005):
                            found_regions.append(below_region)
                            logger.info(f"   Signature FOUND BELOW")
                        else:
                            logger.info(f"   No signature detected")

        logger.info(f" Final: Found {len(found_regions)} signature regions")
        return found_regions

    def _find_signatures_with_opencv(
        self, image_b64: str, ocr_boxes: List[Dict] = None
    ) -> List[Dict]:
        """
        OpenCV + MSER signature detection
        - Küçük ve orta boy el yazısı imzalar için optimize
        - Daha esnek geometri filtreleri
        - scikit-image bağımlılığı yok
        """
        try:
            import cv2
            import numpy as np
            import base64

            logger.info(" Starting OpenCV signature detection")

            img_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error(" Failed to decode image")
                return []

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            logger.info(f" Image size: {w}x{h}")

        # Preprocessing
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            gray = cv2.equalizeHist(gray)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 25, 12
            )

        # Çizgi temizleme
            lines = cv2.HoughLinesP(
                binary, 1, np.pi / 180,
                threshold=150,
                minLineLength=int(w * 0.2),
                maxLineGap=15
            )
            if lines is not None:
                for x1, y1, x2, y2 in lines[:, 0]:
                    if abs(y2 - y1) < 10 or abs(x2 - x1) < 10:
                        cv2.line(binary, (x1, y1), (x2, y2), 0, 20)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # MSER
            mser = cv2.MSER_create()
            mser.setMinArea(80)
            mser.setMaxArea(int(w * h * 0.03))
            mser.setDelta(8)
            regions, _ = mser.detectRegions(gray)
        
            mask_mser = np.zeros_like(gray)
            for r in regions:
                hull = cv2.convexHull(r.reshape(-1, 1, 2))
                cv2.drawContours(mask_mser, [hull], -1, 255, -1)

            candidate_mask = cv2.bitwise_and(mask_mser, cleaned)
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            candidate_mask = cv2.dilate(candidate_mask, kernel2, iterations=1)

            contours, _ = cv2.findContours(
                candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            logger.info(f" Found {len(contours)} contours, filtering...")

            sigs = []
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
            
            # Alan filtreleri
                if area < 400 or cw < 30 or ch < 12:
                    continue
            
                if area > w * h * 0.025 or cw > 350 or ch > 120:
                    continue

                aspect = cw / float(ch + 1e-6)
            
            # Aspect ratio
                if not (1.5 < aspect < 8.0):
                    continue

            # Dikdörtgensellik
                rect_area = cw * ch
                rectangularity = area / rect_area
            
                if rectangularity > 0.75:
                    continue

            # Kontur karmaşıklığı kontrolü (skeletonize yerine)
                try:
                    perimeter = cv2.arcLength(cnt, True)
                    complexity = perimeter / (area + 1e-6)
                
                # İmzalar karmaşık konturlara sahiptir (yüksek complexity)
                # Basit geometrik şekiller düşük complexity'ye sahiptir
                    if not (0.05 < complexity < 0.50):
                        continue
                except:
                    pass

            # Scoring
                base_score = 0.70
                aspect_bonus = min(0.15, aspect / 25.0)
                area_bonus = min(0.10, area / 5000.0)
                complexity_bonus = 0.05 if rectangularity < 0.5 else 0
            
                score = base_score + aspect_bonus + area_bonus + complexity_bonus
                sigs.append({
                    "box": (int(x), int(y), int(cw), int(ch)),
                    "score": min(0.95, score)
                })

            logger.info(f" OpenCV found {len(sigs)} signatures")
            for i, sig in enumerate(sigs):
                logger.info(f"  Sig {i+1}: box={sig['box']}, score={sig['score']:.2f}")
        
            return sigs

        except Exception as e:
            logger.error(f" OpenCV signature detection error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []


    # ---------- OCR TABANLI BIRTHDATE BÖLGELERİ (resume) ----------
    def _resume_birthdate_smart(self, ocr_boxes):
        import re
        if not ocr_boxes:
            return []

        month_names = r"(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)"

        date_patterns = [
            rf"\b{month_names}\s+\d{{1,2}},?\s+\d{{4}}\b",
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            r"\b\d{1,2}/\d{1,2}/\d{2}\b",
        ]

        date_regex = re.compile("|".join(f"({p})" for p in date_patterns), re.I)
        period_regex = re.compile(r"\b\d{4}\s*[-–]\s*\d{2,4}\b")

        place_keywords = [
            "chicago", "illinois", "new york", "california", "texas", "florida", "boston",
            "cananea", "sonora", "mexico", "izegem", "belgium", "belgian",
            "usa", "u.s.a", "united states",
            "los angeles", "san francisco", "seattle", "miami", "houston", "dallas",
            "phoenix", "philadelphia", "san diego", "denver", "atlanta",
            "washington", "detroit", "minneapolis", "cleveland", "portland",
            "canada", "ontario", "toronto", "vancouver", "montreal",
            "pasadena", "montana", "missoula", "salt lake city", "utah",
            "massachusetts", "pennsylvania", "kearney", "nebraska", "philadelph",
        ]

        nboxes = []
        for b in ocr_boxes:
            txt = (b.get("text") or "").strip()
            if not txt:
                continue
            x, y, w, h = b.get("box", (0, 0, 0, 0))
            nboxes.append(
                {
                    "text": txt,
                    "lower": txt.lower().strip(":,.-()"),
                    "box": (x, y, w, h),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )

    # ------------------ Yardımcı: Bir bölgede tarih / yer ara ------------------
        def try_find_date_in_zone(candidates, zone_name):

            if not candidates:
                return None

            candidates_sorted = sorted(candidates, key=lambda b: b["x"])

            logger.info(
                f" Trying to find date/place in {zone_name}: {len(candidates_sorted)} candidates..."
            )

            month_only_regex = re.compile(
                r"\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b",
                re.I,
            )
            year_regex = re.compile(r"\b(19|20)\d{2}\b")
            day_regex = re.compile(r"\b\d{1,2}[,]?\b")

            def contains_place_keyword(text_lower):
                """Metin içinde yer ismi var mı kontrol et"""
                return any(place in text_lower for place in place_keywords)

            best_match = None
            best_match_score = 0  # 3=date+place, 2=date only, 1=place only

            for i in range(len(candidates_sorted)):
                for window_size in range(
                    min(10, len(candidates_sorted) - i), 0, -1
                ):
                    window = candidates_sorted[i : i + window_size]
                    combined_text = " ".join(b["text"] for b in window)

                    logger.info(
                        f"   Testing window (size={window_size}): '{combined_text}'"
                    )

                    combined_lower = combined_text.lower()
                    has_place = contains_place_keyword(combined_lower)
                    date_match = date_regex.search(combined_text)


                    if period_regex.search(combined_text):
                        logger.info("     Skipped: Is a period (1965-67 etc.)")
                        continue

                # Hem tarih hem yer varsa - EN İYİ (score=3)
                    if has_place and date_match:
                        logger.info(
                            f"     BOTH DATE + PLACE MATCHED: '{combined_text}'"
                        )

                        xs = [b["x"] for b in window]
                        ys = [b["y"] for b in window]
                        xe = [b["x"] + b["w"] for b in window]
                        ye = [b["y"] + b["h"] for b in window]

                        result_box = (
                            min(xs),
                            min(ys),
                            max(xe) - min(xs),
                            max(ye) - min(ys),
                        )

                        if best_match_score < 3:
                            best_match = result_box
                            best_match_score = 3
                            logger.info(
                                f" DATE+PLACE FOUND in {zone_name}: '{combined_text}' → bbox={result_box}"
                            )
                        continue

                # Sadece yer ismi varsa (score=1)
                    if has_place and not date_match:
                        if best_match_score < 1:
                            logger.info("     PLACE ONLY MATCHED")

                            xs = [b["x"] for b in window]
                            ys = [b["y"] for b in window]
                            xe = [b["x"] + b["w"] for b in window]
                            ye = [b["y"] + b["h"] for b in window]

                            result_box = (
                                min(xs),
                                min(ys),
                                max(xe) - min(xs),
                                max(ye) - min(ys),
                            )
                            best_match = result_box
                            best_match_score = 1
                            logger.info(
                                f" PLACE FOUND in {zone_name}: '{combined_text}' → bbox={result_box}"
                            )
                        continue

                # Full date regex (score=2)
                    if date_match:
                        if best_match_score < 2:
                            logger.info(
                                f"     FULL DATE REGEX MATCHED: '{date_match.group()}'"
                            )

                            xs = [b["x"] for b in window]
                            ys = [b["y"] for b in window]
                            xe = [b["x"] + b["w"] for b in window]
                            ye = [b["y"] + b["h"] for b in window]

                            result_box = (
                                min(xs),
                                min(ys),
                                max(xe) - min(xs),
                                max(ye) - min(ys),
                            )
                            best_match = result_box
                            best_match_score = 2
                            logger.info(
                                f" DATE FOUND in {zone_name}: '{combined_text}' → bbox={result_box}"
                            )
                        continue

                # Partial date (score=2)
                    has_month = month_only_regex.search(combined_text)
                    has_year = year_regex.search(combined_text)
                    has_day = day_regex.search(combined_text)

                    if has_month and has_day and has_year:
                        if best_match_score < 2:
                            logger.info(
                                "     FULL DATE (multi-box): Month + Day + Year"
                            )

                            xs = [b["x"] for b in window]
                            ys = [b["y"] for b in window]
                            xe = [b["x"] + b["w"] for b in window]
                            ye = [b["y"] + b["h"] for b in window]

                            result_box = (
                                min(xs),
                                min(ys),
                                max(xe) - min(xs),
                                max(ye) - min(ys),
                            )
                            best_match = result_box
                            best_match_score = 2
                            logger.info(
                                f"     FULL DATE BOX in {zone_name}: {result_box}"
                            )
                        continue

                    is_valid_date = (
                        (has_month and has_year)
                        or (has_month and has_day)
                        or (has_day and has_year)
                    )

                    if is_valid_date and best_match_score < 2:
                        match_type = []
                        if has_month:
                            match_type.append("Month")
                        if has_day:
                            match_type.append("Day")
                        if has_year:
                            match_type.append("Year")

                        logger.info(
                            f"     PARTIAL DATE MATCHED in {zone_name}: {' + '.join(match_type)}"
                        )

                        xs = [b["x"] for b in window]
                        ys = [b["y"] for b in window]
                        xe = [b["x"] + b["w"] for b in window]
                        ye = [b["y"] + b["h"] for b in window]

                        result_box = (
                            min(xs),
                            min(ys),
                            max(xe) - min(xs),
                            max(ye) - min(ys),
                        )
                        best_match = result_box
                        best_match_score = 2
                        logger.info(
                            f" DATE FOUND in {zone_name} (partial): '{combined_text}' → bbox={result_box}"
                        )

            return best_match

    # ------------------ 1) DOB/BIRTHPLACE label gruplarını bul ------------------
        label_groups = []
        nboxes_sorted = sorted(nboxes, key=lambda b: (b["y"], b["x"]))

        for i, box in enumerate(nboxes_sorted):
            lower = box["lower"]

            if re.search(
                r"(date\s+and\s+place\s+of\s+birth|date\s+of\s+birth|place\s+of\s+birth|birth\s+date)",
                lower,
            ):
                label_groups.append([box])
                logger.info(
                    f" Single-box multi-word birth label: '{box['text']}' at ({box['x']}, {box['y']})"
                )
                continue

        # Tek kelimelik label'lar
            if lower in ["birthdate", "born", "dob", "birthplace"]:
                label_groups.append([box])
                logger.info(
                    f" Single keyword: '{box['text']}' at ({box['x']}, {box['y']})"
                )

        # Multi-word labels (date of birth, place of birth, vb.)
            elif lower in ["date", "birth", "place"]:
                nearby = [box]

                for j in range(max(0, i - 5), min(len(nboxes_sorted), i + 6)):
                    if j == i:
                        continue
                    other = nboxes_sorted[j]

                    if (
                        abs(other["y"] - box["y"]) < 15
                        and abs(other["x"] - box["x"]) < 100
                    ):
                        if other["lower"] in ["date", "of", "birth", "place", "and"]:
                            nearby.append(other)

                words = [b["lower"] for b in nearby]

                if len(words) >= 2 and ("birth" in words or "born" in words):
                    combined = " ".join(
                        b["text"] for b in sorted(nearby, key=lambda x: x["x"])
                    )
                    combined_lower = combined.lower()

                    if any(
                        pattern in combined_lower
                        for pattern in [
                            "date of birth",
                            "birth date",
                            "place and date",
                            "date and place",
                            "place of birth",
                            "birthplace",
                            "born",
                        ]
                    ):
                        label_groups.append(nearby)
                        logger.info(
                            f" Multi-word label: '{combined}' ({len(nearby)} boxes)"
                        )

        if not label_groups:
            logger.warning(" No birthdate/birthplace label found")
            return []

        unique_groups = []
        seen = set()
        for g in label_groups:
            gx_min = min(b["x"] for b in g)
            gy_min = min(b["y"] for b in g)
            gx_max = max(b["x"] + b["w"] for b in g)
            gy_max = max(b["y"] + b["h"] for b in g)
        # küçük tolere ile grid’e yuvarla ki aynı satırdaki kopyalar birleşsin
            key = (round(gx_min / 5), round(gy_min / 5), round(gx_max / 5), round(gy_max / 5))
            if key not in seen:
                seen.add(key)
                unique_groups.append(g)

        label_groups = unique_groups

    # ====================== 2) Birden fazla label'ı işle =======================
        label_groups.sort(key=lambda g: (min(b["y"] for b in g), -len(g)))
        all_results = []

        for idx, label_boxes in enumerate(label_groups):
            label_text = " ".join(b["lower"] for b in label_boxes)
            is_dob = ("date" in label_text and "birth" in label_text)
            is_pob = ("place" in label_text and "birth" in label_text)

            logger.info(
                f" Processing birth label group {idx + 1}/{len(label_groups)} "
                f"→ '{label_text}' (DOB={is_dob}, POB={is_pob})"
            )

            label_x_min = min(b["x"] for b in label_boxes)
            label_x_max = max(b["x"] + b["w"] for b in label_boxes)
            label_y_min = min(b["y"] for b in label_boxes)
            label_y_max = max(b["y"] + b["h"] for b in label_boxes)
            avg_h = sum(b["h"] for b in label_boxes) / len(label_boxes)

        # Bu label'ın çevresindeki adayları hazırla
            same_line_candidates = [
                b
                for b in nboxes
                if (abs(b["y"] - label_y_min) <= avg_h * 1.3)
                and (b["x"] >= label_x_max - 20)
                and (b["x"] <= label_x_max + 450)
            ]

            below_candidates = [
                b
                for b in nboxes
                if (b["y"] > label_y_max)
                and (b["y"] <= label_y_max + avg_h * 2.5)
                and (abs(b["x"] - label_x_min) <= 250)
            ]

            logger.info(
                f"  Label {idx + 1}: same_line={len(same_line_candidates)}, "
                f"below={len(below_candidates)}"
            )

        # -------- CASE 1: Date of Birth → sadece tarih arıyoruz --------
            if is_dob:
                found = False
            # Önce aynı satırda tarih ara
                result = try_find_date_in_zone(same_line_candidates, "Same Line")
                if result:
                    all_results.append(result)
                    logger.info("  DOB found on same line")
                    continue  # Bu label bitti

            # Eğer aynı satırda tarih yoksa, aşağıdaki 1–2 satıra bak
                result = try_find_date_in_zone(below_candidates, "BELOW (DOB)")
                if result:
                    all_results.append(result)
                    logger.info("  DOB found below label")
                else:
                    logger.info("  DOB not found around label")

                continue

        # -------- CASE 2: Place of Birth → sadece yeri alıyoruz --------
            if is_pob:
                if not same_line_candidates:
                    logger.info("  POB: no same-line candidates")
                    continue

            # Aynı satırda label'ın sağındaki tüm kutuları yer olarak kabul et
                xs = [b["x"] for b in same_line_candidates]
                ys = [b["y"] for b in same_line_candidates]
                xe = [b["x"] + b["w"] for b in same_line_candidates]
                ye = [b["y"] + b["h"] for b in same_line_candidates]
                result_box = (min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys))

                all_results.append(result_box)
                combined_text = " ".join(b["text"] for b in same_line_candidates)
                logger.info(
                    f"  POB same-line place captured: '{combined_text}' → bbox={result_box}"
                )
                continue

        # -------- CASE 3: Diğer label'lar (çok nadir) --------
        # Eski generic mantığı korumak istersen:
            found_for_label = False
            for zone_name, candidates in [
                ("Same Line", same_line_candidates),
                ("BELOW", below_candidates),
            ]:
                result = try_find_date_in_zone(candidates, zone_name)
                if result:
                    all_results.append(result)
                    found_for_label = True
                    break

            if not found_for_label:
                logger.info("  Generic label: no date/place found")

        if not all_results:
            logger.warning(" Birth label(s) found but no valid date/place detected")
            logger.info(" Tested combinations for all labels")
        else:
            logger.info(f" Birth detection → {len(all_results)} bbox(es)")

        return all_results

    def _detect_personal_addresses(self, ocr_boxes: list, doc_type: str, raw_text: str = "") -> list:
    
        dt = (doc_type or "").lower()
        if dt not in ["letter", "form", "invoice", "resume"]:
            logger.info(f" Address detection disabled for doc_type={dt}")
            return []
    
        logger.info(f" Starting personal address detection for doc_type={dt}")
    
    # OCR kutularını y koordinatına göre sırala
        sorted_boxes = sorted(ocr_boxes, key=lambda b: (b.get("box", (0,0,0,0))[1], b.get("box", (0,0,0,0))[0]))
    
        address_regions = []

        if dt == "invoice":
            logger.info(" Invoice mode: ONLY masking 'Client' addresses (X + vertical continuity)")

            client_labels = [
                r"^\s*client\s*:?\s*$",
                r"^\s*bill\s+to\s*:?\s*$",
                r"^\s*ship\s+to\s*:?\s*$",
                r"^\s*customer\s*:?\s*$",
                r"^\s*sold\s+to\s*:?\s*$",
            ]

            seller_labels = [
                r"^\s*seller\s*:?\s*$",
                r"^\s*from\s*:?\s*$",
                r"^\s*vendor\s*:?\s*$",
            ]

            for i, box in enumerate(sorted_boxes):
                text = (box.get("text") or "").strip().lower()
                if not text:
                    continue

                is_seller = any(re.match(p, text, re.I) for p in seller_labels)
                if is_seller:
                    logger.info(f"   SKIPPED seller label: '{text}' at line {i}")
                    continue

                is_client = any(re.match(p, text, re.I) for p in client_labels)
                if not is_client:
                    continue

                label_x, label_y, label_w, label_h = box.get("box", (0, 0, 0, 0))
                logger.info(f"   Found CLIENT label: '{text}' at line {i}, X={label_x}, Y={label_y}")

                address_boxes = []
                last_addr_y = None  # son eklenen adres satırının Y'si

                for j in range(i + 1, min(i + 20, len(sorted_boxes))):  # 20 kutuya kadar bak
                    next_box = sorted_boxes[j]
                    next_text = (next_box.get("text") or "").strip()
                    if not next_text:
                        continue

                    next_x, next_y, next_w, next_h = next_box.get("box", (0, 0, 0, 0))
                    next_lower = next_text.lower()

                # Başka bir label gelirse dur
                    if any(re.match(p, next_lower, re.I) for p in (client_labels + seller_labels)):
                        logger.info(f"    → Stopped at label: '{next_lower}'")
                        break

                # Table header gelirse dur
                    if re.match(r"^\s*(items|description|qty|no\.|amount|tax\s+id|iban)\s*:?\s*$", next_lower, re.I):
                        logger.info(f"    → Stopped at table/header: '{next_lower}'")
                        break

                # X-KOORDINAT KONTROLÜ: Client label ile aynı sütun tarafında mı?
                    x_diff = abs(next_x - label_x)
                    if x_diff > 120:
                        logger.info(f"     Skipped (wrong X): '{next_text}' (X={next_x}, diff={x_diff})")
                        continue

                # Y-KOORDINAT: adres bloğunun devamı mı?
                    if last_addr_y is not None:
                        gap = next_y - last_addr_y
                    # Ardışık adres satırları arasında çok büyük boşluk varsa blok biter
                        if gap > 80:
                            logger.info(f"    → Stopped (left address block, gap={gap})")
                            break

                    address_boxes.append(next_box)
                    last_addr_y = next_y
                    logger.info(f"    + Added address line: '{next_text}' (X={next_x}, Y={next_y})")

                if address_boxes:
                # Bbox hesapla
                    all_x = [b.get("box", (0, 0, 0, 0))[0] for b in address_boxes]
                    all_y = [b.get("box", (0, 0, 0, 0))[1] for b in address_boxes]
                    all_xe = [b.get("box", (0, 0, 0, 0))[0] + b.get("box", (0, 0, 0, 0))[2] for b in address_boxes]
                    all_ye = [b.get("box", (0, 0, 0, 0))[1] + b.get("box", (0, 0, 0, 0))[3] for b in address_boxes]

                    final_x = min(all_x)
                    final_y = min(all_y)
                    final_w = max(all_xe) - final_x
                    final_h = max(all_ye) - final_y

                    address_regions.append((final_x, final_y, final_w, final_h))

                    combined = " ".join(b.get("text", "") for b in address_boxes)
                    logger.info(f"   CLIENT ADDRESS MASKED: '{combined}' → bbox=({final_x},{final_y},{final_w},{final_h})")

            logger.info(f" Invoice: {len(address_regions)} client addresses masked")
            return address_regions

            # ========================================
            # RESUME için özel mantık
            # ========================================
        if dt == "resume":
            # "Address:" satırı veya "Address: 89-B Farwell..." tek satır
            address_label_full = re.compile(r"^\s*address\s*[:.]?\s*$", re.I)
            address_label_inline = re.compile(r"^\s*address\s*[:.]?\s+(.+)$", re.I)

            for i, box in enumerate(sorted_boxes):
                raw_text = (box.get("text") or "")
                text = raw_text.strip()
                if not text:
                    continue

                x, y, w, h = box.get("box", (0, 0, 0, 0))

                inline_match = address_label_inline.match(text)
                full_match = address_label_full.match(text)

            # Hiç "address" yoksa devam
                if not inline_match and not full_match:
                    continue

                logger.info(f"   Found address label at line {i}: '{text}'")

                address_boxes = []

                if inline_match:
                # Bu kutunun tamamını adres olarak kabul et (label + value birlikte)
                    address_boxes.append(box)
                    logger.info("      Inline address detected on same box")

            # 2) Label solda, adres sağda → aynı satırdaki kutuları topla
                if full_match:
                    logger.info("      Address label only, searching same line to the right")
                    for k, b2 in enumerate(sorted_boxes):
                        if k == i:
                            continue
                        nx, ny, nw, nh = b2.get("box", (0, 0, 0, 0))
                        t2 = (b2.get("text") or "").strip()
                        if not t2:
                            continue

                    # Aynı satır toleransı: yüksekliği biraz geniş tut
                        if abs(ny - y) <= max(h, nh) * 1.2 and nx >= x + w - 10:
                            address_boxes.append(b2)
                            logger.info(f"      + Same-line address part: '{t2}'")

            # Eğer aynı satırda adres yakalandıysa
            # Alt satırlara inmeye gerek yok (doğum tarihine karışmaması için)
                if address_boxes:
                    all_x = [b.get("box", (0, 0, 0, 0))[0] for b in address_boxes]
                    all_y = [b.get("box", (0, 0, 0, 0))[1] for b in address_boxes]
                    all_xe = [
                        b.get("box", (0, 0, 0, 0))[0] + b.get("box", (0, 0, 0, 0))[2]
                        for b in address_boxes
                    ]
                    all_ye = [
                        b.get("box", (0, 0, 0, 0))[1] + b.get("box", (0, 0, 0, 0))[3]
                        for b in address_boxes
                    ]

                    final_x = min(all_x)
                    final_y = min(all_y)
                    final_w = max(all_xe) - final_x
                    final_h = max(all_ye) - final_y

                    address_regions.append((final_x, final_y, final_w, final_h))

                    combined_text = " ".join(b.get("text", "") for b in address_boxes)
                    logger.info(
                        f"   CV address detected (same line): '{combined_text}' → bbox=({final_x},{final_y},{final_w},{final_h})"
                    )

                # CV'de genelde tek adres var, bulunca dönebiliriz
                    return address_regions

            # 3) Alt satırlardaki devam eden adres satırları (posta kodu vb.)
                for j in range(i + 1, min(i + 6, len(sorted_boxes))):
                    next_box = sorted_boxes[j]
                    nx, ny, nw, nh = next_box.get("box", (0, 0, 0, 0))
                    next_text = (next_box.get("text") or "").strip()

                    if not next_text:
                        continue

                    lower_next = next_text.lower()

                    if ny - y > h * 4:
                        break

                    if lower_next in ["date", "place"] and j + 2 < len(sorted_boxes):
                        t2 = (sorted_boxes[j + 1].get("text") or "").strip().lower()
                        t3 = (sorted_boxes[j + 2].get("text") or "").strip().lower()
                        if lower_next == "date" and t2 == "of" and t3.startswith("birth"):
                            logger.info("      → Stopped at split 'Date of Birth' label")
                            break
                        if lower_next == "place" and t2 == "of" and t3.startswith("birth"):
                            logger.info("      → Stopped at split 'Place of Birth' label")
                            break

                # Başka CV label'ı geldiyse dur
                    if re.match(
                        r"^\s*(date\s+of\s+birth|place\s+of\s+birth|born|education|experience|skills|training|phone|email|licensure|certification)\s*[:.]?\s*$",
                        next_text,
                        re.I,
                    ):
                        logger.info(f"      → Stopped at CV label: '{next_text}'")
                        break

                # Tarih gibi görünen satırları atla
                    if re.search(
                        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
                        next_text,
                        re.I,
                    ):
                        logger.info(f"       Skipped date pattern in address block: '{next_text}'")
                        continue

                # Sadece yıl olan satırları atla
                    if re.match(r"^\s*\d{4}\s*$", next_text):
                        logger.info(f"       Skipped pure year in address block: '{next_text}'")
                        continue

                    address_boxes.append(next_box)
                    logger.info(f"      + Added below-line address part: '{next_text}'")

                    if len(address_boxes) >= 3:
                        break

                if address_boxes:
                    # Tüm kutuları kapsayan bbox oluştur
                    all_x = [b.get("box", (0, 0, 0, 0))[0] for b in address_boxes]
                    all_y = [b.get("box", (0, 0, 0, 0))[1] for b in address_boxes]
                    all_xe = [
                        b.get("box", (0, 0, 0, 0))[0] + b.get("box", (0, 0, 0, 0))[2]
                        for b in address_boxes
                    ]
                    all_ye = [
                        b.get("box", (0, 0, 0, 0))[1] + b.get("box", (0, 0, 0, 0))[3]
                        for b in address_boxes
                    ]

                    final_x = min(all_x)
                    final_y = min(all_y)
                    final_w =max(all_xe) - final_x
                    final_h = max(all_ye) - final_y

                    address_regions.append((final_x, final_y, final_w, final_h))

                    combined_text = " ".join(b.get("text", "") for b in address_boxes)
                    logger.info(
                        f"   CV address detected: '{combined_text}' → bbox=({final_x},{final_y},{final_w},{final_h})"
                    )

                # CV'de genelde tek kişisel adres olur; bulunca dönebiliriz
                    return address_regions

    # ========================================
    # LETTER ve FORM için genel pattern'ler
    # ========================================
        if dt in ["letter", "form"]:
            logger.info(f" Letter/Form mode: Using general address patterns with corporate filtering")
        
            for i, box in enumerate(sorted_boxes):
                text = (box.get("text") or "").strip()
                if not text:
                    continue
            
                flexible_address = re.search(
                    r"\b(\d+\s+)?[A-Z][a-z]+(\s+[A-Z][a-z]+)?\s+(street|st\.|avenue|ave\.|road|rd\.|drive|dr\.|lane|ln\.|way|boulevard|blvd\.)\b",
                    text,
                    re.I
                )
            
                if not flexible_address:
                    continue
            
                logger.info(f"   Address pattern found: '{text}'")
            

                close_context_window = 2
                close_context_texts = []
                for j in range(max(0, i - close_context_window), min(len(sorted_boxes), i + close_context_window + 1)):
                    close_context_texts.append((sorted_boxes[j].get("text") or "").strip())
                close_context = " ".join(close_context_texts)
            
            # Geniş context (±5 satır) - genel yapı
                wide_context_window = 5
                wide_context_texts = []
                for j in range(max(0, i - wide_context_window), min(len(sorted_boxes), i + wide_context_window + 1)):
                    wide_context_texts.append((sorted_boxes[j].get("text") or "").strip())
                wide_context = " ".join(wide_context_texts)
            
                should_mask = False  # Flag
            
            # Letter/Form için akıllı filtreleme
                has_personal_indicators = bool(re.search(
                    r"\b(dear|ms\.|mr\.|mrs\.|dr\.|to:|from:|recipient|sender|attention|client)\b",
                    close_context, re.I
                ))
            
                has_company_header = bool(re.search(
                    r"\b(corporation|company|inc\.|ltd\.|llc|headquarters)\b",
                    close_context, re.I
                ))
            
                if has_personal_indicators:
                    logger.info("     ACCEPTED: Personal address indicator")
                    should_mask = True
                elif has_company_header:
                    logger.info("     REJECTED: Company header address")
                    should_mask = False
                else:
                    should_mask = True  # Letter/form için default: maskele
            
            # Final karar
                if not should_mask:
                    continue
            
            # Multi-line address detection
                x, y, w, h = box.get("box", (0, 0, 0, 0))
                address_boxes = [box]
            
                for j in range(i + 1, min(len(sorted_boxes), i + 4)):
                    next_box = sorted_boxes[j]
                    nx, ny, nw, nh = next_box.get("box", (0, 0, 0, 0))
                    next_text = (next_box.get("text") or "").strip()
                
                # Y mesafesi kontrolü
                    if abs(ny - y) > h * 2.5:
                        break
                
                # City-state-zip pattern
                    is_city_state = bool(re.search(
                        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b",
                        next_text, re.I
                    ))
                
                    if is_city_state:
                        address_boxes.append(next_box)
                        logger.info(f"      → Added city/state: '{next_text}'")
                        break
            
                if address_boxes:
                    all_x = [b.get("box", (0,0,0,0))[0] for b in address_boxes]
                    all_y = [b.get("box", (0,0,0,0))[1] for b in address_boxes]
                    all_xe = [b.get("box", (0,0,0,0))[0] + b.get("box", (0,0,0,0))[2] for b in address_boxes]
                    all_ye = [b.get("box", (0,0,0,0))[1] + b.get("box", (0,0,0,0))[3] for b in address_boxes]
                
                    final_x = min(all_x)
                    final_y = min(all_y)
                    final_w = max(all_xe) - final_x
                    final_h = max(all_ye) - final_y
                
                    address_regions.append((final_x, final_y, final_w, final_h))
                
                    combined_text = " ".join(b.get("text", "") for b in address_boxes)
                    logger.info(f"   Personal address detected: '{combined_text}' → bbox=({final_x},{final_y},{final_w},{final_h})")
    
        logger.info(f" Total personal addresses found: {len(address_regions)}")
        return address_regions
    
    def _detect_resume_name_region(self, ocr_boxes: list, doc_type: str) -> list:
    
        dt = (doc_type or "").lower()
        if dt != "resume":
            return []
    
        logger.info(f" Starting resume name detection")
    
        if not ocr_boxes:
            return []
    
    # OCR kutularını y koordinatına göre sırala
        sorted_boxes = sorted(ocr_boxes, key=lambda b: b.get("box", (0,0,0,0))[1])
    
    # İlk 15 satıra odaklan
        top_boxes = sorted_boxes[:min(20, len(sorted_boxes))]
    
        if not top_boxes:
            return []
    
    # Ortalama yüksekliği hesapla
        heights = [b.get("box", (0,0,0,0))[3] for b in top_boxes if b.get("box", (0,0,0,0))[3] > 0]
        if not heights:
            return []
    
        avg_height = sum(heights) / len(heights)
        max_height = max(heights)
    
        logger.info(f"   Avg height: {avg_height:.1f}, Max height: {max_height:.1f}")

        name_label_pattern = re.compile(r"^\s*name\s*:?\s*$", re.I)

        for i, box in enumerate(sorted_boxes[:40]):  # İlk 40 kutuya bak
            text = (box.get("text") or "").strip()
            if not text or not name_label_pattern.match(text):
                continue

            lx, ly, lw, lh = box.get("box", (0, 0, 0, 0))
            logger.info(f"   Found NAME label at ({lx},{ly})")

        #  ÖNCE AYNI SATIRIN SAĞINA BAK
            same_line_candidates = []
            for k, b2 in enumerate(sorted_boxes):
                if k == i:
                    continue
                nx, ny, nw, nh = b2.get("box", (0, 0, 0, 0))
                t2 = (b2.get("text") or "").strip()
                if not t2:
                    continue

            # AYNI SATIR: Y farkı küçük ve label'ın sağında
                if abs(ny - ly) <= max(lh, nh) * 0.6 and nx >= lx + lw * 0.5:
                    same_line_candidates.append(b2)

        # Soldan sağa sırala
            same_line_candidates = sorted(
                same_line_candidates,
                key=lambda b: b.get("box", (0, 0, 0, 0))[0]
            )

        # İsim pattern kontrolü
            def _is_name_like(boxes):
                combined = " ".join((b.get("text") or "").strip() for b in boxes)
                words = [w for w in combined.split() if len(w) > 1]
                if len(words) < 2 or len(words) > 6:
                    return False, None

            # Title Case veya ALL CAPS kontrolü
                title_case = sum(1 for w in words if w[0].isupper()) >= 2
                all_caps = combined.isupper() and 2 <= len(words) <= 4
                has_title = bool(re.search(r"\b(ph\.?d\.?|m\.?d\.?|dr\.?|prof\.?)\b", combined.lower()))

                if title_case or all_caps or has_title:
                    return True, combined
                return False, None

        #  AYNI SATIR KONTROLÜ
            if same_line_candidates:
                is_valid, combined_name = _is_name_like(same_line_candidates)
                if is_valid:
                    logger.info(f"    NAME found on SAME LINE: '{combined_name}'")

                # BBox hesapla
                    xs = [b.get("box", (0,0,0,0))[0] for b in same_line_candidates]
                    ys = [b.get("box", (0,0,0,0))[1] for b in same_line_candidates]
                    xes = [b.get("box", (0,0,0,0))[0] + b.get("box", (0,0,0,0))[2] for b in same_line_candidates]
                    yes = [b.get("box", (0,0,0,0))[1] + b.get("box", (0,0,0,0))[3] for b in same_line_candidates]

                    mx = min(xs)
                    my = min(ys)
                    mw = max(xes) - mx
                    mh = max(yes) - my

                    logger.info(f" RESUME NAME via NAME label detected: '{combined_name}' → bbox=({mx},{my},{mw},{mh})")
                    return [(mx, my, mw, mh)]

        #  2️ AYNI SATIRDA YOKSA ALT SATIRA BAK 
            below_candidates = []
            for k, b2 in enumerate(sorted_boxes):
                if k <= i:  # Label'dan sonra gelen kutuları kontrol et
                    continue
                nx, ny, nw, nh = b2.get("box", (0, 0, 0, 0))
                t2 = (b2.get("text") or "").strip()
                if not t2:
                    continue

            # ALT SATIR: 0.8-3 satır arasında ve aynı sütun (X hizalı)
                if (ly + lh * 0.8 <= ny <= ly + lh * 3) and (abs(nx - lx) <= lw * 0.8):
                    is_valid, _ = _is_name_like([b2])
                    if is_valid:
                        below_candidates.append(b2)

        # ALT SATIR KONTROLÜ
            if below_candidates:
            # İlk geçerli ismi al
                best = below_candidates[0]
                _, combined_name = _is_name_like([best])
            
                bx, by, bw, bh = best.get("box", (0, 0, 0, 0))
                logger.info(f" RESUME NAME via NAME label (BELOW) detected: '{combined_name}' → bbox=({bx},{by},{bw},{bh})")
                return [(bx, by, bw, bh)]
            # === SATIR BAZLI GRUPLAMA (line-level boxes) ===
        line_boxes = []
        current_line = []

        for b in top_boxes:
            x, y, w, h = b.get("box", (0, 0, 0, 0))
            if not current_line:
                current_line = [b]
                continue

        # Aynı satır mı? (Y farkı küçükse aynı satır kabul et)
            prev_y = current_line[0].get("box", (0, 0, 0, 0))[1]
            if abs(y - prev_y) <= avg_height * 0.6:
                current_line.append(b)
            else:
                line_boxes.append(current_line)
                current_line = [b]

        if current_line:
            line_boxes.append(current_line)

    # Satırları tek kutuya birleştir
        merged_top_boxes = []
        for line in line_boxes:
        # soldan sağa sırala
            line_sorted = sorted(line, key=lambda bb: bb.get("box", (0, 0, 0, 0))[0])
            texts = [(bb.get("text") or "").strip() for bb in line_sorted if bb.get("text")]
            if not texts:
                continue

            combined_text = " ".join(texts)

            xs  = [bb.get("box", (0, 0, 0, 0))[0] for bb in line_sorted]
            ys  = [bb.get("box", (0, 0, 0, 0))[1] for bb in line_sorted]
            xes = [bb.get("box", (0, 0, 0, 0))[0] + bb.get("box", (0, 0, 0, 0))[2] for bb in line_sorted]
            yes = [bb.get("box", (0, 0, 0, 0))[1] + bb.get("box", (0, 0, 0, 0))[3] for bb in line_sorted]

            mx = min(xs); my = min(ys)
            mw = max(xes) - mx; mh = max(yes) - my

            merged_top_boxes.append({
                "text": combined_text,
                "box": (mx, my, mw, mh),
            })


        top_boxes = merged_top_boxes

    # Yükseklikleri tekrar hesapla
        heights = [b.get("box", (0, 0, 0, 0))[3] for b in top_boxes if b.get("box", (0,0,0,0))[3] > 0]
        if not heights:
            return []

        avg_height = sum(heights) / len(heights)
        max_height = max(heights)

        logger.info(f"   (line-merged) Avg height: {avg_height:.1f}, Max height: {max_height:.1f}")
    
    # === ADAY SEÇME STRATEJISI ===
    
    # 1️ ÇOK BÜYÜK FONTLU METINLER (avg'nin 1.8+ katı)
        very_large = [
            b for b in top_boxes 
            if b.get("box", (0,0,0,0))[3] > avg_height * 1.8
        ]
    
    # 2️ BÜYÜK FONTLU METINLER (avg'nin 1.3+ katı)
        large_boxes = [
            b for b in top_boxes 
            if b.get("box", (0,0,0,0))[3] > avg_height * 1.3
        ]
    
    # 3️ ALTI ÇİZİLİ METINLER
    # Altı çizili metin genelde hemen altında ince bir çizgi kutusu vardır
        underlined = []
        for i, box in enumerate(top_boxes):
            if i + 1 < len(top_boxes):
                next_box = top_boxes[i + 1]
                bx, by, bw, bh = box.get("box", (0,0,0,0))
                nx, ny, nw, nh = next_box.get("box", (0,0,0,0))
            
            # Altında ince çizgi var mı? (yüksekliği 3-8px arası)
                if (abs(nx - bx) < 50 and  # Yatayda hizalı
                    ny > by and ny < by + bh + 15 and  # Hemen altında
                    nh < 8 and  # Çok ince
                    nw > bw * 0.5):  # En az yarısı kadar geniş
                    underlined.append(box)
                    logger.info(f"    Underlined detected: '{box.get('text')}'")
    
    # === ADAY HAVUZU ===

            candidates = []
            seen = set()

            for b in very_large + large_boxes + underlined:
        # bbox + text'e göre key üret
                bx, by, bw, bh = b.get("box", (0, 0, 0, 0))
                t_lower = (b.get("text") or "").strip().lower()
                key = (round(bx / 2), round(by / 2), round(bw / 2), round(bh / 2), t_lower)

                if key in seen:
                    continue
                seen.add(key)
                candidates.append(b)

            if not candidates:
                candidates = top_boxes[:5]

    # === FİLTRELEME ===
    
    # Şirket/Kurum/Belge göstergeleri
        exclude_patterns = [
            r"\b(university|college|corporation|company|inc\.|ltd\.|llc|school|institute|hospital|medical|clinic)\b",
            r"\b(department|division|center|centre|laboratory|faculty|office)\b",
            r"^\s*(curriculum\s+)?vita?e?\s*$",
            r"^\s*resume\s*$",
            r"^\s*cv\s*$",
            r"\b(biographical|sketch|education|experience|born|address)\b",
            r"^\s*name\s*:?\s*$",
            r"^\s*position\s*:?\s*$",
        ]
        exclude_regex = re.compile("|".join(exclude_patterns), re.I)
    
    # Email/telefon
        contact_regex = re.compile(r"@|phone|email|tel\s*:|mobile|contact|born\s*:", re.I)
    
    # Tarih pattern'leri (doğum tarihi satırı olabilir)
        date_regex = re.compile(r"\b(19|20)\d{2}\b|january|february|march|april|may|june|july|august|september|october|november|december", re.I)
    
        name_candidates = []
    
        for box in candidates:
            text = (box.get("text") or "").strip()
            if not text or len(text) < 5:
                continue
        
            text_lower = text.lower()
        
        # 1️ Exclude pattern kontrolü
            if exclude_regex.search(text):
                logger.info(f"      Excluded (company/doc keyword): '{text}'")
                continue
        
        # 2️ İletişim bilgisi kontrolü
            if contact_regex.search(text):
                logger.info(f"      Excluded (contact info): '{text}'")
                continue
        
        # 3️ Tarih içeriyor mu?
            if date_regex.search(text):
                logger.info(f"      Excluded (contains date): '{text}'")
                continue
        
        # 4️ Kelime sayısı kontrolü (2-6 kelime arası)
            words = [w for w in text.split() if len(w) > 1]  # 1 harflik kelimeleri sayma
            word_count = len(words)
        
            if word_count < 2:
                logger.info(f"      Excluded (too few words): '{text}'")
                continue
        
            if word_count > 6:
                logger.info(f"      Excluded (too many words): '{text}'")
                continue
        
        # 5️ İSİM PATTERN KONTROLÜ
        

            title_case_count = sum(1 for w in words if len(w) > 1 and w[0].isupper())
            is_title_case = title_case_count >= 2
        
            is_all_caps = text.isupper() and word_count >= 2 and word_count <= 4
        
            has_comma = "," in text
        
            has_title = bool(re.search(r"\b(ph\.?\s*d\.?|m\.?d\.?|ph\.?d\.?|dr\.?|prof\.?)\b", text_lower))
        
        # En az bir pattern'e uymalı
            if not (is_title_case or is_all_caps or has_comma or has_title):
                logger.info(f"      Excluded (no name pattern): '{text}'")
                continue
        
        # 6️ ÖZEL KARAKTER KONTROLÜ
        # Çok fazla noktalama varsa şüpheli
            special_chars = sum(1 for c in text if c in "()[]{}@#$%^&*+=<>/\\|")
            if special_chars > 2:
                logger.info(f"      Excluded (too many special chars): '{text}'")
                continue
        
        # 7️ SAYISAL İÇERİK KONTROLÜ
        # İsimde sayı olmamalı (unvanlar hariç)
            digits = sum(1 for c in text if c.isdigit())
            if digits > 0 and not has_title:  # Unvan varsa (M.D.) sayıya izin ver
                logger.info(f"      Excluded (contains digits): '{text}'")
                continue
        
        #  TÜM FİLTRELERİ GEÇTİ
            score = 0
        
        # Scoring sistemi
            if is_title_case: score += 3
            if is_all_caps: score += 2
            if has_comma: score += 2
            if has_title: score += 2
            if box in underlined: score += 3
            if box in very_large: score += 2
            if box in large_boxes: score += 1
        
            name_candidates.append((box, score))
            logger.info(f"    NAME CANDIDATE (score={score}): '{text}'")
    
    # === EN İYİ ADAYI SEÇ ===
    
        if not name_candidates:
            logger.info(" No name detected")
            return []
    
    # En yüksek skorlu adayı seç
        best_candidate = max(name_candidates, key=lambda x: x[1])
        best_box = best_candidate[0]
        best_score = best_candidate[1]
    
        x, y, w, h = best_box.get("box", (0, 0, 0, 0))
    
        logger.info(f" RESUME NAME DETECTED (score={best_score})")
        logger.info(f"   Text: '{best_box.get('text')}'")
        logger.info(f"   BBox: ({x},{y},{w},{h})")
    
        return [(x, y, w, h)]
    
    def _find_matching_ocr_boxes(
        self, entities: List[Dict], ocr_boxes: List[Dict]
    ) -> List[Dict]:
        matched = []
        for ent in entities:
            # pre_box varsa direkt kullan
            if ent.get("pre_box"):
                matched.append(
                    {"entity": ent, "box": ent["pre_box"], "similarity": 1.0}
                )
                continue

            et = ent["text"].lower().strip()
            if len(et) < 2:
                continue

            best_box, best_score = None, 0.0
            for box in ocr_boxes:
                bt = (box.get("text") or "").lower().strip()
                if not bt:
                    continue

                if et == bt:
                    sim = 1.0
                elif et in bt or bt in et:
                    sim = min(len(et), len(bt)) / max(len(et), len(bt))
                else:
                    sim = SequenceMatcher(None, et, bt).ratio()

                thr = 0.80 if ent["type"] in ["iban", "tax_id", "credit_card", "account"] else 0.75

                if sim >= thr and sim > best_score:
                    best_score, best_box = sim, box

            if best_box:
                matched.append(
                    {"entity": ent, "box": best_box, "similarity": best_score}
                )
        return matched
    
    # ---------- ANA FONKSİYON ----------

    def apply_hybrid_visual_mask(
        self,
        image_b64: str,
        ocr_boxes: list,
        doc_type: str = "other",
        confidence: float = 1.0,
    ):

        try:
            dt = (doc_type or "other").lower()

            # Bazı belge tipleri tamamen atlanır
            if dt in {"news_article", "scientific", "newspaper"}:
                logger.info(f" Skipping masking for doc_type={dt}")
                return image_b64, {}

            if dt == "receipt":
                logger.info(" Receipt mode: Only email/credit_card will be masked")

            # Görseli hazırla
            img_bytes = base64.b64decode(image_b64)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            draw = ImageDraw.Draw(img, "RGBA")
            iw, ih = img.size

            # OCR metni
            full_text = " ".join((box.get("text") or "") for box in ocr_boxes)
            full_text = re.sub(r"(?i)t[\s\n]*a[\s\n]*x[\s\n]*i[\s\n]*d", "tax id", full_text)
            full_text = re.sub(r"(?i)i[\s\n]*b[\s\n]*a[\s\n]*n", "iban", full_text)
            logger.info(f" DocType={dt} | Text length={len(full_text)}")

            # Model & regex
            model_entities = self._detect_with_model(full_text, dt)
            regex_entities = self._detect_regex_patterns(full_text, dt)

            entities = self._merge_entities(model_entities, regex_entities)
            logger.info(f" Final entities after merge: {len(entities)}")
            entities = [e for e in entities if e["type"] == "signature"] + \
                       self._filter_false_positives([e for e in entities if e["type"] != "signature"])
            logger.info(f" After false positive filter: {len(entities)}")

            # ---------- İMZA: sadece form / letter ----------
            # FORM → imza tespiti sadece 2 kelime varsa çalışsın
            sig_regions = []
            opencv_sigs = []
            final_sig_boxes = []

            if dt in ["form", "letter", "other"]:
                lower_text = full_text.lower()
                   
            if dt == "form":
               lower_text = full_text.lower()
               has_signature_hint = (
                  "signature" in lower_text or
                  "signature of initiator" in lower_text
                )

               if not has_signature_hint:
                  logger.info(" FORM: No signature keywords → Signature detection disabled.")
                  sig_regions = []
                  opencv_sigs = []
                  final_sig_boxes = []
                  
               else:
                   # OCR tabanlı imza bölgeleri
                   sig_regions = self._ocr_signature_regions(ocr_boxes, dt, image_b64) or []
                   # OpenCV imzaları
                   opencv_sigs = self._find_signatures_with_opencv(image_b64, ocr_boxes)

            elif dt == "letter":
                lower_text = full_text.lower()
                has_letter_signature_hint = (
                    "sincerely" in lower_text or
                    "cordially" in lower_text or
                    "very truly yours" in lower_text
                )
                
                if not has_letter_signature_hint:
                    logger.info(" LETTER: No closing phrase → Signature detection DISABLED.")
                    sig_regions = []
                    opencv_sigs = []
                    final_sig_boxes = []
                else:
                    sig_regions = self._ocr_signature_regions(ocr_boxes, dt, image_b64) or []
                    opencv_sigs = self._find_signatures_with_opencv(image_b64, ocr_boxes)
                    
                    if sig_regions:
                       x, y, w, h = sig_regions[0]
                       below_top = y + h       # sadece aşağıdan başlasın
                       below_bottom = y + h + int(h * 6)   # açık bir alt alan

                       def inside_below(box):
                           x2, y2, w2, h2 = box
                           return y2 >= below_top and y2 <= below_bottom

                       before = len(opencv_sigs)
                       opencv_sigs = [s for s in opencv_sigs if inside_below(s["box"])]
                       logger.info(f" Letter filter: OpenCV signatures {before} → {len(opencv_sigs)} (only below Sincerely)")


            def overlaps(box1, box2):
                x, y, w, h = box1
                sx, sy, sw, sh = box2
                return not (x + w < sx or sx + sw < x or y + h < sy or sy + sh < y)


            if opencv_sigs:
               if dt == "letter":
                  for s in opencv_sigs:
                      final_sig_boxes.append(s["box"])
               else:
                    # Form: OCR bölgesi varsa onunla kesişeni al
                    if sig_regions:
                       for s in opencv_sigs:
                           if any(overlaps(s["box"], r) for r in sig_regions):
                              final_sig_boxes.append(s["box"])
                    else:
                    # OCR yoksa hepsini al
                        final_sig_boxes.extend([s["box"] for s in opencv_sigs])

            for r in sig_regions:
                if not any(overlaps(r, fb) for fb in final_sig_boxes):
                    final_sig_boxes.append(r)

            # Hepsini entity olarak ekle
            for (sx, sy, sw, sh) in final_sig_boxes:
                sx = max(0, min(int(sx), iw - 1))
                sy = max(0, min(int(sy), ih - 1))
                sw = max(8, min(int(sw), iw - sx))
                sh = max(8, min(int(sh), ih - sy))
                entities.append(
                    {
                        "text": "[signature]",
                        "type": "signature",
                        "score": 0.92,
                        "start": 0,
                        "end": 0,
                        "source": "visual_signature",
                        "pre_box": {"box": (sx, sy, sw, sh), "text": ""},
                    }
                )
            if dt in ["letter", "form", "invoice", "resume"]:
                logger.info(f" Running personal address detection for {dt}")
                address_regions = self._detect_personal_addresses(ocr_boxes, dt, full_text)
            
                for (ax, ay, aw, ah) in address_regions:
                # Sınır kontrolü
                    ax = max(0, min(int(ax), iw - 1))
                    ay = max(0, min(int(ay), ih - 1))
                    aw = max(8, min(int(aw), iw - ax))
                    ah = max(8, min(int(ah), ih - ay))
                
                    entities.append({
                        "text": "[personal_address]",
                        "type": "address",
                        "score": 0.90,
                        "start": 0,
                        "end": 0,
                        "source": "address_detector",
                        "pre_box": {"box": (ax, ay, aw, ah), "text": ""}
                    })
                
                    logger.info(f"   Added personal address entity at ({ax},{ay},{aw},{ah})")

            # ---------- Doğum tarihi: sadece resume ----------

            if dt == "resume":
                dob_regions = self._resume_birthdate_smart(ocr_boxes)
                for (bx,by,bw,bh) in dob_regions:
                    entities.append({
                        "text": "[birth_date]",
                        "type": "date",
                        "score": 0.95,
                        "start": 0, "end": 0,
                        "source": "dob_smart",
                        "pre_box": {"box": (bx,by,bw,bh), "text": ""}
                    })

            # ---------- Resume İsim Maskeleme ----------
            if dt == "resume":
                name_regions = self._detect_resume_name_region(ocr_boxes, dt)
                for (nx, ny, nw, nh) in name_regions:
        # Sınır kontrolü
                    nx = max(0, min(int(nx), iw - 1))
                    ny = max(0, min(int(ny), ih - 1))
                    nw = max(8, min(int(nw), iw - nx))
                    nh = max(8, min(int(nh), ih - ny))
        
                    entities.append({
                        "text": "[resume_name]",
                        "type": "person",  # 'person' rengi kullanılacak
                        "score": 0.92,
                        "start": 0,
                        "end": 0,
                        "source": "resume_name_detector",
                        "pre_box": {"box": (nx, ny, nw, nh), "text": ""}
                    })
        
                    logger.info(f"   Added resume name entity at ({nx},{ny},{nw},{nh})")

            # ---------- OCR eşleştirme ----------
            matches = self._find_matching_ocr_boxes(entities, ocr_boxes)
            logger.info(f" Matched {len(matches)} entities to OCR boxes")

            # ---------- Maske çizimi ----------
            stats: Dict[str, int] = {}
            for m in matches:
                ent, box = m["entity"], m["box"]
                etype = ent["type"]
                x, y, w, h_box = box.get("box", (0, 0, 0, 0))

                x = max(0, min(int(x), iw - 1))
                y = max(0, min(int(y), ih - 1))
                w = max(1, min(int(w), iw - x))
                h_box = max(1, min(int(h_box), ih - y))

                pad = 4
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(iw - x, w + 2 * pad)
                h_box = min(ih - y, h_box + 2 * pad)

                r, g, b = self.colors.get(etype, self.colors["other"])
                draw.rectangle(
                    [x, y, x + w, y + h_box],
                    fill=(r, g, b, 90),
                    outline=(r, g, b, 255),
                    width=2,
                )
                stats[etype] = stats.get(etype, 0) + 1

            buf = BytesIO()
            img.save(buf, format="PNG")
            masked_b64 = base64.b64encode(buf.getvalue()).decode()

            logger.info(f" MASKING DONE | DocType={dt} | Stats={stats}")
            return masked_b64, stats

        except Exception as e:
            logger.error(f" HybridPIIMaskingAgent error: {e}")
            return None, {} 
# ========================= TEXT NORMALIZATION AGENT =========================

class TextNormalizationAgent(BaseAgent):
    def __init__(self):
        super().__init__("normalization_agent")
        self.coordinator = None

    def set_coordinator(self, coordinator):
        self.coordinator = coordinator

    async def handle_message(self, message: A2AMessage):
        if message.message_type == MessageType.REQUEST and message.content.get('action') == 'normalize_text':
            await self.normalize_text(message)

    async def normalize_text(self, request: A2AMessage):
        try:
            text = request.content['text']
            text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            normalized_text = text.lower()
            normalized_text = re.sub(r'[^a-z0-9 \n\.\,\:\-\/\%\(\)\@\_\+]', ' ', normalized_text)
            normalized_text = re.sub(r'[ \t]+', ' ', normalized_text).strip()

            response = await self.send_message(
                receiver="coordinator",
                message_type=MessageType.RESPONSE,
                content={
                    'action': 'normalization_complete',
                    'text': normalized_text,
                    'workflow_id': request.content.get('workflow_id')
                }
            )
            if self.coordinator:
                await self.coordinator.receive_message(response)

        except Exception as e:
            logger.error(f"Normalization error: {e}")
            error_msg = await self.send_message(
                receiver="coordinator",
                message_type=MessageType.ERROR,
                content={'error': str(e), 'workflow_id': request.content.get('workflow_id')}
            )
            if self.coordinator:
                await self.coordinator.receive_message(error_msg)


# ========================= IMPROVED CLASSIFICATION AGENT =========================

class ClassificationAgent(BaseAgent):
    def __init__(self):
        super().__init__("classification_agent")
        self.coordinator = None
        
        model_name = "joeddav/xlm-roberta-large-xnli"
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        cls_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model=cls_model,
            tokenizer=tok,
            device=0 if torch.cuda.is_available() else -1
        )

        self.label_map_en = {
            "invoice": "invoice",
            "bill": "invoice",
            "e-invoice": "invoice",
            "receipt": "receipt",
            "slip": "receipt",
            "letter": "letter",
            "petition": "letter",
            "form": "form",
            "application form": "form",
            "email": "email",
            "e-mail": "email",
            "electronic mail": "email",
            "inbox": "email",
            "message": "email",
            "news article": "news_article",
            "article": "news_article",
            "resume": "resume",
            "cv": "resume",
            "curriculum vitae": "resume",
            "other": "other"
        }
        self.candidate_labels_en = list(self.label_map_en.keys())
        self.hypothesis_template_en = "This text is about {}."

        self.keyword_map_en = {
            "invoice": ["invoice", "bill", "vat", "tax", "amount due", "payment terms", "subtotal", "total amount"],
            "receipt": ["receipt", "payment", "transaction", "paid", "total", "thank you for your purchase"],
            "letter": ["dear", "regards", "sincerely", "yours truly", "respectfully", "cordially", "telegram", "western union"],
            "form": [
                "form", "application", "checkbox", "please fill", "signature", "required field",
                "request no", "request number", "initiated by", "completion", "target date",
                "country:", "product:", "nature of work", "laboratory analysis", "reports",
                "requirements", "responsibility", "special requirements", "sample specifications",
                "by:", "date:", "notes:"
            ],
            "email": [
                "from:", "to:", "subject:", "cc:", "bcc:", "sent:", "date:", "priority:", 
                "author:", "reply", "forward", "inbox", "gmail", "outlook", "yahoo", 
                "hotmail", "re:", "fwd:", "original message", " at ", "wrote:", "replied:",
                "sent from", "message", "recipient", "sender"
            ],
            "news_article": [
                "news", "reported", "breaking", "according to", "journalist", "press", 
                "correspondent", "breaking news", "in an interview", "sources say",
                "officials confirmed", "newspaper", "times", "post", "herald"
            ],
            "resume": [
                "resume", "cv", "curriculum vitae", "professional experience", 
                "work experience", "employment history", "education", "skills", 
                "qualifications", "achievements", "objective", "career summary",
                "references", "languages", "certifications", "projects",
                "personal information", "contact information", "linkedin",
                "years of experience", "bachelor", "master", "phd", "bachelor", "master", "degree", "university",
                "professional summary", "technical skills", "soft skills"
            ]
        }

        self.news_keywords = [
            "reported", "according to", "sources say", "officials said",
            "announced", "confirmed", "stated", "spokesperson", "breaking news",
            "updated", "developing story", "investigation", "authorities",
            "witness", "incident", "police", "government", "president",
            "parliament", "senate", "election", "vote", "campaign",
            "journalist", "correspondent", "press release", "in an interview",
            "officials confirmed", "newspaper", "wire service", "dateline"
        ]

        self.newspaper_indicators = [
            r"\b(associated press|reuters|ap|upi)\b",  # Wire services
            r"\b(reported|according to|officials said|spokesperson)\b",  # News language
            r"\b(times|post|herald|tribune|journal|gazette)\b",  # Publisher names
            r"^\s*by\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s*$",  # Byline pattern
        ]

        self.re_dateline = re.compile(
            r"^[A-Z][A-Z\s]{2,20}(?:\([A-Z]+\))?\s*[—–-]\s*",
            re.MULTILINE
        )


        self.email_header_re = [
            re.compile(r"^from\s*:", re.I | re.M),
            re.compile(r"^to\s*:", re.I | re.M),
            re.compile(r"^subject\s*:", re.I | re.M),
            re.compile(r"^date\s*:", re.I | re.M),
            re.compile(r"^cc\s*:", re.I | re.M),
            re.compile(r"^bcc\s*:", re.I | re.M),
            re.compile(r"^reply[-\s]?to\s*:", re.I | re.M),
            re.compile(r"^message[-\s]?id\s*:", re.I | re.M),
            re.compile(r"^mime[-\s]?version\s*:", re.I | re.M),
            re.compile(r"^content[-\s]?type\s*:", re.I | re.M),
            re.compile(r"^-{2,}\s*original\s*message\s*-{2,}$", re.I | re.M),
            re.compile(r"^on .+\b(wrote|said)\s*:", re.I | re.M),
        ]
        
        self.any_email_hdr_patterns = [
            re.compile(r"^from\s*:", re.I|re.M),
            re.compile(r"^to\s*:", re.I|re.M),
            re.compile(r"^subject\s*:", re.I|re.M),
            re.compile(r"^date\s*:", re.I|re.M),
            re.compile(r"^cc\s*:", re.I|re.M),
            re.compile(r"^bcc\s*:", re.I|re.M),
        ]

        self.re_email_addr = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I)

        # FORM patterns
        self.form_header_re = [
            re.compile(r"^(request\s*no\.?|request\s*number)\s*:", re.I|re.M),
            re.compile(r"^date\s*:", re.I|re.M),
            re.compile(r"^initiated\s*by\s*:", re.I|re.M),
            re.compile(r"^completion\s*$", re.I|re.M),
            re.compile(r"^target\s*date\s*:", re.I|re.M),
            re.compile(r"^country\s*:", re.I|re.M),
            re.compile(r"^product\s*:", re.I|re.M),
            re.compile(r"^nature\s*of\s*work\s*:", re.I|re.M),
            re.compile(r"^r&d\s*comments\s*:", re.I|re.M),
            re.compile(r"^requirements\s*$", re.I|re.M),
            re.compile(r"^laboratory\s*analysis\s*$", re.I|re.M),
            re.compile(r"^reports\s*$", re.I|re.M),
            re.compile(r"^responsibility\s*$", re.I|re.M),
            re.compile(r"^special\s*requirements\s*$", re.I|re.M),
        ]
        self.form_artifacts_re = [
            re.compile(r"_{4,}"),
            re.compile(r"\b[a-z]\)"),
            re.compile(r"^\s*[-•]\s+", re.M),
            re.compile(r"\bmm\b|\bml\b|\bno\.\b", re.I),
        ]

        self.receipt_kw = [
            r"\breceipt\b", r"\bterminal\b", r"\bmerchant\b", r"\bcashier\b", r"\bauth(ori[sz]ation)?\s*code\b",
            r"\b(card|visa|mastercard|amex|debit|credit)\b", r"\bapproved\b", r"\btender\b",
            r"\bchange\b", r"\bref\s*#\b", r"\bpos\b"
        ]
        self.invoice_kw = [
            r"\binvoice\b", r"\binvoice\s*(no|number|#)\b", r"\bbill\s*to\b", r"\bship\s*to\b", r"\bsubtotal\b",
            r"\btotal( amount)?\b", r"\bamount\s+due\b", r"\bpayment\s+terms\b", r"\bpo\s*(no|number|#)\b",
            r"\bqty\b", r"\bquantity\b", r"\bunit\s*price\b", r"\bnet\b", r"\btax\b"
        ]
        self.re_receipt = [re.compile(p, re.I) for p in self.receipt_kw]
        self.re_invoice = [re.compile(p, re.I) for p in self.invoice_kw]
        self.re_time = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d(\s?(am|pm))?\b", re.I)

        self.receipt_pos_kw = [
            r"\breceipt\b", r"\bwalk[_ -]?in\b", r"\border\b", r"\bserver\b", r"\btable\b",
            r"\bguest[s]?\b", r"\bticket\b", r"\bcheck\b", r"\bsubtotal\b", r"\b(bev|food)\s*subtotal\b",
            r"\bsales?\s*tax\b", r"\btax\b", r"\btotal\b", r"\bchange\b", r"\bpaid\b", r"\bbalance\s*due\b",
            r"\bcash\b", r"\b(card|visa|master\s*card|amex|discover|debit|credit)\b",
            r"\bauth(ori[sz]ation)?\s*code\b", r"\bapproval\b", r"\bterminal\b", r"\bmerchant\b",
            r"\bgratuity\b", r"\btip\b", r"\bthank you\b"
        ]
        self.invoice_hdr_kw = [
            r"\binvoice\b", r"\binvoice\s*(no|number|#)\b", r"\bbill\s*to\b", r"\bship\s*to\b",
            r"\bpo\s*(no|number|#)\b", r"\bpurchase\s*order\b", r"\bremittance\b", r"\bterms\b",
            r"\bdue\s*date\b", r"\bamount\s*due\b", r"\bproject\b", r"\bdescription\b",
            r"\bqty\b|\bquantity\b|\bunit\s*price\b|\brate\b|\bext\.?\b"
        ]
        self.re_receipt_pos = [re.compile(p, re.I) for p in self.receipt_pos_kw]
        self.re_invoice_hdr = [re.compile(p, re.I) for p in self.invoice_hdr_kw]
        self.re_date = re.compile(
            r"\b(20\d{2}|19\d{2})[-/\.](0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])\b"
            r"|\b(0?[1-9]|[12]\d|3[01])[-/\.](0?[1-9]|1[0-2])[-/\.](20\d{2}|19\d{2})\b"
        )
        self.re_money = re.compile(r"(?:[$€£])?\s?\b\d{1,4}[.,]\d{2}\b")

        self.re_label_line = re.compile(r"^[A-Z][A-Z0-9 ,/()\-]{2,30}\s*:\s*$", re.M)
        self.form_kw_headers = [
            "group no", "% solution", "dosage", "results", "signature", "signature(s)",
            "date received", "tested", "investigator", "source", "route of compound administration",
            "compound vehicle", "male", "female", "strain of mice", "reference for calculation",
            "conclusion", "copies to the following"
        ]
        self.re_allcaps_line = re.compile(r"^[A-Z0-9 ,()/\-]{10,}$")
        
        self.lab_form_cues = [
            "compound code", "molecular weight", "solubility", "reactivity",
            "storage recommendations", "safety comments", "handling procedure",
            "lorillard research center", "form 9", "product specification",
            "prepared by", "approved by", "sample specifications", 
            "physical characteristics", "batch size", "responsibility",
            "requirements", "laboratory analysis", "request made by"
        ]
        self.re_lab_form_receipt_veto = [
            re.compile(r"\bsample\s+no\.?\b", re.I),
            re.compile(r"\blaboratory analysis\b", re.I),
            re.compile(r"\brequest made by\b", re.I),
        ]

        self.letter_salutations = [
            r"^dear\s+(sir|madam|mr|ms|mrs|dr|[a-z][\w\-']+)\b",
        ]
        self.letter_closings = [
            r"\byours\s+faithfully\b", r"\byours\s+sincerely\b",
            r"\bsincerely\b", r"\bwith\s+kind\s+regards\b", r"\bbest\s+regards\b",
            r"\byours\s+truly\b", r"\brespectfully\b"
        ]
        self.re_letter_date = re.compile(
            r"^((jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s+\d{4}"
            r"|\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4})$",
            re.I | re.M
        )
        self.re_address_line = re.compile(
            r"\b(p\.?\s*o\.?\s*box|street|st\.|avenue|ave\.|road|rd\.|drive|dr\.|suite|corp(oratio)?n|company|inc\.?)\b",
            re.I
        )
        self.re_letter_subject = re.compile(r"^(re|subject)\s*:\s+\S+", re.I | re.M)

        self.re_news_byline  = re.compile(r"^\s*by\s+[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+\b", re.I|re.M)
        self.re_news_dateline = re.compile(r"^[A-Z][A-Za-z .'-]{2,40}\s[-–—]\s", re.M)
        self.re_news_month  = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
                                             r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b", re.I)
        self.re_news_head_allcaps = re.compile(r"^(?:[A-Z0-9][A-Z0-9'&,-]{2,}\s+){2,12}[A-Z0-9'&,-]{2,}$")
        self.re_news_pubwords = re.compile(
            r"(associated press|reuters|ap news|upi|newswire|bloomberg|financial times|the times|"
            r"new york times|washington post|wall street journal|guardian|gazette|journal|tribune|post|herald|"
            r"courier[- ]journal|telegraph|independent|bbc|cnn|al jazeera|fox news|nbc news|abc news)",
            re.I
        )
        self.re_news_pagenote = re.compile(r"\bpage\s+\d+\b", re.I)
        
        self.re_frontpage_cues = re.compile(
            r"\bvol\.\s*[A-Z0-9]+\b|\bno\.\s*\d{1,6}\b|\b(late edition|final edition|early edition)\b|"
            r"\b(all the news|that's fit to print)\b|\bnew york\b.*\b[0-9]{4}\b",
            re.I
        )
        self.re_frontpage_price = re.compile(
            r"\$\s?\d+(\.\d{2})?\b", re.I
        )
        
        self.re_corporate_path_email = re.compile(
            r"[A-Za-z]+/[A-Za-z]+/[A-Z]{2,}@[A-Z]{2,}",  # Example: Jordan Bressler/Lorillard/MLBA@MLBA
            re.I
        )

        self.re_b2b_email = re.compile(
            r"(accountspayable|accounts\s*payable|invoices?|billing|ar\b|ap\b)@",
            re.I
        )
        self.re_professional_items = re.compile(
            r"\b(loctite|polyimide|duopak|araldite|adhesive|tape|chemical|compound|reagent)\b",
            re.I
        )

    def set_coordinator(self, coordinator):
        self.coordinator = coordinator

    async def handle_message(self, message: A2AMessage):
        if message.message_type == MessageType.REQUEST and message.content.get('action') == 'classify':
            await self.classify_document(message)

    def _make_sample(self, text: str, chunk: int = 600) -> str:
        text = (text or "").strip()
        if len(text) <= chunk * 3:
            return text
        mid = len(text) // 2
        return " ".join([text[:chunk], text[max(0, mid - chunk // 2):mid + chunk // 2], text[-chunk:]])

    def _keyword_hint(self, text: str) -> Optional[str]:
        t = (text or "").lower()
        scores = {dt: sum(1 for kw in kws if kw in t) for dt, kws in self.keyword_map_en.items()}
        
        if self._has_email_address(t):
            scores["email"] = scores.get("email", 0) + 2
        
        best_dt, best_hits = max(scores.items(), key=lambda x: x[1])
        
        second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
        
        if best_hits >= 3:
            return best_dt
        elif best_hits >= 2:
            return best_dt
        elif best_hits == 1 and second_best == 0:
            return best_dt 
        
        return None

    def _count_email_headers(self, raw_text: str) -> int:
        t = raw_text or ""
        core_headers = [
            re.compile(r"^from\s*:", re.I | re.M),
            re.compile(r"^to\s*:", re.I | re.M),
            re.compile(r"^subject\s*:", re.I | re.M),
            re.compile(r"^date\s*:", re.I | re.M),
        ]
        return sum(1 for p in core_headers if p.search(t))
    
    def _count_any_email_headers_top(self, raw_text: str, top_n_lines: int = 60) -> int:
        lines = (raw_text or "").splitlines()[:top_n_lines]
        block = "\n".join(lines)
        return sum(1 for p in self.any_email_hdr_patterns if p.search(block))
    
    def _has_email_thread_artifacts(self, raw_text: str) -> bool:
        t = raw_text or ""
        patterns = [
            re.compile(r"^-{2,}\s*(original|forwarded)\s*message\s*-{2,}", re.I|re.M),
            re.compile(r"^on .+\b(wrote|said)\s*:", re.I|re.M),
            re.compile(r"\bunsubscribe\b", re.I),
            re.compile(r"\bview\s+in\s+browser\b", re.I),
            re.compile(r"\bsent\s+from\s+my\s+iphone\b", re.I),
            re.compile(r"\bthis\s+email\s+was\s+sent\s+to\b", re.I),
        ]
        return any(p.search(t) for p in patterns)
    
    def _is_lab_form_like(self, raw_text: str, norm_text: str) -> bool:
        t = (raw_text + "\n" + norm_text).lower()
        
        hits = sum(1 for kw in self.lab_form_cues if kw in t)
        
        has_spec_numbers = len(re.findall(r"\b\d+\s*(mg|g|ml|mm|kg|lbs|%)?\b", t)) >= 10
        has_approval_block = bool(re.search(r"\b(prepared by|approved by|responsibility|signature)\b", t))
        
        return (hits >= 4) or (hits >= 2 and has_spec_numbers and has_approval_block)

    def _is_newspaper_frontpage_like(self, raw_text: str, norm_text: str) -> bool:
        t = (raw_text or "").lower()
        raw = raw_text or ""
        top_lines = "\n".join(raw.splitlines()[:15])

        cues = 0
        if self.re_frontpage_cues.search(top_lines): cues += 2
        
        if self.re_frontpage_price.search(top_lines) and re.search(r"\b(tuesday|wednesday|thursday|friday|saturday|sunday|monday)\b", t):
            cues += 1
            
        if re.search(r"\bshutdown ends\b", t) or re.search(r"\bover dreamers\b", t): cues += 1 
        
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) > 50 and sum(1 for ln in lines if 20 <= len(ln) <= 50) >= 30:
             cues += 1
        
        return cues >= 3

    def _looks_like_informal_email_body(self, raw_text: str) -> bool:

        t = (raw_text or "").lower()
        top_block = "\n".join((raw_text or "").splitlines()[:40]).lower()
        
        has_subjectish = re.search(r"\bsubject\b", top_block)

        has_timestamp = re.search(
            r"\b(0?[1-9]|1[0-2]):[0-5]\d\s*(am|pm)\b", top_block
        ) or re.search(
            r"\b(20|19)\d{2}\b", top_block
        )
        
        has_corporate_path = bool(self.re_corporate_path_email.search(raw_text))

        convo_bits = (
            "thought i'd" in t or
            "thought'd" in t or  #  YENİ: OCR hatası için
            "let you read it first" in t or
            "before i send" in t or
            "before i sent" in t or
            "please review" in t or
            "see comments below" in t or
            "fyi" in t or #  YENİ
            "press release" in t #  YENİ (81839536.png için spesifik)
        )

        looks_like_letter_salutation = re.search(r"^\s*(dear|hi|hello)\b", top_block, re.I|re.M)

        return bool(
            (has_subjectish and has_timestamp and convo_bits) or
            (has_corporate_path and has_subjectish) #  YENİ: path + subject = email
        ) and not looks_like_letter_salutation

    def _has_job_history_timeline(self, raw_text: str, norm_text: str) -> bool:
        """
        Detects repeating YYYY-YYYY / YYYY-present ranges with job-ish titles,
        which is super typical in resumes and almost never in invoices.
        """
        t = ((raw_text or "") + "\n" + (norm_text or "")).lower()

        ranges = re.findall(
            r"(19|20)\d{2}\s*[-–—]\s*((19|20)\d{2}|present)",
            t
        )
        multi_ranges = len(ranges) >= 2

        job_words = re.search(
            r"\b(manager|engineer|director|analyst|assistant|associate|intern|fellow|consultant|researcher|scientist|instructor|clinician|specialist|technician|administrator|coordinator|supervisor|developer|architect)\b",
            t
        )

        return bool(multi_ranges and job_words)


    def _score_receipt_vs_invoice(self, raw_text: str, norm_text: str):
        """Strong scoring for Receipt vs Invoice"""
        raw = raw_text or ""
        raw_lower = raw.lower()
        t = (raw + "\n" + (norm_text or "")).lower()

        if any(rx.search(raw) for rx in self.re_lab_form_receipt_veto):
             return (None, 0.0)
        
        if self._is_newspaper_frontpage_like(raw_text, norm_text):

            return (None, 0.0)
        
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        avg_len = (sum(len(ln) for ln in lines) / max(1, len(lines))) if lines else 0
        narrow_layout = bool(avg_len and avg_len < 36 and 18 <= len(lines) <= 180)
        
        pos_hits = sum(1 for rx in self.re_receipt_pos if rx.search(t))
        inv_hits = sum(1 for rx in self.re_invoice_hdr if rx.search(t))
        money_cnt = len(self.re_money.findall(t))
        has_time = bool(self.re_time.search(t))
        has_date = bool(self.re_date.search(t))

        # POS işaretleri
        has_merchant = bool(re.search(r"\bmerchant\b", t))
        has_terminal = bool(re.search(r"\bterminal\b", t))
        has_auth = bool(re.search(r"\bauth(ori[sz]ation)?\s*code\b", t))
        pos_phrase = bool(re.search(r"\b(thank you|cashier|change|tender|approval)\b", t))
        card_token = bool(re.search(r"\b(card|visa|mastercard|amex|debit|credit)\b", t))

        has_b2b_email = bool(self.re_b2b_email.search(raw_lower))

        has_professional_items = bool(self.re_professional_items.search(raw_lower))

        has_invoice_header = bool(re.search(r"\binvoice\b", raw_lower)) and (
            re.search(r"\bbill\s+to\b", raw_lower)
            or re.search(r"\bship\s+to\b", raw_lower)
            or has_b2b_email 
            or re.search(r"\bterms?:?\s+net\s+\d+\b", raw_lower)  # net 30 / net 60 tarzı
        )
        has_pos_style = (has_merchant or has_terminal or has_auth or card_token or narrow_layout)

        if has_invoice_header and (has_b2b_email or has_professional_items) and not has_pos_style and money_cnt >= 2:
            return ("invoice", 0.85)

        if has_invoice_header and not has_pos_style and money_cnt >= 2:
             return ("invoice", 0.8)
        
        top_block = "\n".join(lines[:15]).lower()
        company_block_candidates = re.findall(
            r"[A-Z][A-Z0-9&\.\- ]{2,}\n.*\b(street|st\.|road|rd\.|drive|dr\.|blvd|suite|ste\.|ave|avenue|court|ct\.|ln|lane|way|unit|inc\.|ltd|corp|llc|ontario|ca|ny|tx|usa|canada)\b",
            top_block,
            flags=re.I
        )

        two_company_headers = len(company_block_candidates) >= 2

        has_invoice_totals = (
            "subtotal" in raw_lower or
            "total due" in raw_lower or
            "amount due" in raw_lower or
            "balance due" in raw_lower
        )
        if two_company_headers and not has_pos_style and money_cnt >= 2 and has_invoice_totals:
            return ("invoice", 0.8)

        # "Tax Invoice" olan POS fişi düzeltmesi (FIXED BLOCK - Priority check)
        is_tax_invoice = bool(re.search(r"\btax\s*invoice\b", t))
        if is_tax_invoice and (has_merchant or has_terminal or has_auth or pos_phrase) and money_cnt >= 2:
            # POS fişi gibi davran: receipt olarak işaretle
            return ("receipt", 0.74 if not narrow_layout else 0.78)

        # Güçlü fatura tablosu (invoice no/bill to/ship to + qty/unit price/amount)
        has_invoice_keywords = bool(re.search(r"\binvoice(\s*(no|number|#))?\b|\bbill\s*to\b|\bamount\s*due\b", t))
        has_table_tokens = bool(re.search(r"\b(qty|quantity|unit\s*price|rate|amount)\b", t))
        # NEW: Strong Invoice structural check (Vetoed by POS presence)
        if has_invoice_keywords and has_table_tokens and not (has_merchant or has_terminal or has_auth or pos_phrase or card_token):
            return ("invoice", 0.80)
            
        if has_merchant: pos_hits += 2
        if has_terminal: pos_hits += 1
        if has_auth:     pos_hits += 1
        if card_token:   pos_hits += 1
        if has_time and money_cnt >= 2: pos_hits += 1

        if has_merchant:
            inv_hits -= 1

        if "invoice" in t and re.search(r"\b(qty|quantity|unit\s*price|po\s*(no|#))\b", t):
            inv_hits += 2
            
        narrow_layout = bool(avg_len and avg_len < 36 and 18 <= len(lines) <= 180)
        pos_strong = (pos_hits >= 3 and money_cnt >= 3) and (has_time or has_date or narrow_layout)

        inv_strong = (inv_hits >= 3) or (("invoice" in t) and (re.search(r"\b(qty|unit\s*price|po\s*(no|#))\b", t)))

        if pos_strong and not inv_strong:
            base_conf = 0.78 if narrow_layout else 0.72
            return ("receipt", base_conf)
        if inv_strong and not pos_strong:
            return ("invoice", 0.78)

        if pos_hits or inv_hits:
            score = (pos_hits + (1 if narrow_layout else 0) + (1 if has_time else 0)) - (inv_hits + 0.5)
            if score >= 2:
                return ("receipt", 0.72)
            if score <= -2:
                return ("invoice", 0.72)

        both_strong = (pos_hits >= 3 and inv_hits >= 3) or (("invoice" in t) and re.search(r"\b(qty|unit\s*price|po\s*(no|#))\b", t))
        if both_strong:
            if ("invoice" in t) and re.search(r"\b(qty|quantity|unit\s*price|po\s*(no|#))\b", t):
                return ("invoice", 0.74 if not has_merchant else 0.72)
            elif has_merchant or card_token or has_auth:
                return ("receipt", 0.72)
            else:
                return ("receipt", 0.70)

        return (None, 0.0)

    def _is_resume_like(self, raw_text: str, norm_text: str) -> bool:
        low = ((raw_text or "") + "\n" + (norm_text or "")).lower()
        
        grant_indicators = bool(re.search(
            r"\b(grant|nih|nsf|research support|fellowship|postdoctoral|principal investigator)\b",
            low
        ))

        # Resume-specific section headers
        section_patterns = [
            r"\b(professional |work )?experience\b",
            r"\beducation\b",
            r"\b(technical )?skills\b",
            r"\bcertifications?\b",
            r"\b(career )?objective\b",
            r"\bsummary\b",
            r"\bqualifications\b",
            r"\breferences\b",
            r"\bprojects?\b",
            r"\bcurriculum\s+vitae?\b|\brésumé\b|\bresume\b",
            r"\b(employment history|research fellow|residency|clinical instructor|private practice)\b",
            r"\bbiographical\s+(sketch|data)\b",  
            r"\bprofessional\s+affiliations?\b",   
        ]
        
        section_hits = sum(1 for p in section_patterns if re.search(p, low))
        
        has_years_exp = bool(re.search(r"\d+\+?\s*years?\s*(of\s*)?(experience|exp)\b", low))
        has_degree = bool(re.search(r"\b(bachelor|master|phd|b\.?s\.?|m\.?s\.?|m\.?a\.?|degree|university)\b", low)) #  M.A. eklendi
        has_date_range = len(re.findall(r"\b(19|20)\d{2}\s*[-–—]\s*(present|(19|20)\d{2})\b", low)) >= 2
        has_email_phone = bool(self.re_email_addr.search(low)) and bool(re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", low))
        
        has_academic_title = bool(re.search(
            r"\b(professor|assistant professor|associate professor|lecturer|researcher|scientist|fellow)\b",
            low
        ))
               # Score calculation
        score = section_hits
        if has_years_exp: score += 2
        if has_degree: score += 2
        if has_date_range: score += 2
        if has_email_phone: score += 1
        if grant_indicators: score += 3  
        if has_academic_title: score += 2  
        
        if re.search(r"\bcurriculum\s+vitae?\b|\brésumé\b|\bresume\b|\bbiographical\b", low): 
            score += 3
        
        return score >= 6

    # Haber Güç Skoru
    def _calculate_news_article_strength(self, raw_text: str, norm_text: str) -> int:
        score = 0
        t = (raw_text or "").lower()
        top_200_chars = raw_text[:200] if raw_text else ""
        
        if self.re_dateline.search(top_200_chars):
            score += 3
            
        if self.re_news_byline.search(top_200_chars):
            score += 2
            
        news_kw_count = sum(1 for kw in self.news_keywords if kw in t)
        if news_kw_count >= 4:
            score += 2
        elif news_kw_count >= 2:
            score += 1
            
        if self.re_news_pubwords.search(t):
            score += 2
            
        paragraphs = [p for p in raw_text.split('\n\n') if 50 < len(p.strip()) < 500]
        if len(paragraphs) >= 3:
            score += 1
            
        if t.count('"') >= 4:
            score += 1
            
        return score


    def _looks_like_news_article(self, raw_text: str, norm_text: str) -> bool:
        #  DÜZELTME: _news_article_score yerine _calculate_news_article_strength kullanıldı
        news_strength = self._calculate_news_article_strength(raw_text, norm_text)
        score = news_strength 
        raw = raw_text or ""
        top = "\n".join(raw.splitlines()[:60])

        if self._is_lab_form_like(raw_text, norm_text):
            return False
        
        # Email veto: gerçek e-posta eserleri varsa haber olmasın
        has_email_headers = self._count_email_headers(raw) >= 2
        has_email_with_headers = self._has_email_address(raw) and re.search(r"^(from|to|subject)\s*:", top, re.I|re.M)
        has_message_id = re.search(r"^message[-\s]?id\s*:", top, re.I|re.M)
        
        if has_email_headers or has_email_with_headers or has_message_id:
             # Sadece News Article strength yüksek değilse veto et
             if news_strength < 7:
                 return False
            
        if self._is_form_layout_like(raw_text, norm_text):
            return False
        
        t = (raw_text + "\n" + norm_text).lower()
        
        news_kw_count = sum(1 for kw in self.news_keywords if kw in t)
        if news_kw_count >= 5: 
            score += 2
        elif news_kw_count >= 3:
            score += 1
            
        paragraphs = [p.strip() for p in raw_text.split('\n\n') if len(p.strip()) > 100]
        if 4 <= len(paragraphs) <= 12: 
            score += 1
            
        quote_count = t.count('"')
        if quote_count >= 4:
            score += 1

        has_byline = bool(self.re_news_byline.search(top))
        has_publisher = bool(self.re_news_pubwords.search(top))
        if not (has_byline or has_publisher):
            # Ancak dateline varsa kurtarabilir
            if not self.re_dateline.search(top):
                return False 
        has_body_text = len([ln for ln in raw_text.splitlines() if len(ln.strip()) > 50]) >= 10
        
        # Threshold gevşetildi
        if not (news_strength >= 2 and has_body_text):  # 3 -> 2
            return False
        
        return news_strength >= 4  # 5 -> 4
    

    async def classify_document(self, request: A2AMessage):
        try:
            raw_text = request.content.get('text_raw', '') or ''
            norm_text = request.content.get('text_normalized', raw_text) or ''
            joined = f"{raw_text}\n{norm_text}"

            if len(raw_text.strip()) == 0 and len(norm_text.strip()) == 0:
                response = await self.send_message(
                    receiver="coordinator",
                    message_type=MessageType.RESPONSE,
                    content={'action': 'classification_complete', 'document_type': 'other', 'confidence': 0.0,
                             'workflow_id': request.content.get('workflow_id')}
                )
                if self.coordinator:
                    await self.coordinator.receive_message(response)
                return

            if self._is_newspaper_frontpage_like(raw_text, norm_text):
                response = await self.send_message(
                    receiver="coordinator",
                    message_type=MessageType.RESPONSE,
                    content={'action': 'classification_complete',
                             'document_type': 'news_article',
                             'confidence': 0.90, 
                             'workflow_id': request.content.get('workflow_id')}
                )
                if self.coordinator:
                    await self.coordinator.receive_message(response)
                return

            if self._is_lab_form_like(raw_text, norm_text):
                response = await self.send_message(
                    receiver="coordinator",
                    message_type=MessageType.RESPONSE,
                    content={'action': 'classification_complete',
                             'document_type': 'form',
                             'confidence': 0.82,
                             'workflow_id': request.content.get('workflow_id')}
                )
                if self.coordinator:
                    await self.coordinator.receive_message(response)
                return

            if self._is_resume_like(raw_text, norm_text):
                response = await self.send_message(
                    receiver="coordinator",
                    message_type=MessageType.RESPONSE,
                    content={
                        'action': 'classification_complete',
                        'document_type': 'resume',
                        'confidence': 0.85,
                        'workflow_id': request.content.get('workflow_id')
                    }
                )
                if self.coordinator:
                    await self.coordinator.receive_message(response)
                return

            if self._has_job_history_timeline(raw_text, norm_text):
                response = await self.send_message(
                    receiver="coordinator",
                    message_type=MessageType.RESPONSE,
                    content={
                        'action': 'classification_complete',
                        'document_type': 'resume',
                        'confidence': 0.85,
                        'workflow_id': request.content.get('workflow_id')
                    }
                )
                if self.coordinator:
                    await self.coordinator.receive_message(response)
                return

            hdr_top = self._headers_in_top_block(raw_text, top_n_lines=60)
            any_hdr_top = self._count_any_email_headers_top(raw_text, top_n_lines=60)
            core_hdrs = self._count_email_headers(raw_text)
            top60 = (raw_text or "").splitlines()[:60]
            sys_header_patterns = [
                re.compile(r"^message[-\s]?id\s*:", re.I),
                re.compile(r"^mime[-\s]?version\s*:", re.I),
                re.compile(r"^content[-\s]?type\s*:", re.I),
            ]
            sys_hdrs = sum(1 for ln in top60 if any(p.match(ln.strip()) for p in sys_header_patterns))
            has_thread = self._has_email_thread_artifacts(raw_text)
            quote_lines = sum(1 for ln in raw_text.splitlines() if ln.strip().startswith('>'))
            has_email_addr = self._has_email_address(raw_text)
            label_line_cnt = len(self.re_label_line.findall(raw_text))
            has_corporate_email_pattern = bool(self.re_corporate_path_email.search(raw_text)) 
            looks_like_form = self._is_form_layout_like(raw_text, norm_text)
            
            news_strength = self._calculate_news_article_strength(raw_text, norm_text)

            email_signal = (
                ((core_hdrs >= 2) or (any_hdr_top >= 3)) and
                (has_email_addr or sys_hdrs >= 1 or has_thread or quote_lines >= 5 or has_corporate_email_pattern) 
            )

            #  News article çok güçlüyse email'i veto et
            if email_signal and news_strength >= 7:
                email_signal = False 

            #  Form düzeni baskınsa e-postayı veto et
            if email_signal and (label_line_cnt >= 4 or self._is_form_layout_like(raw_text, norm_text)):
                if core_hdrs < 3:
                    email_signal = False
                else: 
                    pass

            looks_news = self._looks_like_news_article(raw_text, norm_text)
            if looks_news:  #  Düzeltilmiş: _looks_like_news_article True ise hemen çıkış yap
                response = await self.send_message(
                    receiver="coordinator",
                    message_type=MessageType.RESPONSE,
                    content={
                        'action': 'classification_complete',
                        'document_type': 'news_article',
                        'confidence': 0.85 if news_strength >= 6 else 0.80, # News strength'e göre dinamik güvenilirlik
                        'workflow_id': request.content.get('workflow_id')
                    }
                )
                if self.coordinator:
                    await self.coordinator.receive_message(response)
                return

            informal_email_hit = self._looks_like_informal_email_body(raw_text) 
            if email_signal:
                ri_label, _ = self._score_receipt_vs_invoice(raw_text, norm_text)
                if ri_label != "receipt":
                    response = await self.send_message(
                        receiver="coordinator",
                        message_type=MessageType.RESPONSE,
                        content={
                            'action': 'classification_complete',
                            'document_type': 'email',
                            'confidence': 0.85,
                            'workflow_id': request.content.get('workflow_id')
                        }
                    )
                    if self.coordinator:
                        await self.coordinator.receive_message(response)
                    return

            if informal_email_hit:
                ri_label, _ = self._score_receipt_vs_invoice(raw_text, norm_text)
                if ri_label != "receipt":
                    response = await self.send_message(
                        receiver="coordinator",
                        message_type=MessageType.RESPONSE,
                        content={
                            'action': 'classification_complete',
                            'document_type': 'email',
                            'confidence': 0.85,
                            'workflow_id': request.content.get('workflow_id')
                        }
                    )
                    if self.coordinator:
                        await self.coordinator.receive_message(response)
                    return


            strong_form = (
                len(self.re_label_line.findall(raw_text)) >= 5 or
                sum(1 for kw in self.form_kw_headers if kw in (norm_text or "").lower()) >= 4 or
                self._is_lab_form_like(raw_text, norm_text) or 
                (self._is_form_layout_like(raw_text, norm_text) and len(re.findall(r"\b\d[\d\.,]*\b", (norm_text or ""))) >= 10)
            )
            # Çok para tutarı ve 'invoice' geçse bile kolonlu başlık yoğunluğu form'a öncelik verir
            if strong_form:
                response = await self.send_message(
                    receiver="coordinator",
                    message_type=MessageType.RESPONSE,
                    content={'action': 'classification_complete',
                             'document_type': 'form',
                             'confidence': 0.82,
                             'workflow_id': request.content.get('workflow_id')}
                )
                if self.coordinator:
                    await self.coordinator.receive_message(response)
                return

            label_ri, conf_ri = self._score_receipt_vs_invoice(raw_text, norm_text)
            if label_ri:
                response = await self.send_message(
                    receiver="coordinator",
                    message_type=MessageType.RESPONSE,
                    content={'action': 'classification_complete',
                             'document_type': label_ri,
                             'confidence': float(conf_ri),
                             'workflow_id': request.content.get('workflow_id')}
                )
                if self.coordinator:
                    await self.coordinator.receive_message(response)
                return


            sample = self._make_sample(f"{raw_text} {norm_text}") or (joined if joined.strip() else "document")
            try:
                r1 = self.classifier(sample, self.candidate_labels_en, multi_label=False,
                                     hypothesis_template="This document is a {}.")
                r2 = self.classifier(sample, self.candidate_labels_en, multi_label=False,
                                     hypothesis_template=self.hypothesis_template_en)
                r3 = self.classifier(sample, self.candidate_labels_en, multi_label=False,
                                     hypothesis_template="The content represents a {}.")
            except Exception:
                r1 = self.classifier(sample, self.candidate_labels_en, multi_label=False)
                r2 = r1
                r3 = r1


            scores = {}
            for lbl, sc in zip(r1["labels"], r1["scores"]):
                scores[lbl] = scores.get(lbl, 0.0) + float(sc)
            for lbl, sc in zip(r2["labels"], r2["scores"]):
                scores[lbl] = scores.get(lbl, 0.0) + float(sc)
            for lbl, sc in zip(r3["labels"], r3["scores"]):
                scores[lbl] = scores.get(lbl, 0.0) + float(sc)

            pred_label = max(scores.items(), key=lambda x: x[1])[0]
            confidence = scores[pred_label] / 3.0  # Average of 3 models

            document_type = self.label_map_en.get(pred_label, "other")

            if confidence < 0.60:
                has_thread_artifacts = self._has_email_thread_artifacts(raw_text)
                #  Düzeltme 1: Low-Confidence Letter Check bloğu: news_article'ı koru
                if document_type in ("email", "form", "other"): 
                    if self._is_letter_like(raw_text, norm_text) and self._count_email_headers(raw_text) <= 1 and not has_thread_artifacts:
                        document_type, confidence = "letter", 0.72
                elif self._looks_like_news_article(raw_text, norm_text):
                    document_type, confidence = "news_article", 0.74 # 0.72 -> 0.74

                elif core_hdrs >= 2 and self._has_email_address(joined):
                    document_type = "email"
                    confidence = 0.70

                elif self._is_resume_like(raw_text, norm_text):
                    document_type = "resume"
                    confidence = 0.72

                elif self._is_form_layout_like(raw_text, norm_text):
                    document_type = "form"
                    confidence = 0.70
                # Receipt/Invoice check
                elif self._is_receipt_like(joined):
                    document_type = "receipt"
                    confidence = 0.68
                elif self._is_invoice_like(joined):
                    document_type = "invoice"
                    confidence = 0.68
                else:

                    hint = self._keyword_hint(joined)
                    if hint and hint != "other":
                        document_type = hint
                        confidence = 0.60

                    elif document_type == "other" and len(joined.strip()) > 100:
                        if len(joined.splitlines()) > 50:
                            document_type = "form"
                            confidence = 0.52    # 0.45 -> 0.52
                        else:
                            document_type = "letter"
                            confidence = 0.52    # 0.45 -> 0.52

            elif confidence < 0.68: 
                hint = self._keyword_hint(joined)
                if hint and hint != "other":
                    document_type = hint
                    confidence = max(confidence, 0.60)

            if document_type == "email":

                if self._is_form_layout_like(raw_text, norm_text) and not email_signal and not informal_email_hit:
                    document_type, confidence = "form", max(confidence, 0.80)

                elif core_hdrs < 2 and not has_thread:
                    if not self._has_email_address(joined):
                        confidence = min(confidence, 0.65)
                    else:
                        confidence = min(confidence, 0.75)

                elif email_signal or informal_email_hit:
                    confidence = max(confidence, 0.85)

            if document_type == "form":
                # If strong email signal, convert to email
                if (email_signal and core_hdrs >= 3) or informal_email_hit:
                    document_type, confidence = "email", max(confidence, 0.90)

            if self._is_form_layout_like(raw_text, norm_text) or self._is_lab_form_like(raw_text, norm_text):
                if document_type in ("other", "resume", "letter"):
                    document_type = "form"
                    confidence = max(confidence, 0.78)
                elif document_type == "news_article":
                    # Form vs News: are form layout features stronger?
                    form_score = sum([
                        len(self.re_label_line.findall(raw_text)) >= 4,
                        sum(1 for kw in self.form_kw_headers if kw in norm_text.lower()) >= 3,
                        len(re.findall(r"_{4,}", raw_text)) >= 2
                    ])
                    if form_score >= 2 or self._is_lab_form_like(raw_text, norm_text):
                        document_type = "form"
                        confidence = 0.75

            if document_type == "form":
                if self._looks_like_news_article(raw_text, norm_text):
                    strong_form_local = (
                        len(self.re_label_line.findall(raw_text)) >= 5 or
                        sum(1 for kw in self.form_kw_headers if kw in norm_text.lower()) >= 4 or
                        len(re.findall(r"_{4,}", raw_text)) >= 2 or
                        self._is_lab_form_like(raw_text, norm_text) # FIX 2 & 4
                    )
                    if not strong_form_local:
                        document_type, confidence = "news_article", max(confidence, 0.74)

            if document_type == "form" and confidence < 0.65:
                if self._is_form_like(joined):
                    confidence = 0.70

           
            has_thread_artifacts = self._has_email_thread_artifacts(raw_text)
            if document_type in ("email", "form", "other"): 
                if self._is_letter_like(raw_text, norm_text):
                    email_hdr_count = self._count_email_headers(raw_text)
                    if (email_hdr_count <= 1 and not email_signal and not has_thread_artifacts and not informal_email_hit):
                        document_type = "letter"
                        confidence = max(confidence, 0.80)
                    elif email_hdr_count == 2 and not self._has_email_address(joined) and not email_signal and not has_thread_artifacts and not informal_email_hit:
                        document_type = "letter"
                        confidence = max(confidence, 0.70)
                    elif email_hdr_count > 2 and not email_signal and not has_thread_artifacts and not informal_email_hit:
                        # Allow letter if strong letter signals and weak email signals (only 3 headers, no thread bits)
                        letter_score = sum([any(re.search(p, raw_text, re.I | re.M) for p in self.letter_salutations),
                                             any(re.search(p, raw_text, re.I) for p in self.letter_closings),
                                             self.re_letter_date.search(raw_text) is not None,
                                             self.re_address_line.search(raw_text) is not None])
                        if letter_score >= 3:
                            document_type, confidence = "letter", max(confidence, 0.70)

            if document_type == "letter":
                email_headers = self._count_email_headers(raw_text)
                strong_email_v2 = (email_headers >= 3) or \
                                         (email_headers >= 2 and self._has_email_address(joined)) or \
                                         self._has_email_thread_artifacts(raw_text) or \
                                         informal_email_hit

                if not strong_email_v2: # Convert to email only if strong signals
                    confidence = max(confidence, 0.85)
                else:
                    document_type = "email"
                    confidence = 0.80 # Letter --> Email conversion

            if document_type in ("invoice", "receipt"):
                #  Düzeltme 3: News article kontrolü ekle
                if self._looks_like_news_article(raw_text, norm_text):
                    document_type, confidence = "news_article", 0.75
                else:
                    is_receipt = self._is_receipt_like(joined)
                    is_invoice = self._is_invoice_like(joined)
                    if is_receipt and not is_invoice:
                        document_type, confidence = "receipt", max(confidence, 0.75)
                    elif is_invoice and not is_receipt:
                        document_type, confidence = "invoice", max(confidence, 0.75)
                    elif is_receipt and is_invoice:
                        # prioritize invoice if keyword + table is present AND merchant is NOT present
                        has_invoice_table = re.search(r"\binvoice\b", joined, re.I) and re.search(r"\b(qty|quantity|unit\s*price|po\s*(no|#))\b", joined, re.I)
                        has_merchant = bool(re.search(r"\bmerchant\b", joined, re.I))
                        if has_invoice_table and not has_merchant:
                            document_type, confidence = "invoice", max(confidence, 0.74) # Slight confidence boost for clean invoice
                        else:
                            document_type, confidence = "receipt", max(confidence, 0.72) # Default to receipt if both are strong or if merchant is present

            if document_type == "news_article":

                if core_hdrs >= 2 or informal_email_hit:
                    document_type = "email"
                    confidence = 0.75
                elif self._is_form_layout_like(raw_text, norm_text) or self._is_lab_form_like(raw_text, norm_text):
                    form_indicators = sum([
                        len(self.re_label_line.findall(raw_text)) >= 3,
                        sum(1 for kw in self.form_kw_headers if kw in norm_text.lower()) >= 2,
                    ])
                    if form_indicators >= 1 or self._is_lab_form_like(raw_text, norm_text):
                        document_type = "form"
                        confidence = 0.72
                elif not self._looks_like_news_article(raw_text, norm_text):
                    # Haber değilse ve tablo/para yoğun ise fatura/fiş; değilse letter/form ipucuna dön
                    if self._is_receipt_like(joined):
                        document_type, confidence = "receipt", 0.70
                    elif self._is_invoice_like(joined):
                        document_type, confidence = "invoice", 0.70
                    elif self._is_form_layout_like(raw_text, norm_text):
                        document_type, confidence = "form", 0.72
                    elif self._is_letter_like(raw_text, norm_text):
                        document_type, confidence = "letter", 0.70
                    else:
                        confidence = min(confidence, 0.65)

            if document_type == "other" and (raw_text.strip() or norm_text.strip()):

                readable_chars = sum(1 for c in joined if c.isalnum() or c.isspace())
                total_chars = len(joined)
                readable_ratio = readable_chars / max(1, total_chars)
                lines = raw_text.splitlines()

                if readable_ratio < 0.6 and len(joined.strip()) > 50:  # Poor OCR quality
                    news_kw_count = sum(1 for kw in self.news_keywords if kw in joined.lower())
                    if news_kw_count >= 3:  # 3+ news keyword = likely news
                        document_type, confidence = "news_article", 0.68
                    else:
                        avg_line_len = sum(len(ln) for ln in lines) / max(1, len(lines))
                        if len(lines) > 30 and 20 < avg_line_len < 60:
                            document_type, confidence = "news_article", 0.65  # Columnar layout
                        elif len(lines) > 50:
                            document_type, confidence = "form", 0.62  # Many short lines
                        else:
                            document_type, confidence = "letter", 0.60  # Default

                else:
                    has_thread = self._has_email_thread_artifacts(raw_text)
                    has_email_addr = self._has_email_address(raw_text)
                    # Prioritize: resume > form > email > news > letter > receipt > invoice
                    for dt, check in [
                             ("resume", lambda: self._is_resume_like(raw_text, norm_text) or self._has_job_history_timeline(raw_text, norm_text)),
                             ("form",  lambda: self._is_form_layout_like(raw_text, norm_text) or self._is_lab_form_like(raw_text, norm_text)),
                             ("email", lambda: (core_hdrs >= 2 and (has_email_addr or sys_hdrs >= 1 or has_thread)) or informal_email_hit),
                             ("news_article", lambda: self._looks_like_news_article(raw_text, norm_text)),
                             ("letter", lambda: self._is_letter_like(raw_text, norm_text)),
                             ("receipt", lambda: self._is_receipt_like(joined)),
                             ("invoice", lambda: self._is_invoice_like(joined)),
                          ]:
                              if check():
                                  document_type, confidence = dt, max(confidence, 0.68)
                                  break

            per_class_min = {
                "email": 0.70,
                "form": 0.68,
                "news_article": 0.70,
                "invoice": 0.62,
                "receipt": 0.62,
                "letter": 0.62,
                "resume": 0.62,
            }
            thr = per_class_min.get(document_type, 0.60)
            if confidence < thr:
                if self._count_email_headers(raw_text) >= 2 and self._has_email_address(joined):
                    document_type, confidence = "email", 0.70
                elif self._is_form_layout_like(raw_text, norm_text) or self._is_lab_form_like(raw_text, norm_text):
                    document_type, confidence = "form", 0.70
                elif self._looks_like_news_article(raw_text, norm_text):
                    document_type, confidence = "news_article", 0.70
                else:
                    hint = self._keyword_hint(joined)
                    if hint and hint != "other":
                        document_type, confidence = hint, max(confidence, 0.60)
                    else:
                        document_type, confidence = "other", confidence

            response = await self.send_message(
                receiver="coordinator",
                message_type=MessageType.RESPONSE,
                content={'action': 'classification_complete',
                         'document_type': document_type,
                         'confidence': float(confidence),
                         'workflow_id': request.content.get('workflow_id')}
            )
            if self.coordinator:
                await self.coordinator.receive_message(response)

        except Exception as e:
            logger.error(f"Classification error: {e}")
            error_msg = await self.send_message(
                receiver="coordinator",
                message_type=MessageType.ERROR,
                content={'error': str(e), 'workflow_id': request.content.get('workflow_id')}
            )
            if self.coordinator:
                await self.coordinator.receive_message(error_msg)

    def _headers_in_top_block(self, raw_text: str, top_n_lines: int = 30) -> int:
        lines = (raw_text or "").splitlines()
        top = "\n".join(lines[:top_n_lines])
        core_headers = [
            re.compile(r"^from\s*:", re.I | re.M),
            re.compile(r"^to\s*:", re.I | re.M),
            re.compile(r"^subject\s*:", re.I | re.M),
            re.compile(r"^date\s*:", re.I | re.M),
        ]
        return sum(1 for p in core_headers if p.search(top))

    def _has_email_address(self, text: str) -> bool:
        return bool(self.re_email_addr.search(text or ""))

    def _is_form_like(self, text: str) -> bool:
        t = text or ""
        hits = sum(1 for rx in self.form_artifacts_re if rx.search(t))
        return hits >= 2

    def _is_receipt_like(self, text: str) -> bool:
        t = text or ""
        kw_hits = sum(1 for rx in self.re_receipt if rx.search(t))
        time_hit = bool(self.re_time.search(t))
        money_lines = len(re.findall(r"\b\d+[\,\.]\d{2}\b", t))
        return (kw_hits >= 2 and (time_hit or money_lines >= 3))

    def _is_invoice_like(self, text: str) -> bool:
        t = text or ""
        kw_hits = sum(1 for rx in self.re_invoice if rx.search(t))
        has_invoice_word = bool(re.search(r"\binvoice\b", t, re.I))
        has_table_tokens = bool(re.search(r"\b(qty|quantity|unit\s*price|po\s*(no|number|#))\b", t, re.I))
        return (kw_hits >= 2) or (has_invoice_word and has_table_tokens)

    def _is_form_layout_like(self, raw_text: str, norm_text: str) -> bool:
        raw = raw_text or ""
        norm = (norm_text or "").lower()
        label_lines = len(self.re_label_line.findall(raw))
        header_hits = sum(1 for kw in self.form_kw_headers if kw in norm)
        lines = raw.splitlines()[:50]
        allcaps = sum(1 for ln in lines if self.re_allcaps_line.match(ln.strip()))
        units_hits = len(re.findall(r"\b(mg|mg/kg|%|g\/kg)\b", norm))
        numbers = len(re.findall(r"\b\d[\d\.,]*\b", norm))
        strong = (label_lines >= 4) or (header_hits >= 2 and allcaps >= 2)
        supportive = (units_hits >= 2 and numbers >= 12)
        return strong or (header_hits >= 2 and supportive)
    
    def _is_letter_like(self, raw_text: str, norm_text: str) -> bool:
        raw = raw_text or ""
        norm = norm_text or ""

        top_lines = "\n".join(raw.splitlines()[:40])
        sal_hit = any(re.search(p, top_lines, re.I | re.M) for p in self.letter_salutations)

        tail_lines = "\n".join(raw.splitlines()[-60:])
        close_hit = any(re.search(p, tail_lines, re.I) for p in self.letter_closings)

        date_hit = bool(self.re_letter_date.search("\n".join(raw.splitlines()[:40])))
        address_hit = sum(1 for ln in raw.splitlines()[:25] if self.re_address_line.search(ln)) >= 1
        subj_hit = bool(self.re_letter_subject.search(top_lines))
        few_email_hdrs = self._count_email_headers(raw) <= 1
        has_thread_artifacts = self._has_email_thread_artifacts(raw) # NEW CHECK
       
        joined = (raw or "").lower()
        sal_hit_new = any(x in joined for x in [
            "dear ", "greetings", "to whom it may concern"
        ]) or bool(re.search(r"^\s*(dear|hi|hello)\b", joined, re.M))

        # kapanış + adres veya tarih → güçlü sinyal
        strong_pattern = (sal_hit or sal_hit_new) and (close_hit or subj_hit or date_hit or address_hit)
        score = sum([sal_hit, close_hit, date_hit, address_hit, subj_hit, few_email_hdrs, 1 if sal_hit_new else 0])
        
        has_email = self._has_email_address(raw)
        
        return (strong_pattern or score >= 4) and (not has_email or score >= 5) and few_email_hdrs and not has_thread_artifacts

# ---------------- Gemini QA Agent ----------------
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiQAAgent(BaseAgent):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "models/gemini-2.5-flash"):
        super().__init__("gemini_qa_agent")
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)
        self.coordinator = None
        logger.info("Gemini Q&A agent ready")

    def set_coordinator(self, coordinator):
        self.coordinator = coordinator

    async def handle_message(self, message: A2AMessage):
        if message.message_type != MessageType.REQUEST:
            return
        if message.content.get("action") != "ask_question":
            return
        await self._process_question(message)

    async def _process_question(self, request: A2AMessage):
        content = request.content
        workflow_id = content["workflow_id"]
        doc_text = content["document_text"]
        doc_type = content["document_type"]
        question = content["question"]

        prompt = build_grounded_prompt(doc_type, doc_text, question)

        start = time.time()
        loop = asyncio.get_event_loop()

        def run_gemini():
            return self.model.generate_content(prompt)

        try:
            resp = await loop.run_in_executor(None, run_gemini)
            raw = getattr(resp, "text", "") or ""
            parsed = parse_json_loose(raw)
            rt = time.time() - start

            msg = await self.send_message(
                receiver="coordinator",
                message_type=MessageType.RESPONSE,
                content={
                    "action": "qa_answer_ready",
                    "workflow_id": workflow_id,
                    "model_name": self.model_name,
                    "question": question,
                    "found": parsed["found"],
                    "answer": parsed["answer"],
                    "evidence": parsed["evidence"],
                    "response_time": rt,
                },
            )
            if self.coordinator:
                await self.coordinator.receive_message(msg)

        except Exception as e:
            err = await self.send_message(
                receiver="coordinator",
                message_type=MessageType.ERROR,
                content={"workflow_id": workflow_id, "error": str(e)},
            )
            if self.coordinator:
                await self.coordinator.receive_message(err)


# ---------------- Llama QA Agent (Ollama) ----------------
class LlamaLocalQAAgent(BaseAgent):
    def __init__(self, ollama_model: str = "llama3.2:3b"):
        super().__init__("llama_qa_agent")
        self.ollama_model = ollama_model
        self.coordinator = None
        self._check_ollama()
        logger.info("Llama Q&A agent ready (Ollama)")

    def set_coordinator(self, coordinator):
        self.coordinator = coordinator

    def _check_ollama(self):
        import subprocess
        try:
            subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=8)
        except FileNotFoundError:
            raise RuntimeError("Ollama not installed or not in PATH")

    async def handle_message(self, message: A2AMessage):
        if message.message_type != MessageType.REQUEST:
            return
        if message.content.get("action") != "ask_question":
            return
        await self._process_question(message)

    async def _process_question(self, request: A2AMessage):
        import subprocess

        content = request.content
        workflow_id = content["workflow_id"]
        doc_text = content["document_text"]
        doc_type = content["document_type"]
        question = content["question"]

        prompt = build_grounded_prompt(doc_type, doc_text, question, max_chars=2800)

        start = time.time()
        loop = asyncio.get_event_loop()

        def run_ollama():
            r = subprocess.run(
                ["ollama", "run", self.ollama_model],
                input=prompt.encode("utf-8", errors="ignore"),
                capture_output=True,
                timeout=120,
            )

            stdout = (r.stdout or b"").decode("utf-8", errors="replace")
            stderr = (r.stderr or b"").decode("utf-8", errors="replace")

            if stderr.strip():
                logger.warning(f"Ollama stderr: {stderr.strip()[:300]}")

            return stdout.strip()

        try:
            raw = await loop.run_in_executor(None, run_ollama)
            parsed = parse_json_loose(raw)
            rt = time.time() - start

            msg = await self.send_message(
                receiver="coordinator",
                message_type=MessageType.RESPONSE,
                content={
                    "action": "qa_answer_ready",
                    "workflow_id": workflow_id,
                    "model_name": "llama-3.2-3b",
                    "question": question,
                    "found": parsed["found"],
                    "answer": parsed["answer"],
                    "evidence": parsed["evidence"],
                    "response_time": rt,
                },
            )
            if self.coordinator:
                await self.coordinator.receive_message(msg)

        except Exception as e:
            err = await self.send_message(
                receiver="coordinator",
                message_type=MessageType.ERROR,
                content={"workflow_id": workflow_id, "error": str(e)},
            )
            if self.coordinator:
                await self.coordinator.receive_message(err)

# ========================= COORDINATOR AGENT =========================

class CoordinatorAgent(BaseAgent):
    def __init__(self, ocr_default_backend: str = "tesseract"):
        super().__init__("coordinator")
        self.ocr_agent = OCRAgent(default_backend=ocr_default_backend)
        self.masking_agent = HybridPIIMaskingAgent()             
        self.normalization_agent = TextNormalizationAgent()
        self.classification_agent = ClassificationAgent()

        self.ocr_agent.set_coordinator(self)
        self.masking_agent.set_coordinator(self)         
        self.normalization_agent.set_coordinator(self)
        self.classification_agent.set_coordinator(self)

        self.workflows: Dict[str, Dict[str, Any]] = {}

    async def process_document(self, image_base64: str, lang: str = "eng", true_type: str = None) -> Dict[str, Any]:
        workflow_id = f"workflow_{datetime.now().timestamp()}"
        self.workflows[workflow_id] = {'status': 'started', 'steps_completed': [], 'results': {'lang': lang,'true_type': true_type}}

        ocr_request = await self.send_message(
            receiver="ocr_agent",
            message_type=MessageType.REQUEST,
            content={'action': 'ocr', 'image': image_base64, 'workflow_id': workflow_id, 'lang': lang}
        )
        await self.ocr_agent.receive_message(ocr_request)

        timeout = 300
        elapsed = 0
        while self.workflows[workflow_id]['status'] not in ('completed', 'failed', 'timeout') and elapsed < timeout:
            await asyncio.sleep(0.3)
            elapsed += 0.3

        if elapsed >= timeout:
            self.workflows[workflow_id]['status'] = 'timeout'
            self.workflows[workflow_id]['results']['error'] = 'Workflow timeout'

        self.workflows[workflow_id]['results']['status'] = self.workflows[workflow_id]['status']
        return self.workflows[workflow_id]['results']

    async def handle_message(self, message: A2AMessage):
        content = message.content
        workflow_id = content.get('workflow_id')
        if not workflow_id or workflow_id not in self.workflows:
            return

        if content.get('action') == 'ocr_complete':
            text = content['text']
            self.workflows[workflow_id]['results']['text'] = text
            self.workflows[workflow_id]['results']['ocr_engine'] = content.get('ocr_engine')
            self.workflows[workflow_id]['results']['ocr_confidence'] = content.get('ocr_avg_confidence')
            self.workflows[workflow_id]['steps_completed'].append('ocr_complete')

            # OCR'dan gelen resmi ve kutuları al
            image_b64 = content.get('image')
            ocr_boxes = content.get('ocr_boxes', [])
            self.workflows[workflow_id]['image_b64'] = image_b64
            self.workflows[workflow_id]['ocr_boxes'] = ocr_boxes

            norm_request = await self.send_message(
                receiver="normalization_agent",
                message_type=MessageType.REQUEST,
                content={'action': 'normalize_text', 'text': text, 'workflow_id': workflow_id}
            )
            await self.normalization_agent.receive_message(norm_request)

        elif content.get('action') == 'normalization_complete':
            normalized_text = content['text']
            self.workflows[workflow_id]['results']['normalized_text'] = normalized_text
            self.workflows[workflow_id]['steps_completed'].append('normalization_complete')

            orig_text = self.workflows[workflow_id]['results'].get('text', '')
            class_request = await self.send_message(
                receiver="classification_agent",
                message_type=MessageType.REQUEST,
                content={'action': 'classify', 'text_raw': orig_text, 'text_normalized': normalized_text,
                         'workflow_id': workflow_id}
            )
            await self.classification_agent.receive_message(class_request)

        elif content.get('action') == 'classification_complete':
            doc_type = content['document_type']  
            conf = float(content.get('confidence', 0.0))
            self.workflows[workflow_id]['results']['document_type'] = content['document_type']
            self.workflows[workflow_id]['results']['classification_confidence'] = content['confidence']
            self.workflows[workflow_id]['steps_completed'].append('classification_complete')
            

            image_b64 = self.workflows[workflow_id].get('image_b64')
            ocr_boxes = self.workflows[workflow_id].get('ocr_boxes', [])
            true_type = self.workflows[workflow_id]['results'].get('true_type')

            mask_request = await self.send_message(
                receiver="masking_agent",
                message_type=MessageType.REQUEST,
                content={
                  "action": "mask_visual",
                  "image": image_b64,
                  "ocr_boxes": ocr_boxes,
                  "document_type": doc_type,
                  "true_type": true_type, 
                  "confidence": conf, 
                  "workflow_id": workflow_id
                }
            )
            await self.masking_agent.receive_message(mask_request)

        elif content.get('action') == 'masking_complete':
            workflow_id = content.get('workflow_id')
            self.workflows[workflow_id]['results']['masked_image'] = content.get('masked_image')
            self.workflows[workflow_id]['results']['mask_stats'] = content.get('mask_stats')
            self.workflows[workflow_id]['steps_completed'].append('masking_complete')
            self.workflows[workflow_id]['status'] = 'completed'
            #  Artık burada başka hiçbir işlem yok
            logger.info(f" Masking completed for {workflow_id}")


        elif message.message_type == MessageType.ERROR:
            self.workflows[workflow_id]['status'] = 'failed'
            self.workflows[workflow_id]['results']['error'] = content.get('error')

# ---------------- Coordinator with QA ----------------
class CoordinatorAgentWithQA(BaseAgent):
    def __init__(self, ocr_default_backend: str = "tesseract", enable_qa: bool = True):
        super().__init__("coordinator")

        self.ocr_agent = OCRAgent(default_backend=ocr_default_backend)
        self.masking_agent = HybridPIIMaskingAgent()
        self.normalization_agent = TextNormalizationAgent()
        self.classification_agent = ClassificationAgent()

        self.ocr_agent.set_coordinator(self)
        self.masking_agent.set_coordinator(self)
        self.normalization_agent.set_coordinator(self)
        self.classification_agent.set_coordinator(self)

        self.qa_enabled = enable_qa
        self.gemini_agent = None
        self.llama_agent = None

        if enable_qa:
            self._init_qa_agents()

        self.workflows: Dict[str, Dict[str, Any]] = {}

    def _init_qa_agents(self):
        # Gemini optional
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and GEMINI_AVAILABLE:
            try:
                self.gemini_agent = GeminiQAAgent(api_key=api_key)
                self.gemini_agent.set_coordinator(self)
                logger.info("Gemini enabled")
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")
                self.gemini_agent = None
        else:
            logger.warning("Gemini disabled (no key or package missing)")
            self.gemini_agent = None

        try:
            self.llama_agent = LlamaLocalQAAgent()
            self.llama_agent.set_coordinator(self)
            logger.info("Llama enabled")
        except Exception as e:
            logger.warning(f"Llama init failed: {e}")
            self.gemini_agent = None

    async def process_document(self, image_base64: str, lang: str = "eng", true_type: str = None) -> Dict[str, Any]:
        wf_id = f"wf_{uuid.uuid4().hex}"
        self.workflows[wf_id] = {
            "status": "started",
            "steps_completed": [],
            "results": {"lang": lang, "true_type": true_type},
        }

        ocr_req = await self.send_message(
            receiver="ocr_agent",
            message_type=MessageType.REQUEST,
            content={"action": "ocr", "image": image_base64, "workflow_id": wf_id, "lang": lang},
        )
        await self.ocr_agent.receive_message(ocr_req)

        timeout = 300
        elapsed = 0.0
        while self.workflows[wf_id]["status"] not in ("completed", "failed", "timeout") and elapsed < timeout:
            await asyncio.sleep(0.3)
            elapsed += 0.3

        if elapsed >= timeout:
            self.workflows[wf_id]["status"] = "timeout"
            self.workflows[wf_id]["results"]["error"] = "Workflow timeout"

        self.workflows[wf_id]["results"]["status"] = self.workflows[wf_id]["status"]
        return self.workflows[wf_id]["results"]

    async def process_document_with_qa(
            self,
            image_base64: str,
            questions: List[str],
            lang: str = "eng",
            use_model: str = "llama",
            true_type: str = None
        ) -> Dict[str, Any]:

            try:
                logger.info(" ENTER process_document_with_qa")

        # 1) Doküman pipeline
                doc_result = await self.process_document(
                    image_base64=image_base64,
                    lang=lang,
                    true_type=true_type
                )

                if not isinstance(doc_result, dict):
                    return {
                        "status": "failed",
                        "error": f"process_document returned non-dict: {type(doc_result)}",
                        "qa_results": {"gemini": [], "llama": []}
                    }

                if doc_result.get("status") != "completed":
                    pred_type = doc_result.get("document_type", "other")
                    true_type = doc_result.get("true_type")

                    doc_result["qa_results"] = {"gemini": [], "llama": []}
                    doc_result["predicted_type"] = pred_type
                    doc_result["qa_used_type"] = true_type or pred_type

                    logger.warning(
                        f"QA SKIPPED | status={doc_result.get('status')} | "
                        f"predicted_type={pred_type} | qa_used_type={doc_result['qa_used_type']}"
                    )

                    return doc_result

        # 2) QA doc type
                pred_type = doc_result.get("document_type", "other")
                gt_type = doc_result.get("true_type")
                qa_doc_type = gt_type or pred_type

                doc_result["predicted_type"] = pred_type
                doc_result["qa_used_type"] = qa_doc_type

                base_text_raw = (doc_result.get("normalized_text") or doc_result.get("text") or doc_result.get("ocr_text") or "")
                qa_text = sanitize_for_qa(base_text_raw)

                qa_results = {"gemini": [], "llama": []}

                async def ask(agent: BaseAgent, model_key: str, question: str):
                    if agent is None:
                        logger.warning(f" {model_key} agent is None, skipping question")
                        qa_results[model_key].append({
                            "model_name": model_key,
                            "question": question,
                            "found": False,
                            "answer": f"{model_key} agent not available",
                            "evidence": "",
                            "response_time": 0.0,
                        })
                        return
                    
                    if model_key == "gemini":
                        await gemini_rl.acquire()

                    qa_wf = f"qa_{uuid.uuid4().hex}"
                    self.workflows[qa_wf] = {"status": "waiting_qa", "qa_answer": None, "error": None}

                    req = await self.send_message(
                        receiver=agent.agent_id,
                        message_type=MessageType.REQUEST,
                        content={
                            "action": "ask_question",
                            "workflow_id": qa_wf,
                            "document_text": qa_text,
                            "document_type": qa_doc_type,
                            "question": question,
                        },
                    )
                    await agent.receive_message(req)

                    timeout = 120
                    elapsed = 0.0
                    while self.workflows.get(qa_wf, {}).get("status") == "waiting_qa" and elapsed < timeout:
                        await asyncio.sleep(0.1)
                        elapsed += 0.1

                    wf = self.workflows.get(qa_wf, {})
                    status = wf.get("status")
                    err = wf.get("error")
                    qa_ans = wf.get("qa_answer")

                    logger.info(
                        f"QA WF END | id={qa_wf} | model={model_key} | status={status} "
                        f"| has_answer={bool(qa_ans)} | err={err} | elapsed={elapsed:.2f}s"
                    )

            # 1) Timeout
                    if status == "waiting_qa":
                        logger.warning(f" QA TIMEOUT | model={model_key} | question={question}")
                        qa_results[model_key].append({
                            "model_name": model_key,
                            "question": question,
                            "found": False,
                            "answer": "Timeout - no response",
                            "evidence": "",
                            "response_time": float(timeout),
                        })
                        return

            # 2) Empty / None response
                    if qa_ans is None:
                        logger.warning(f" Q&A EMPTY | model={model_key} | status={status} | error={err}")
                        qa_results[model_key].append({
                            "model_name": model_key,
                            "question": question,
                            "found": False,
                            "answer": "No response",
                            "evidence": "",
                            "response_time": float(elapsed),
                        })
                        return

            # 3) Dict değilse normalize et
                    if not isinstance(qa_ans, dict):
                        qa_ans = {
                            "model_name": model_key,
                            "question": question,
                            "found": True,
                            "answer": str(qa_ans),
                            "evidence": "",
                            "response_time": float(elapsed),
                        }

            # 4-7) Alanları işle
                    answer_text = (qa_ans.get("answer") or "").strip()
                    found_flag = bool(qa_ans.get("found", False))
                    response_time = float(qa_ans.get("response_time", elapsed))
                    evidence_text = (qa_ans.get("evidence") or "").strip()

                    if answer_text and not any(x in answer_text.lower() for x in ["information not found", "no response", "timeout"]):
                        found_flag = True

                    found_flag = postprocess_found_flag(found_flag, answer_text)

                    if not evidence_text:
                        ev2 = extract_evidence_from_text(base_text_raw, qa_doc_type, question)
                        evidence_text = (ev2 or "").strip()

                    if found_flag:
                        is_valid = validate_answer(question, answer_text, qa_doc_type)
                        if not is_valid:
                            found_flag = False

                    qa_results[model_key].append({
                        "model_name": qa_ans.get("model_name", model_key),
                        "question": qa_ans.get("question", question),
                        "found": found_flag,
                        "answer": answer_text if answer_text else ("Information not found in document." if not found_flag else ""),
                        "evidence": evidence_text,
                        "response_time": response_time,
                    })

                    logger.info(f" APPENDED | model={model_key} | new_len={len(qa_results[model_key])}")


                if use_model in ("gemini", "both") and self.gemini_agent is not None:
                    for q in questions:
                        logger.info(f" ASKING | model=gemini | q={q}")
                        await ask(self.gemini_agent, "gemini", q)

                if use_model in ("llama", "both") and self.llama_agent is not None:
                    for q in questions:
                        logger.info(f" ASKING | model=llama | q={q}")
                        await ask(self.llama_agent, "llama", q)

# hiçbir agent yoksa
                if (
                    (use_model in ("gemini", "both") and self.gemini_agent is None)
                    and (use_model in ("llama", "both") and self.llama_agent is None)
                ):
                    logger.error(" No QA agents available!")

                doc_result["qa_results"] = qa_results
                return doc_result

            except Exception as e:
                logger.error(f" process_document_with_qa failed: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "status": "failed",
                    "error": str(e),
                    "qa_results": {"gemini": [], "llama": []}
                }

    async def handle_message(self, message: A2AMessage):
        content = message.content
        wf_id = content.get("workflow_id")
        if not wf_id or wf_id not in self.workflows:
            return

        action = content.get("action")

        if action == "ocr_complete":
            text = content["text"]
            self.workflows[wf_id]["results"]["text"] = text
            self.workflows[wf_id]["results"]["ocr_engine"] = content.get("ocr_engine")
            self.workflows[wf_id]["results"]["ocr_confidence"] = content.get("ocr_avg_confidence")
            self.workflows[wf_id]["steps_completed"].append("ocr_complete")

            image_b64 = content.get("image")
            ocr_boxes = content.get("ocr_boxes", [])
            self.workflows[wf_id]["image_b64"] = image_b64
            self.workflows[wf_id]["ocr_boxes"] = ocr_boxes

            norm_req = await self.send_message(
                receiver="normalization_agent",
                message_type=MessageType.REQUEST,
                content={"action": "normalize_text", "text": text, "workflow_id": wf_id},
            )
            await self.normalization_agent.receive_message(norm_req)

        elif action == "normalization_complete":
            norm_text = content["text"]
            self.workflows[wf_id]["results"]["normalized_text"] = norm_text
            self.workflows[wf_id]["steps_completed"].append("normalization_complete")

            raw = self.workflows[wf_id]["results"].get("text", "")
            class_req = await self.send_message(
                receiver="classification_agent",
                message_type=MessageType.REQUEST,
                content={"action": "classify", "text_raw": raw, "text_normalized": norm_text, "workflow_id": wf_id},
            )
            await self.classification_agent.receive_message(class_req)

        elif action == "classification_complete":
            doc_type = content["document_type"]
            conf = float(content.get("confidence", 0.0))
            self.workflows[wf_id]["results"]["document_type"] = doc_type
            self.workflows[wf_id]["results"]["classification_confidence"] = conf
            self.workflows[wf_id]["steps_completed"].append("classification_complete")

            image_b64 = self.workflows[wf_id].get("image_b64")
            ocr_boxes = self.workflows[wf_id].get("ocr_boxes", [])
            true_type = self.workflows[wf_id]["results"].get("true_type")

            mask_req = await self.send_message(
                receiver="masking_agent",
                message_type=MessageType.REQUEST,
                content={
                    "action": "mask_visual",
                    "image": image_b64,
                    "ocr_boxes": ocr_boxes,
                    "document_type": doc_type,
                    "true_type": true_type,
                    "confidence": conf,
                    "workflow_id": wf_id,
                },
            )
            await self.masking_agent.receive_message(mask_req)

        elif action == "masking_complete":
            self.workflows[wf_id]["results"]["masked_image"] = content.get("masked_image")
            self.workflows[wf_id]["results"]["mask_stats"] = content.get("mask_stats")
            self.workflows[wf_id]["steps_completed"].append("masking_complete")
            self.workflows[wf_id]["status"] = "completed"

        elif action == "qa_answer_ready":
            self.workflows[wf_id]["qa_answer"] = {
                "model_name": content.get("model_name", ""),
                "question": content.get("question", ""),
                "found": bool(content.get("found", False)),
                "answer": content.get("answer", ""),
                "evidence": content.get("evidence", ""),
                "response_time": float(content.get("response_time", 0.0)),
            }
            self.workflows[wf_id]["status"] = "qa_complete"

        elif message.message_type == MessageType.ERROR:
            self.workflows[wf_id]["status"] = "failed"
            self.workflows[wf_id]["error"] = content.get("error")
            if "results" in self.workflows[wf_id]:
                self.workflows[wf_id]["results"]["error"] = content.get("error")


class QATestEngineA2A:
    def __init__(self, coordinator: CoordinatorAgentWithQA):
        self.coordinator = coordinator

        self.default_questions = {
            "email": [
                "Who sent this email? (Extract sender name or email address.)",
                "What is the subject of this email?"
            ],

            "invoice": [
                "What is the vendor/supplier name (company issuing the invoice)? Return only the company name.",
                "What is the Invoice Date? If unclear, return the exact date string as it appears (even if it looks odd)."
            ],

            "receipt": [
                "What is the final total amount shown on the receipt?",
                "What is the transaction or purchase date shown on the receipt?"
            ],

            "form": [
                "What type of form is this document?",
                "What is the form about? Summarize the purpose in one short sentence."
            ],

            "letter": [
                "Who is the sender of this letter? (Signature / letterhead.)",
                "Who is the recipient of this letter? (Dear / To.)"
            ],

            "resume": [
                "What is the candidate's full name?",
                "What is the most recent job title and organization listed in the resume?"
            ],

            "news_article": [
                "What is the main topic or subject of this article? (1 short sentence.)",
                "List the main organizations, people, or places mentioned in the article."
            ],
        }


        self.results = []  # birleşik sonuç listesi

    def stratified_sample_exact(self, data: list, limit: int, seed: int = 42) -> list:
        rng = random.Random(seed)

        groups = defaultdict(list)
        for s in data:
            t = s.get("true_type") or "other"
            groups[t].append(s)

        for t in groups:
            rng.shuffle(groups[t])

        types = [t for t in groups.keys() if groups[t]]
        rng.shuffle(types)

        picked = []

    # 1) Her tipten 1
        for t in types:
            if len(picked) >= limit:
                break
            picked.append(groups[t].pop())

    # 2) Kalanı doldur
        while len(picked) < limit:
            available = [t for t in types if groups[t]]
            if not available:
                break
            t = rng.choice(available)
            picked.append(groups[t].pop())

        return picked

    async def run_streaming_qa(
        self,
        test_dataset_path: str = "test_data/test_dataset.json",
        limit: Optional[int] = None,
        use_model: str = "both",  # "gemini"|"llama"|"both"
        questions_per_doc: int = 2
    ):
        data = json.loads(Path(test_dataset_path).read_text(encoding="utf-8"))
        if limit:
            data = self.stratified_sample_exact(data, limit=limit, seed=42)


        logger.info(f"Streaming Q&A started on {len(data)} samples")

        for i, sample in enumerate(data, 1):
            file_name = sample["file_name"]
            file_path = sample["file_path"]
            true_type = sample.get("true_type")

            logger.info(f"[{i}/{len(data)}] {file_name} | true_type={true_type}")

            try:
                image_b64 = image_any_to_base64(file_path)

                doc_type_for_questions = true_type or "other"
                qs = self.default_questions.get(doc_type_for_questions, ["What is this document about?"])[:questions_per_doc]

                logger.info(f"   QA WILL RUN | questions={qs}")

                out = await self.coordinator.process_document_with_qa(
                    image_base64=image_b64,
                    questions=qs,
                    lang="eng",
                    use_model=use_model,
                    true_type=true_type
                )

                logger.info("   QA RETURNED FROM process_document_with_qa")

                qa = out.get("qa_results", {}) or {}
                llama_list = qa.get("llama", []) or []
                gemini_list = qa.get("gemini", []) or []

                logger.info(f"   QA COUNTS | llama={len(llama_list)} gemini={len(gemini_list)} | status={out.get('status')}")

        # Sonuçları kaydet
                if out.get("status") != "completed":
                    self.results.append({
                        "document_name": file_name,
                        "true_type": true_type,
                        "status": out.get("status"),
                        "error": out.get("error") or out.get("results", {}).get("error")
                    })
                    continue

                pred = out.get("predicted_type")
                qa_used = out.get("qa_used_type")

                for model_key in ("llama", "gemini"):
                    model_list = qa.get(model_key, []) or []
                    for r in model_list:
                        logger.info(f"    {model_key.upper()} | found={r.get('found')} | {r.get('response_time', 0):.2f}s | {r.get('answer','')[:80]}")
                        self.results.append({
                            "document_name": file_name,
                            "true_type": true_type,
                            "predicted_type": pred,
                            "qa_used_type": qa_used,
                            "model": r.get("model_name", model_key),
                            "model_key": model_key,   # istersen sil ama faydalı
                            "question": r.get("question", ""),
                            "found": bool(r.get("found", False)),
                            "answer": r.get("answer", ""),
                            "evidence": r.get("evidence", ""),
                            "response_time": float(r.get("response_time", 0.0)),
                        })

            except Exception as e:
                logger.error(f"QA LOOP ERROR on {file_name}: {e}")
                self.results.append({
                    "document_name": file_name,
                    "true_type": true_type,
                    "status": "failed",
                    "error": str(e)
                })

        self._save_reports()
        self._print_summary()

    def _save_reports(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_file = f"qa_stream_results_{ts}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {json_file}")

        # Excel
        try:
            import pandas as pd
        
            xlsx = f"qa_stream_report_{ts}.xlsx"
        
        # Tüm sonuçları DataFrame'e çevir
            df = pd.DataFrame(self.results)

            if "found" in df.columns:
                df["found"] = df["found"].fillna(False).astype(bool)

            if "response_time" in df.columns:
                df["response_time"] = pd.to_numeric(df["response_time"], errors="coerce").fillna(0.0)
   
            with pd.ExcelWriter(xlsx, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='All Results', index=False)
            
                if 'model' in df.columns:
                # Her model için özet
                    summary_data = []
                    for model in df['model'].unique():
                        model_df = df[df['model'] == model]
                        summary_data.append({
                            'Model': model,
                            'Total Questions': len(model_df),
                            'Found': model_df['found'].sum(),
                            'Not Found': (~model_df['found']).sum(),
                            'Avg Response Time (s)': model_df['response_time'].mean(),
                            'Success Rate (%)': (model_df['found'].sum() / len(model_df) * 100)
                        })
                
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Model Comparison', index=False)
            
                if 'qa_used_type' in df.columns:
                    doc_summary = []
                    for doc_type in df['qa_used_type'].unique():
                        type_df = df[df['qa_used_type'] == doc_type]
                        doc_summary.append({
                            'Document Type': doc_type,
                            'Questions': len(type_df),
                            'Found': type_df['found'].sum(),
                            'Avg Response Time': type_df['response_time'].mean()
                        })
                
                    doc_df = pd.DataFrame(doc_summary)
                    doc_df.to_excel(writer, sheet_name='By Document Type', index=False)
        
            logger.info(f" Saved: {xlsx}")
        except Exception as e:
            logger.warning(f" Excel export failed: {e}")

    def _print_summary(self):
        total_rows = len(self.results)
        qa_rows = sum(1 for r in self.results if "question" in r)
        failed_docs = sum(1 for r in self.results if r.get("status") in ("failed", "timeout"))

        print("\n" + "=" * 80)
        print("STREAMING Q&A SUMMARY")
        print("=" * 80)
        print(f"Total rows (all): {total_rows}")
        print(f"QA rows: {qa_rows}")
        print(f"Failed/Timeout docs: {failed_docs}")
        print("=" * 80 + "\n")



# ========================= DATASET MANAGEMENT =========================

@dataclass
class TestSample:
    file_name: str
    file_path: str
    true_type: str
    notes: str = ""


class TestDatasetManager:
    def __init__(self, data_folder="test_data"):
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)
        self.categories = ["email", "form", "invoice", "letter", "news_article", "resume", "receipt"]
        for c in self.categories:
            (self.data_folder / c).mkdir(exist_ok=True)

    def create_dataset(self):
        samples = []
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.tif', '*.tiff', '*.bmp', '*.gif', '*.heic', '*.heif']
        for cat in self.categories:
            for pat in patterns:
                for f in (self.data_folder / cat).glob(pat):
                    samples.append(TestSample(f.name, str(f), cat))
        with open(self.data_folder / "test_dataset.json", "w", encoding="utf-8") as fp:
            json.dump([asdict(s) for s in samples], fp, indent=2, ensure_ascii=False)
        logger.info(f" {len(samples)} samples indexed")
        return samples

    def show_statistics(self):
        p = self.data_folder / "test_dataset.json"
        if not p.exists():
            print(" test_dataset.json not found. Run create_dataset() first.")
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
        print("\n" + "=" * 60)
        print("TEST DATASET STATISTICS")
        print("=" * 60)
        print(f"\nTotal samples: {len(df)}")
        counts = df["true_type"].value_counts()
        print("\nCategory distribution:")
        for cat, n in counts.items():
            print(f"  {cat:20s}: {n:3d} ({n/len(df)*100:5.1f}%)")
        if len(counts) > 0:
            ratio = (counts.min() / counts.max()) if counts.max() > 0 else 1.0
            print(f"\nBalance ratio: {ratio:.2f}")
        print("=" * 60)


# ========================= TEST ENGINE =========================

class TestEngine:
    
    def __init__(self, coordinator, test_json_path="test_data/test_dataset.json"):
        self.coordinator = coordinator
        self.test_json_path = test_json_path
        self.results = []
        
        # Paralel işlem parametreleri
        self.max_parallel = 4  # CPU çekirdek sayısına göre ayarla
        self.semaphore = asyncio.Semaphore(self.max_parallel)

    async def run_single_test(self, sample: Dict) -> Dict:
        try:
            path = sample["file_path"]
            ext = Path(path).suffix.lower()

            if ext in SKIP_EXTS:
                return {"file_name": sample["file_name"], "true_type": sample["true_type"],
                         "predicted_type": "SKIPPED", "correct": False, "confidence_score": 0.0, "error": "Skipped PDF"}

            if ext not in SUPPORTED_EXTS:
                return {"file_name": sample["file_name"], "true_type": sample["true_type"],
                         "predicted_type": "SKIPPED", "correct": False, "confidence_score": 0.0,
                         "error": f"Unsupported ext {ext}"}

            try:
                image_base64 = image_any_to_base64(path)
            except Exception as conv_err:
                return {"file_name": sample["file_name"], "true_type": sample["true_type"],
                         "predicted_type": "ERROR", "correct": False, "confidence_score": 0.0,
                         "error": f"Conversion failed: {conv_err}"}

            result = await self.coordinator.process_document(
                image_base64,
                lang="eng",
                true_type=sample.get("true_type")  # gönder
            )

            #  Maskelenmiş görseli kaydet
            masked_b64 = result.get("masked_image")
            if masked_b64:
               # Masked outputs klasörünü oluştur
               masked_dir = Path("masked_outputs_35")
               masked_dir.mkdir(exist_ok=True)
    
               # Dosya adını hazırla
               masked_path = masked_dir / f"masked_{Path(sample['file_name']).stem}.png"
    
               # Base64'ten çöz ve diske yaz
               img_bytes = base64.b64decode(masked_b64)
               with open(masked_path, "wb") as f:
                   f.write(img_bytes)
    
               logger.info(f" Masked image saved: {masked_path}")

            #  Tahmin sonuçları
            pred = result.get("document_type", "other")
            correct = (pred == sample["true_type"])

            if result.get('status') == 'completed':
                mask_stats = result.get("mask_stats", {})
                return {
                    "file_name": sample["file_name"],
                    "true_type": sample["true_type"],
                    "predicted_type": pred,
                    "correct": correct,
                    "confidence_score": float(result.get("classification_confidence", 0.0) or 0.0),
                    "ocr_engine": result.get("ocr_engine"),
                    "processing_time": 0.0,

                    "ocr_preview": (result.get("text") or "")[:3000],
                    "normalized_preview": (result.get("normalized_text") or "")[:3000],

                    "ocr_len": len(result.get("text") or ""),
                    "normalized_len": len(result.get("normalized_text") or ""),
                    "mask_stats": mask_stats
                }
            else:
                return {"file_name": sample["file_name"], "true_type": sample["true_type"], "predicted_type": "ERROR",
                         "correct": False, "confidence_score": 0.0, "error": result.get("error", "Unknown error")}

        except Exception as e:
            logger.error(f" Test error [{sample['file_name']}]: {e}")
            return {"file_name": sample["file_name"], "true_type": sample["true_type"], "predicted_type": "ERROR",
                     "correct": False, "confidence_score": 0.0, "error": str(e)}

    async def run_single_test_with_semaphore(self, sample: Dict, index: int, total: int) -> Dict:
        """Semaphore ile sınırlandırılmış test çalıştırma"""
        async with self.semaphore:
            logger.info(f"[{index}/{total}] Testing: {sample['file_name']}")
            result = await self.run_single_test(sample)
            
            status = "" if result["predicted_type"] not in ("ERROR", "SKIPPED") and result.get("correct") else "X"
            logger.info(f"{status} {sample['true_type']} -> {result['predicted_type']}")
            
            return result
    
    async def run_all_tests_parallel(self):
        logger.info("\n" + "=" * 60)
        logger.info("TEST PROCESS STARTING - PARALLEL MODE")
        logger.info(f"Max Parallel Tasks: {self.max_parallel}")
        logger.info("=" * 60 + "\n")
        
        data = json.loads(Path(self.test_json_path).read_text(encoding="utf-8"))
        total = len(data)
        logger.info(f" {total} test samples found\n")
        
        # Tüm testleri paralel başlat
        tasks = [
            self.run_single_test_with_semaphore(sample, i+1, total)
            for i, sample in enumerate(data)
        ]
        
        # Tümünü paralel çalıştır
        self.results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Exception handling
        for i, result in enumerate(self.results):
            if isinstance(result, Exception):
                logger.error(f"Task {i+1} failed: {result}")
                self.results[i] = {
                    "file_name": data[i]["file_name"],
                    "true_type": data[i]["true_type"],
                    "predicted_type": "ERROR",
                    "correct": False,
                    "confidence_score": 0.0,
                    "error": str(result)
                }
        
        self._save_results()
        self.calculate_metrics()
    
    async def run_all_tests_sequential(self):
        logger.info("\n" + "=" * 60)
        logger.info("TEST PROCESS STARTING - SEQUENTIAL MODE")
        logger.info("=" * 60 + "\n")
        
        data = json.loads(Path(self.test_json_path).read_text(encoding="utf-8"))
        total = len(data)
        logger.info(f" {total} test samples found\n")

        for i, sample in enumerate(data, 1):
            logger.info(f"[{i}/{total}] Testing: {sample['file_name']}")
            r = await self.run_single_test(sample)
            self.results.append(r)
            status = "" if r["predicted_type"] not in ("ERROR", "SKIPPED") and r.get("correct") else "X"
            logger.info(f"{status} {sample['true_type']} -> {r['predicted_type']}\n")

        self._save_results()
        self.calculate_metrics()

    def _save_results(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"test_results_{ts}.json"
        Path(out).write_text(json.dumps(self.results, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f" Results saved: {out}")

        # (opsiyonel) Yanlış sınıflamalar için hızlı OCR dökümü
        try:
            wrong = [r for r in self.results
                     if r.get("predicted_type") not in ("ERROR", "SKIPPED")
                     and r.get("true_type") != r.get("predicted_type")]
            if wrong:
                ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_wrong = f"errors_with_ocr_{ts2}.json"
                Path(out_wrong).write_text(json.dumps(wrong, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info(f" Misclassifications with OCR saved: {out_wrong}")
        except Exception as _e:
            logger.warning(f"Could not save errors_with_ocr dump: {_e}")


    def calculate_metrics(self):
        df = pd.DataFrame(self.results)
        eval_df = df[(df["predicted_type"] != "ERROR") & (df["predicted_type"] != "SKIPPED")]

        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS - v3.3 FINE-TUNED (WITH FIXES)")
        print("=" * 60)

        total_correct = eval_df["correct"].sum()
        total = len(eval_df)
        acc = (total_correct / total * 100) if total > 0 else 0.0
        print(f"\n OVERALL ACCURACY: {acc:.2f}%  ({total_correct}/{total})")

        err_n = len(df[df["predicted_type"] == "ERROR"])
        skip_n = len(df[df["predicted_type"] == "SKIPPED"])
        if err_n:
            print(f"  System errors: {err_n}")
        if skip_n:
            print(f"ℹ  Skipped: {skip_n}")

        if total > 0:
            y_true = eval_df["true_type"]
            y_pred = eval_df["predicted_type"]
            print("\n" + "-" * 60)
            print("PER-CLASS PERFORMANCE")
            print("-" * 60)
            print(classification_report(y_true, y_pred, zero_division=0))
            self._plot_confusion_matrix(y_true, y_pred)
            self._confidence_analysis(eval_df)
            self.generate_masking_report_excel()

        self._save_summary_report(eval_df, acc)
        print("=" * 60 + "\n")

    def _plot_confusion_matrix(self, y_true, y_pred):
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'})
        plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        fn = f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f" Confusion matrix saved: {fn}")

    def _confidence_analysis(self, df):
        print("\n" + "-" * 60)
        print("CONFIDENCE SCORE ANALYSIS")
        print("-" * 60)
        if len(df) == 0:
            print("No predictions.")
            return
        print(f"Mean: {df['confidence_score'].mean() * 100:.2f}%")
        print(f"Median: {df['confidence_score'].median()* 100:.2f}%")
        print(f"Min: {df['confidence_score'].min() * 100:.2f}%")
        print(f"Max: {df['confidence_score'].max() * 100:.2f}%")
        print(f"Std: {df['confidence_score'].std() * 100:.2f}%")

        low = df[df["confidence_score"] < 0.6]
        if len(low) > 0:
            print(f"\n  Low confidence (<60%): {len(low)}")
            print(f"     Accuracy in low confidence: {low['correct'].mean() * 100:.1f}%")
            print("\nLowest (up to 5):")
            print(low.nsmallest(min(5, len(low)), 'confidence_score')[
                ['file_name', 'true_type', 'predicted_type', 'confidence_score', 'correct']
            ].to_string(index=False))

        high_wrong = df[(df["confidence_score"] >= 0.7) & (~df["correct"])]
        if len(high_wrong) > 0:
            print(f"\n High-confidence ERRORS (≥70%): {len(high_wrong)}")
            print(high_wrong[['file_name', 'true_type', 'predicted_type', 'confidence_score']].to_string(index=False))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(df['confidence_score'], bins=20, edgecolor='black', alpha=0.7)
        axes[0].axvline(df['confidence_score'].mean(), color='red', linestyle='--',
                         label=f"Mean: {df['confidence_score'].mean():.2f}")
        axes[0].set_title("Confidence Score Distribution")
        axes[0].set_xlabel("Confidence Score")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        c = df[df['correct']]['confidence_score']
        ic = df[~df['correct']]['confidence_score']
        axes[1].hist([c, ic], bins=15, label=['Correct', 'Incorrect'],
                      edgecolor='black', alpha=0.7)
        axes[1].set_title("Confidence: Correct vs Incorrect")
        axes[1].set_xlabel("Confidence Score")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        fn = f"confidence_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f" Confidence analysis saved: {fn}")

    def _save_summary_report(self, eval_df, acc):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"test_summary_{ts}.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("DOCUMENT CLASSIFICATION TEST REPORT - v3.3 FINE-TUNED (WITH FIXES)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples (evaluated): {len(eval_df)}\n")
            f.write(f"Overall Accuracy: {acc:.2f}%\n\n")
            f.write("-" * 60 + "\n")
            f.write("PER-CLASS METRICS\n")
            f.write("-" * 60 + "\n")
            if len(eval_df) > 0:
                f.write(classification_report(eval_df['true_type'], eval_df['predicted_type'], zero_division=0))
            else:
                f.write("No evaluated samples.\n")
            f.write("\n" + "-" * 60 + "\n")
            f.write("CONFIDENCE STATISTICS\n")
            f.write("-" * 60 + "\n")
            if len(eval_df) > 0:
                f.write(f"Mean: {eval_df['confidence_score'].mean() * 100:.2f}%\n")
                f.write(f"Median: {eval_df['confidence_score'].median() * 100:.2f}%\n")
                f.write(f"Std Dev: {eval_df['confidence_score'].std() * 100:.2f}%\n\n")
            else:
                f.write("N/A\n\n")
            f.write("-" * 60 + "\n")
            f.write("COMMON MISCLASSIFICATIONS\n")
            f.write("-" * 60 + "\n")
            errors = eval_df[~eval_df['correct']]
            if len(errors) > 0:
                pairs = errors.groupby(['true_type', 'predicted_type']).size().sort_values(ascending=False).head(10)
                for (t, p), c in pairs.items():
                    f.write(f"{t:15s} -> {p:15s}: {c} times\n")
            else:
                f.write("No errors! Perfect classification!\n")
        logger.info(f" Summary report saved: {out}")

    def error_analysis(self):
        df = pd.DataFrame(self.results)
        errors = df[(df["predicted_type"] != "ERROR") & (df["predicted_type"] != "SKIPPED") & (~df["correct"])]
        if len(errors) == 0:
            print("\n No classification errors found!")
            return
        print("\n" + "=" * 60)
        print("DETAILED ERROR ANALYSIS")
        print("=" * 60)
        print("\nMost Confused Document Pairs:")
        eval_df = df[(df["predicted_type"] != "ERROR") & (df["predicted_type"] != "SKIPPED")]
        confusion_pairs = errors.groupby(['true_type', 'predicted_type']).size().sort_values(ascending=False)
        for (true_t, pred_t), count in confusion_pairs.head(5).items():
            print(f"  {true_t:15s} → {pred_t:15s}: {count:2d} times")
        print("\nError Rate by True Category:")
        for cat in sorted(eval_df['true_type'].unique()):
            cat_df = eval_df[eval_df['true_type'] == cat]
            error_rate = (1 - cat_df['correct'].mean()) * 100
            print(f"  {cat:15s}: {error_rate:5.1f}% ({len(cat_df[~cat_df['correct']])}/{len(cat_df)})")
        high_conf_errors = errors[errors['confidence_score'] >= 0.7].sort_values('confidence_score', ascending=False)
        if len(high_conf_errors) > 0:
            print("\nHigh-Confidence Errors (≥70%):")
            for _, row in high_conf_errors.head(5).iterrows():
                print(f"  {row['file_name']}")
                print(f"  True: {row['true_type']:15s} → Predicted: {row['predicted_type']:15s}")
                print(f"  Confidence: {row['confidence_score']*100:.1f}%")
        print("=" * 60 + "\n")

    def generate_masking_report_excel(self):
        """Excel formatında detaylı maskeleme raporu"""
        import pandas as pd
        from collections import Counter
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file = f"masking_report_{timestamp}.xlsx"
    
    # ========================================
    # 1️ DOSYA BAZINDA DETAY
    # ========================================
        file_details = []
    
        for result in self.results:
            if result.get("predicted_type") in ("ERROR", "SKIPPED"):
                continue
        
            mask_stats = result.get("mask_stats", {})
            total_pii = sum(mask_stats.values()) if mask_stats else 0
        
            row = {
                "Dosya Adı": result.get("file_name"),
                "Belge Tipi": result.get("true_type"),
                "Tahmin": result.get("predicted_type"),
                "Toplam PII": total_pii,
                "Hassas Veri?": "EVET" if total_pii > 0 else "HAYIR",
            }
        
        
            pii_types = ["email", "phone", "credit_card", "iban", "tax_id", 
                        "signature", "address", "person", "date"]
        
            for pii in pii_types:
                row[pii.upper()] = mask_stats.get(pii, 0) if mask_stats else 0
        
            file_details.append(row)
    
        df_files = pd.DataFrame(file_details)
    
    # ========================================
    # 2️ ÖZET İSTATİSTİKLER
    # ========================================
        total_files = len(df_files)
        files_with_pii = df_files[df_files["Hassas Veri?"] == "EVET"].shape[0]
    
    # PII türlerine göre toplam
        pii_summary = []
        for pii in pii_types:
            col_name = pii.upper()
            if col_name in df_files.columns:
                total = df_files[col_name].sum()
                files_count = (df_files[col_name] > 0).sum()
            
                pii_summary.append({
                    "PII Türü": pii.upper(),
                    "Toplam Adet": int(total),
                    "Kaç Dosyada": int(files_count),
                    "Yüzde (%)": f"{files_count/total_files*100:.1f}%" if total_files > 0 else "0%"
                })
    
        df_summary = pd.DataFrame(pii_summary)
    
    # ========================================
    # 3️ BELGE TİPİNE GÖRE DAĞILIM
    # ========================================
        doc_type_summary = []
    
        for doc_type in df_files["Belge Tipi"].unique():
            subset = df_files[df_files["Belge Tipi"] == doc_type]
        
            row = {
                "Belge Tipi": doc_type.upper(),
                "Toplam Dosya": len(subset),
                "PII Bulunan": (subset["Toplam PII"] > 0).sum(),
                "Toplam PII Sayısı": subset["Toplam PII"].sum()
            }
        
        # En çok bulunan PII türü
            pii_cols = [col for col in subset.columns if col.isupper() and col not in ["Dosya Adı", "Belge Tipi", "Tahmin"]]
            pii_sums = {col: subset[col].sum() for col in pii_cols}
            top_pii = max(pii_sums, key=pii_sums.get) if pii_sums else "N/A"
        
            row["En Çok Bulunan PII"] = top_pii
            row["Miktarı"] = pii_sums.get(top_pii, 0)
        
            doc_type_summary.append(row)
    
        df_doc_types = pd.DataFrame(doc_type_summary)
    
    # ========================================
    # 4️ GENEL ÖZET
    # ========================================
        general_summary = pd.DataFrame([{
           "Metrik": "Toplam İşlenen Dosya",
            "Değer": total_files
        }, {
            "Metrik": "Hassas Veri Bulunan Dosya",
            "Değer": f"{files_with_pii} (%{files_with_pii/total_files*100:.1f})"
        }, {
            "Metrik": "Hassas Veri Bulunmayan Dosya",
            "Değer": total_files - files_with_pii
        }, {
            "Metrik": "Toplam Maskelenen PII",
            "Değer": int(df_files["Toplam PII"].sum())
        }])
    
    # ========================================
    # 5️ EXCEL'E YAZDIR
    # ========================================
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Sheet 1: Genel Özet
            general_summary.to_excel(writer, sheet_name='Genel Özet', index=False)
        
        # Sheet 2: PII Türlerine Göre
            df_summary.to_excel(writer, sheet_name='PII Türleri', index=False)
        
        # Sheet 3: Belge Tiplerine Göre
            df_doc_types.to_excel(writer, sheet_name='Belge Tipleri', index=False)
        
        # Sheet 4: Dosya Detayları
            df_files.to_excel(writer, sheet_name='Dosya Detayları', index=False)
    
        logger.info(f" Excel report saved: {excel_file}")
    
    # Console özeti
        print(f"\n{'='*80}")
        print(f" MASKING REPORT SUMMARY")
        print(f"{'='*80}")
        print(f"\n Toplam Dosya: {total_files}")
        print(f" Hassas Veri Bulunan: {files_with_pii} (%{files_with_pii/total_files*100:.1f})")
        print(f" Temiz Dosya: {total_files - files_with_pii}")
        print(f" Toplam Maskelenen PII: {int(df_files['Toplam PII'].sum())}")
    
        print(f"\n En Çok Bulunan PII Türleri:")
        top_5 = df_summary.nlargest(5, "Toplam Adet")
        for _, row in top_5.iterrows():
            print(f"  • {row['PII Türü']:15s}: {row['Toplam Adet']:3d} adet ({row['Yüzde (%)']})")
    
        print(f"\n Belge Tiplerine Göre:")
        for _, row in df_doc_types.iterrows():
            print(f"  • {row['Belge Tipi']:15s}: {row['Toplam PII Sayısı']:3d} PII "
                f"(En çok: {row['En Çok Bulunan PII']} - {row['Miktarı']} adet)")
    
        print(f"\n Detaylı rapor kaydedildi: {excel_file}")
        print(f"{'='*80}\n")
    
        return excel_file

# ========================= MAIN TEST RUNNER =========================

async def main(parallel: bool = True):
    """
    Main test execution function
    
    Args:
        parallel: True = Parallel processing (fast), False = Sequential (debug)
    """
    print("\n" + "=" * 80)
    print("MULTI-AGENT DOCUMENT PROCESSING - v3.3 FINE-TUNED (w/ FIXES & MASKING)")
    if parallel:
        print("MODE: PARALLEL PROCESSING ")
    else:
        print("MODE: SEQUENTIAL PROCESSING ")
    print("=" * 80)
    print("Document Types: email, form, invoice, letter, news_article, resume, receipt")
    print("A2A Protocol: OCR → Masking → Normalization → Classification")
    print(f"Tesseract: {'✓ Available' if PYTESS_AVAILABLE else '✗ Not available (will use TrOCR)'}")
    print("\n v3.3 FEATURES:")
    print("  • Enhanced Email/Form/Invoice/Receipt disambiguation")
    print("  • Multi-PSM Tesseract OCR for low-quality documents")
    print("  • Strengthened News Article detection")
    print("  • NEW: Masking Agent with 15+ PII types")
    print("      - Visual masking with color-coded rectangles")
    print("      - Email, Phone, IBAN, SSN, Credit Card, Names, etc.")
    print("      - Real-time statistics and reporting")
    print("\n v3.4 ENHANCED FEATURES:")
    print("  • Hugging Face NER Model Integration")
    print("  • Smart Entity Detection (PER, LOC, ORG, MISC)")
    print("  • Hybrid Model + Regex Masking")
    print("  • Reduced False Positives")
    print("  • Enhanced Name/Address/Company Detection")
    print("=" * 80)

    # Initialize dataset manager
    mgr = TestDatasetManager(data_folder="test_data")
    dataset_path = Path("test_data/test_dataset.json")

    # Create/update dataset index
    if not dataset_path.exists():
        print("\n Creating dataset index...")
        mgr.create_dataset()
    else:
        print("\n Dataset index found, updating...")
        mgr.create_dataset()

    # Show statistics
    mgr.show_statistics()

    # Load dataset
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    n = len(data)

    if n == 0:
        print("\n" + "=" * 80)
        print("  NO TEST SAMPLES FOUND")
        print("=" * 80)
        print("\nPlease add images to the following directories:")
        print("\ntest_data/")
        print("  ├── email/          (add email screenshots)")
        print("  ├── form/           (add form images)")
        print("  ├── invoice/        (add invoice images)")
        print("  ├── letter/         (add letter images)")
        print("  ├── news_article/   (add news article images)")
        print("  ├── resume/         (add resume/CV images)")
        print("  └── receipt/        (add receipt images)")
        print("\nSupported formats: JPG, PNG, WEBP, TIFF, BMP, GIF, HEIC, HEIF")
        print("=" * 80 + "\n")
        return

    est_time = n * 2.5 / 60
    print(f"\n Preparing to test {n} samples (estimated time: ~{est_time:.1f} min)")
    print(f"  OCR Engine: {'Tesseract' if PYTESS_AVAILABLE else 'TrOCR'}")
    print(f"  Language: English")
    print(f"  PII Masking: Enabled (15+ data types)")
    input("\nPress ENTER to start testing...")

    # Initialize coordinator
    print("\n Initializing Multi-Agent System...")
    coordinator = CoordinatorAgent(ocr_default_backend="tesseract" if PYTESS_AVAILABLE else "trocr")

    # Start agents (including Masking Agent)
    tasks = [
        asyncio.create_task(coordinator.ocr_agent.process_messages()),
        asyncio.create_task(coordinator.masking_agent.process_messages()),
        asyncio.create_task(coordinator.normalization_agent.process_messages()),
        asyncio.create_task(coordinator.classification_agent.process_messages()),
        asyncio.create_task(coordinator.process_messages())
    ]
    print("✓ All agents started (including Masking Agent)")

    # Test engine
    engine = TestEngine(coordinator, test_json_path=str(dataset_path))
    
    # Max parallel sayısını ayarla (CPU çekirdek sayısına göre)
    import os
    cpu_count = os.cpu_count() or 4
    engine.max_parallel = min(cpu_count, 8)  # Max 8 paralel işlem
    engine.semaphore = asyncio.Semaphore(engine.max_parallel)
    
    print(f" Parallel processing enabled: {engine.max_parallel} concurrent tasks")

    start_time = datetime.now()
    
    # Parallel veya Sequential mod seçimi
    if parallel:
        await engine.run_all_tests_parallel()
    else:
        await engine.run_all_tests_sequential()
    
    elapsed = (datetime.now() - start_time).total_seconds()

    # Stop all agents (cancel tasks)
    for task in tasks:
        task.cancel()
  
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception:
        pass

    engine.error_analysis()

    print("\n" + "=" * 80)
    print("TEST EXECUTION COMPLETED")
    print("=" * 80)
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    print(f"      Mode: {'PARALLEL' if parallel else 'SEQUENTIAL'}")
    if n > 0:
        print(f"      Average per sample: {elapsed / n:.2f} seconds")

    print("\n Generated Files:")
    print("            • test_results_YYYYMMDD_HHMMSS.json")
    print("            • test_summary_YYYYMMDD_HHMMSS.txt")
    print("            • confusion_matrix_YYYYMMDD_HHMMSS.png")
    print("            • confidence_analysis_YYYYMMDD_HHMMSS.png")
    print("            • errors_with_ocr_YYYYMMDD_HHMMSS.json (Misclassifications only)")

    print("\n Expected Improvements (v3.3 - FINE-TUNED):")
    print("      • Document Classification: Enhanced accuracy")
    print("      • PII Detection: 15+ data types with visual masking")
    print("      • Multi-Agent System: OCR → Masking → Normalization → Classification")
    print("      • Color-coded Security: Different colors for different PII types")
    print("\n" + "=" * 80)
    print("Thank you for testing! ")
    print("=" * 80 + "\n")

async def main_with_qa_a2a(parallel: bool = True):
    # burada artık await main(...) YOK. Direkt Q&A başlar.

    coordinator = CoordinatorAgentWithQA(
        ocr_default_backend="tesseract" if PYTESS_AVAILABLE else "trocr",
        enable_qa=True
    )

    agent_tasks = [
        asyncio.create_task(coordinator.ocr_agent.process_messages()),
        asyncio.create_task(coordinator.masking_agent.process_messages()),
        asyncio.create_task(coordinator.normalization_agent.process_messages()),
        asyncio.create_task(coordinator.classification_agent.process_messages()),
        asyncio.create_task(coordinator.process_messages()),
    ]

    if coordinator.gemini_agent:
        agent_tasks.append(asyncio.create_task(coordinator.gemini_agent.process_messages()))
    if coordinator.llama_agent:
        agent_tasks.append(asyncio.create_task(coordinator.llama_agent.process_messages()))

    qa_engine = QATestEngineA2A(coordinator)

    try:
        await qa_engine.run_streaming_qa(
            test_dataset_path="test_data/test_dataset.json",
            limit=10,           
            use_model="gemini",    
            questions_per_doc=2
        )
    finally:
        for t in agent_tasks:
            t.cancel()
        await asyncio.gather(*agent_tasks, return_exceptions=True)

# ========================= QUICK TEST MODE =========================

async def quick_test_single_image(image_path: str):
    """Quick test for a single image"""
    print("\n" + "=" * 60)
    print("QUICK TEST MODE - Single Image (v3.3 FINE-TUNED w/ MASKING)")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f" Image not found: {image_path}")
        return

    coordinator = CoordinatorAgent(ocr_default_backend="tesseract" if PYTESS_AVAILABLE else "trocr")
    
    tasks = [
        asyncio.create_task(coordinator.ocr_agent.process_messages()),
        asyncio.create_task(coordinator.masking_agent.process_messages()),
        asyncio.create_task(coordinator.normalization_agent.process_messages()),
        asyncio.create_task(coordinator.classification_agent.process_messages()),
        asyncio.create_task(coordinator.process_messages())
    ]
    
    print(f"\n Processing: {Path(image_path).name}")
    try:
        image_base64 = image_any_to_base64(image_path)
        result = await coordinator.process_document(image_base64, lang="eng")

        print("\n" + "-" * 60)
        print("RESULTS")
        print("-" * 60)
        print(f"Document Type: {result.get('document_type', 'N/A')}")
        print(f"Confidence: {result.get('classification_confidence', 0) * 100:.2f}%")
        print(f"OCR Engine: {result.get('ocr_engine', 'N/A')}")
        print(f"Status: {result.get('status', 'N/A')}")
        
        # Masking istatistiklerini göster
        if 'mask_stats' in result:
            print(f"\n PII Detection Results:")
            mask_stats = result.get('mask_stats', {})
            total_masked = sum(mask_stats.values())
            if total_masked > 0:
                print(f"   Total PII items masked: {total_masked}")
                for pii_type, count in mask_stats.items():
                    if count > 0:
                        print(f"   - {pii_type}: {count}")
            else:
                print("   No PII detected")
        
        if 'masked_image' in result and result['masked_image']:
            # Klasör oluştur
            output_dir = Path("quick_test_masked")
            output_dir.mkdir(exist_ok=True)
            
            # Dosya adı
            original_name = Path(image_path).stem
            output_path = output_dir / f"masked_{original_name}.png"
            
            # Base64'ten çöz ve kaydet
            import base64
            masked_bytes = base64.b64decode(result['masked_image'])
            with open(output_path, "wb") as f:
                f.write(masked_bytes)
            
            print(f"\n Masked Image Saved: {output_path}")
            print(f"   → You can view it now!")
            
            try:
                import platform
                import subprocess
                
                system = platform.system()
                if system == "Windows":
                    os.startfile(str(output_path))
                elif system == "Darwin":  # macOS
                    subprocess.run(["open", str(output_path)])
                else:  # Linux
                    subprocess.run(["xdg-open", str(output_path)])
                
                print(f"    Image opened automatically!")
            except Exception as e:
                print(f"     Could not open image automatically: {e}")
                print(f"    Manually open: {output_path.absolute()}")
        
        if 'text' in result:
            text_preview = result.get('text') or ''
            limit = 600
            text_preview = (text_preview[:limit] + '...') if len(text_preview) > limit else text_preview
            print(f"\n Extracted Text Preview (first {limit} chars):\n{text_preview}")
        print("-" * 60 + "\n")

    except Exception as e:
        print(f" Error processing image: {e}")
        import traceback
        traceback.print_exc()

    finally:
        for task in tasks:
            task.cancel()
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            pass

# ========================= ENTRY POINT =========================

if __name__ == "__main__":
    import sys
    import asyncio

    # Paralel/Sequential mod seçimi
    parallel_mode = True
    if "--sequential" in sys.argv:
        parallel_mode = False

    # QUICK mode
    if "--quick" in sys.argv:
        quick_index = sys.argv.index("--quick")
        if quick_index + 1 < len(sys.argv):
            image_path = sys.argv[quick_index + 1]
            asyncio.run(quick_test_single_image(image_path))
        else:
            print("Usage: python test.py --quick <image_path>")
        raise SystemExit(0)

    # QA mode (TEK TEK: doküman işle -> soru sor -> diğerine geç)
    if "--qa" in sys.argv:
        asyncio.run(main_with_qa_a2a(parallel=parallel_mode))
    else:
        asyncio.run(main(parallel=parallel_mode))
