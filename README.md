# 📄 Document Intelligence System

Çok ajanlı (multi-agent) mimari kullanarak doküman sınıflandırma, OCR ve PII (kişisel veri) maskeleme işlemleri gerçekleştiren bir yapay zeka sistemi.

---

## 🚀 Özellikler

- **Doküman Sınıflandırma** — Fatura, e-posta, makbuz ve daha fazlasını otomatik olarak tanır
- **OCR (Optik Karakter Tanıma)** — Tesseract ve TrOCR destekli çift motorlu metin çıkarımı
- **PII Maskeleme** — 15+ kişisel veri tipi (isim, TC kimlik, e-posta, telefon vb.) tespiti ve görsel maskeleme
- **Çok Ajanlı Mimari** — OCR → Masking → Normalization → Classification pipeline'ı
- **Paralel İşleme** — CPU çekirdeği sayısına göre eş zamanlı belge işleme
- **QA Modu** — Belgeler üzerine doğal dil soruları sorabilme (Gemini / LLaMA destekli)
- **HEIC/HEIF Desteği** — iPhone fotoğrafları dahil geniş format yelpazesi

---

## 🏗️ Mimari

```
Giriş Görseli
     │
     ▼
┌─────────────┐
│  OCR Agent  │  ← Tesseract veya TrOCR
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ Masking Agent│  ← PII tespiti ve görsel maskeleme
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Normalization Agent  │  ← Metin temizleme ve düzenleme
└──────────┬───────────┘
           │
           ▼
┌────────────────────────┐
│ Classification Agent   │  ← Doküman tipi tahmini
└────────────────────────┘
```

---

## 📦 Kurulum

### Gereksinimler

- Python 3.9+
- (İsteğe bağlı) Tesseract OCR

```bash
# Repoyu klonla
git clone https://github.com/KULLANICI_ADIN/REPO_ADIN.git
cd REPO_ADIN

# Sanal ortam oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### Tesseract Kurulumu (İsteğe Bağlı)

| İşletim Sistemi | Komut |
|---|---|
| Ubuntu/Debian | `sudo apt install tesseract-ocr` |
| macOS | `brew install tesseract` |
| Windows | [İndir](https://github.com/UB-Mannheim/tesseract/wiki) |

---

## ⚙️ Ortam Değişkenleri

Proje kök dizininde `.env` dosyası oluştur:

```env
GEMINI_API_KEY=your_gemini_api_key_here
HUGGINGFACE_TOKEN=your_hf_token_here
```

> ⚠️ `.env` dosyasını asla GitHub'a göndermeyiniz!

---

## 🖥️ Kullanım

### Standart Mod (Tüm Test Seti)

```bash
# Paralel mod (varsayılan)
python test.py

# Sıralı mod
python test.py --sequential
```

### Tek Görsel Testi

```bash
python test.py --quick /path/to/image.jpg
```

### QA (Soru-Cevap) Modu

```bash
python test.py --qa
```

---

## 📁 Desteklenen Formatlar

`.jpg` `.jpeg` `.png` `.webp` `.tif` `.tiff` `.bmp` `.gif` `.heic` `.heif`

---

## 📊 Çıktılar

Çalıştırma sonunda aşağıdaki dosyalar oluşturulur:

| Dosya | Açıklama |
|---|---|
| `test_results_YYYYMMDD_HHMMSS.json` | Ham test sonuçları |
| `test_summary_YYYYMMDD_HHMMSS.txt` | Özet rapor |
| `confusion_matrix_YYYYMMDD_HHMMSS.png` | Sınıflandırma matrisi |
| `confidence_analysis_YYYYMMDD_HHMMSS.png` | Güven skoru analizi |
| `errors_with_ocr_YYYYMMDD_HHMMSS.json` | Yanlış sınıflandırmalar |

---

## 🛠️ Kullanılan Teknolojiler

- **[TrOCR](https://huggingface.co/microsoft/trocr-base-printed)** — Microsoft'un transformer tabanlı OCR modeli
- **[Tesseract](https://github.com/tesseract-ocr/tesseract)** — Açık kaynak OCR motoru
- **[Transformers](https://huggingface.co/transformers/)** — Hugging Face model kütüphanesi
- **[OpenCV](https://opencv.org/)** — Görüntü işleme
- **[Pillow](https://pillow.readthedocs.io/)** — Görsel manipülasyon
- **[scikit-learn](https://scikit-learn.org/)** — Metrik hesaplama
- **[Pandas](https://pandas.pydata.org/) / [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/)** — Veri analizi ve görselleştirme

---

## 📝 Lisans

Bu proje staj kapsamında geliştirilmiştir.