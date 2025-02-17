import json
import os

import google.generativeai as genai
from TurkishStemmer import TurkishStemmer
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.settings import settings  # settings nesnesini alıyoruz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from zemberek import TurkishMorphology

# Google Gemini API anahtarı ortam değişkeni olarak ayarlanabilir
GOOGLE_API_KEY = "XXXXXXXXXXXXXXXXX9-XX9XX_XXXX_XX99" # Buraya Google api keyinizi giriniz.

if not GOOGLE_API_KEY:
    raise ValueError("Google API key bulunamadı. Lütfen geçerli bir API anahtarı girin.")

# Marker yapılandırma ayarlarına Google Gemini API anahtarını ekleyelim
os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

# Google Gemini yapılandırmasını güncelle
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    raise ValueError(f"Google Gemini API yapılandırma hatası: {e}")


def use_gemini_for_layout_analysis(text):
    """Google Gemini modelini kullanarak daha iyi layout analizi yapar."""
    prompt = f"""
    Aşağıdaki akademik makale metnindeki bölümleri belirle:
    - İçindekiler
    - Başlık
    - Altbilgi
    - Kaynakça
    - Şekiller ve Tablolar
    - Chunking için uygun ayrım noktaları

    **Yalnızca geçerli JSON döndür.** Ek metin, açıklama veya kod bloğu işaretleri ekleme.
    JSON dışında herhangi bir metin olmamalı.

    Makale Metni (ilk 5000 karakter):
    {text[:5000]}
    """  # İlk 5000 karakteri analiz edelim.

    # models/gemini-2.0-flash yerine, sizde mevcut olan herhangi bir modeli kullanabilirsiniz
    model = genai.GenerativeModel("models/gemini-2.0-flash")

    response = model.generate_content([{"role": "user", "parts": [prompt]}])
    response_text = response.text.strip()

    if not response_text:
        # Boş geldi, fallback yapabilir ya da direkt boş {}
        print("LLM'den boş yanıt döndü, JSON parse edilemiyor.")
        return {}

    try:
        parsed_json = json.loads(response_text)
    except json.JSONDecodeError:
        print("LLM çıktısı geçerli JSON formatında değil. Çıktı:")
        print(response_text)
        return {}

    return parsed_json


# PDF'den metin çıkaran fonksiyon (Marker kullanarak ve Gemini ile geliştirilmiş layout analizi yaparak)
def pdf_to_text(pdf_path):
    # Yeni config ayarı oluşturuyoruz
    config_dict = settings.model_dump()

    # Gereksiz bölümleri kaldırmak için özel seçenekler ekleyelim
    config_dict["exclude_page_elements"] = ["header", "footer", "table_of_contents", "references", "figures"]

    # Google Gemini API kullanımı için ek konfigürasyon
    if GOOGLE_API_KEY:
        config_dict["use_llm"] = True
        config_dict["gemini_api_key"] = GOOGLE_API_KEY  # Bu satırı ekleyin

    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config=config_dict  # Güncellenmiş config'i kullanıyoruz
    )

    rendered = converter(pdf_path)
    text, layout_info, _ = text_from_rendered(rendered)

    # Gemini ile Layout Analizi
    enhanced_layout = use_gemini_for_layout_analysis(text)

    # Layout bilgisini JSON olarak kaydet
    with open("layout_info.json", "w", encoding="utf-8") as f:
        json.dump({"marker": layout_info, "gemini": enhanced_layout}, f, ensure_ascii=False, indent=4)

    return text


# Metni tokenize eden ve stopword'leri temizleyen fonksiyon
def clean_text(text):
    words = word_tokenize(text, language='turkish')
    stop_words = set(stopwords.words('turkish'))
    clean_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return clean_words


# Kelimelerin köklerini bulan fonksiyon
def stem_words(words):
    stemmer = TurkishStemmer()
    return [stemmer.stem(word) for word in words]


# Zemberek kullanarak kelime analizini gerçekleştiren fonksiyon
def analyze_with_zemberek(words):
    morphology = TurkishMorphology.create_with_defaults()
    results = {}
    for word in words:
        analysis = morphology.analyze(word)
        results[word] = [str(ana) for ana in analysis]
    return results


# Metni vektör haline getiren fonksiyon
def vectorize_text(words):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([" ".join(words)])
    return X.toarray()


# Cümleleri vektör haline getiren fonksiyon
def embed_sentences(sentences):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device="cuda")
    embeddings = model.encode(sentences)
    return embeddings


# Ana fonksiyon (Tüm işlemleri çalıştırır ve süreci denetlemek için JSON çıktısı üretir)
def process_pdf(pdf_path):
    print("PDF'den metin çıkarılıyor...")
    text = pdf_to_text(pdf_path)

    print("Metin temizleniyor...")
    clean_words = clean_text(text)

    print("Kelime kökleri bulunuyor...")
    stemmed_words = stem_words(clean_words)

    print("Zemberek ile kelime analizi yapılıyor...")
    zemberek_analysis = analyze_with_zemberek(stemmed_words)

    print("Metin vektörleştiriliyor...")
    vectorized_text = vectorize_text(stemmed_words)

    print("Cümleler gömülüyor...")
    sentence_embeddings = embed_sentences(text.split('\n'))

    # Çıkarılan metinleri JSON formatında kaydetme
    process_output = {
        "clean_words": clean_words,
        "stemmed_words": stemmed_words,
        "zemberek_analysis": zemberek_analysis,
        "vectorized_text": vectorized_text.tolist(),
        "sentence_embeddings": sentence_embeddings.tolist()
    }

    with open("processed_text.json", "w", encoding="utf-8") as f:
        json.dump(process_output, f, ensure_ascii=False, indent=4)

    return process_output


# Örnek çalışma
if __name__ == "__main__":
    pdf_path = "pdfs/Turkiye’dekiTurizmRehberligiKonuluLisansustuTezcalismalarininBibliyometrikProfili(1998-2018).pdf"  # Buraya PDF dosya yolunuzu girin
    result = process_pdf(pdf_path)
    print("İşlem tamamlandı! JSON dosyaları oluşturuldu.")
