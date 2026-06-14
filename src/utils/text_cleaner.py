import re
from pyvi import ViTokenizer

# Stopwords tinh gọn: Chỉ loại bỏ các từ vô nghĩa, giữ lại các từ pháp lý quan trọng (như trách_nhiệm, xử_phạt, đăng_ký)
VIETNAMESE_STOPWORDS = {
    "khi", "nào", "bao_lâu", "cần", "được", "bị", "việc", "có", "là", "gì", "cho", "về",
    "và", "hoặc", "nhưng", "thì", "của", "các", "này", "nếu", 
    "để", "với", "trong", "tại", "theo", "như", "ở", "bởi", 
    "do", "từ", "đến", "sẽ", "đã", "đang", "cũng", "đều", "chỉ", 
    "tự", "lại", "qua", "những", "sự", "cái", "như_thế_nào", 
    "thế_nào", "ra_sao", "làm_sao", "cơ", "bản", "nơi", "ai"
}

class TextCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans input text by removing extra whitespaces, newlines, etc.
        """
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def segment_words(text: str, engine: str = "pyvi") -> str:
        """
        Applies Vietnamese word segmentation using pyvi or underthesea.
        e.g., "Nghị định này quy định" -> "Nghị_định này quy_định"
        """
        if not text:
            return ""
        cleaned = TextCleaner.clean_text(text)
        
        if engine.lower() == "pyvi":
            try:
                return ViTokenizer.tokenize(cleaned)
            except Exception:
                # Fallback to underthesea
                pass
                
        # Fallback / explicit underthesea
        try:
            from underthesea import word_tokenize
            return word_tokenize(cleaned, format="text")
        except Exception:
            return cleaned

    @staticmethod
    def tokenize(text: str, lowercase: bool = True, segmented: bool = True, engine: str = "pyvi", remove_stopwords: bool = True) -> list[str]:
        """
        Tokenizes text into a list of words.
        If segmented is True, applies word segmentation.
        If remove_stopwords is True, filters out common grammatical Vietnamese stopwords.
        """
        if not text:
            return []
            
        if segmented:
            text = TextCleaner.segment_words(text, engine=engine)
        else:
            text = TextCleaner.clean_text(text)
            
        if lowercase:
            text = text.lower()
            
        # Remove common punctuation (preserve underscores — they are PyVi word boundaries like nhãn_hiệu)
        text = re.sub(r"[.,;:?!\-\"\'\(\)\[\]\{\}\<\>\/\\\|@#\$%\^\&\*\+\=\~\`]", " ", text)
        tokens = text.split()
        
        if remove_stopwords:
            tokens = [tok for tok in tokens if tok not in VIETNAMESE_STOPWORDS]
            
        return tokens