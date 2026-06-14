import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class TaxonomyRouter:
    def __init__(self):
        use_llm_router = os.getenv("USE_LLM_TAXONOMY", "false").lower() == "true"
        self.api_key = (os.getenv("OPENROUTER_API_KEY") or os.getenv("TOGETHER_API_KEY")) if use_llm_router else None
        self.base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else "https://api.together.xyz/v1"
        self.model = "qwen/qwen-2.5-7b-instruct" if os.getenv("OPENROUTER_API_KEY") else "Qwen/Qwen2.5-72B-Instruct-Turbo"
        
        self.client = None
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                print(f"[TaxonomyRouter] Initialized Qwen-2.5 via API ({self.base_url})", flush=True)
            except Exception as e:
                print(f"[TaxonomyRouter] Failed to init API client: {e}. Falling back to Rule-based.", flush=True)

        self.domain_aliases = {
            "thuế": ["Thuế, phí, lệ phí, các khoản thu khác", "Tài chính", "Kế toán, kiểm toán"],
            "phí, lệ phí": ["Thuế, phí, lệ phí, các khoản thu khác", "Tài chính"],
            "kế toán": ["Kế toán, kiểm toán", "Tài chính"],
            "kiểm toán": ["Kế toán, kiểm toán"],
            "tài chính": ["Tài chính", "Kế toán, kiểm toán"],
            "lao động": ["Lao động", "Bảo hiểm", "An toàn, vệ sinh lao động"],
            "bảo hiểm": ["Bảo hiểm", "Lao động"],
            "bảo hiểm xã hội": ["Bảo hiểm", "Lao động"],
            "an toàn, vệ sinh lao động": ["An toàn, vệ sinh lao động", "Lao động"],
            "doanh nghiệp": ["Doanh nghiệp, hợp tác xã", "Thương mại, đầu tư, chứng khoán"],
            "đầu tư": ["Thương mại, đầu tư, chứng khoán", "Doanh nghiệp, hợp tác xã"],
            "đấu thầu": ["Thương mại, đầu tư, chứng khoán", "Doanh nghiệp, hợp tác xã"],
            "thương mại": ["Thương mại, đầu tư, chứng khoán", "Doanh nghiệp, hợp tác xã"],
            "thương mại điện tử": ["Thương mại, đầu tư, chứng khoán", "Thuế, phí, lệ phí, các khoản thu khác"],
            "sở hữu trí tuệ": ["Khoa học, công nghệ", "Thương mại, đầu tư, chứng khoán"],
            "dân sự": ["Dân sự"],
            "hình sự": ["Hình sự"],
            "khác": [],
        }

    def _normalize_domains(self, domains: list[str] | None) -> list[str] | None:
        if not domains:
            return None

        normalized = []
        seen = set()
        for domain in domains:
            key = (domain or "").strip().lower()
            mapped = self.domain_aliases.get(key, [domain])
            for item in mapped:
                if item and item not in seen:
                    seen.add(item)
                    normalized.append(item)
        return normalized or None

    def route(self, query: str) -> list[str]:
        """
        Uses Qwen-2.5 (via API) to classify the query into Legal Domains.
        Falls back to rule-based classification if API is unavailable or fails.
        """
        if self.client:
            try:
                prompt = """Bạn là chuyên gia phân loại pháp lý Việt Nam. 
Nhiệm vụ: Phân loại câu hỏi sau vào MỘT HOẶC NHIỀU lĩnh vực (domain) chính xác nhất để tối ưu hóa công cụ tìm kiếm.
Chỉ trả về các lĩnh vực nằm trong danh sách sau: ["Thuế", "Lao động", "Bảo hiểm xã hội", "Doanh nghiệp", "Đầu tư", "Sở hữu trí tuệ", "Kế toán", "Dân sự", "Hình sự", "An toàn, vệ sinh lao động", "Khác"].

Hãy trả về DUY NHẤT một chuỗi JSON hợp lệ với cấu trúc:
{
  "domains": ["Tên lĩnh vực 1", "Tên lĩnh vực 2"]
}"""
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Câu hỏi: {query}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    timeout=10 # Fail fast so RAG doesn't hang
                )
                
                result = json.loads(response.choices[0].message.content)
                domains = result.get("domains", [])
                if isinstance(domains, list) and len(domains) > 0:
                    return self._normalize_domains([d for d in domains if d != "Khác"])
            except Exception as e:
                print(f"[TaxonomyRouter] API Call failed ({e}). Falling back to Rule-based routing.", flush=True)
        
        # --- Fallback: Rule-based routing ---
        return self._rule_based_route(query)
        
    def _rule_based_route(self, query: str) -> list[str]:
        query_lower = query.lower()
        domains = []
        
        if any(kw in query_lower for kw in ["trợ cấp tuất", "hưu trí", "tai nạn lao động", "bệnh nghề nghiệp", "thai sản", "bảo hiểm", "bhxh", "bhyt"]):
            domains.extend(["bảo hiểm", "lao động", "an toàn, vệ sinh lao động"])
        elif any(kw in query_lower for kw in ["sa thải", "kỷ luật", "làm thêm giờ", "tiền lương", "hợp đồng lao động", "thử việc"]):
            domains.extend(["lao động"])
        elif any(kw in query_lower for kw in ["sở hữu công nghiệp", "sở hữu trí tuệ", "nhãn hiệu", "bản quyền", "sáng chế", "kiểu dáng công nghiệp", "văn bằng bảo hộ"]):
            domains.extend(["sở hữu trí tuệ"])
        elif any(kw in query_lower for kw in ["thuế", "hóa đơn", "kê khai", "mã số thuế", "đóng thuế", "lệ phí trước bạ", "hoàn thuế"]):
            domains.extend(["thuế", "phí, lệ phí", "kế toán"])
        elif any(kw in query_lower for kw in ["kế toán", "hạch toán", "báo cáo tài chính", "kiểm toán"]):
            domains.extend(["tài chính", "kế toán", "kiểm toán"])
        elif any(kw in query_lower for kw in ["doanh nghiệp nhỏ và vừa", "sme", "ươm tạo", "khởi nghiệp"]):
            domains.extend(["doanh nghiệp"])
            if any(kw in query_lower for kw in ["đấu thầu", "nhà thầu", "gói thầu"]):
                domains.extend(["đấu thầu", "đầu tư"])
        elif any(kw in query_lower for kw in ["thương mại điện tử", "sàn giao dịch", "nền tảng", "hộ kinh doanh"]):
            domains.extend(["thương mại", "thuế", "doanh nghiệp"])
        elif any(kw in query_lower for kw in ["đấu thầu", "nhà thầu", "gói thầu"]):
            domains.extend(["đấu thầu", "đầu tư", "doanh nghiệp"])
        elif any(kw in query_lower for kw in ["thành lập", "đăng ký doanh nghiệp", "giải thể", "phá sản", "hội đồng quản trị", "cổ đông", "vốn điều lệ"]):
            domains.extend(["doanh nghiệp", "đầu tư"])
        
        return self._normalize_domains(list(set(domains)) if domains else None)
