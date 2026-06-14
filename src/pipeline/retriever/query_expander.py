class QueryExpander:
    def __init__(self):
        self.synonyms = {
            "người đại diện": ["đại diện theo pháp luật", "đại diện theo ủy quyền"],
            "nhãn hiệu": ["sở hữu công nghiệp", "xâm phạm nhãn hiệu", "quyền đối với nhãn hiệu"],
            "sáng chế": ["sở hữu công nghiệp", "quyền đối với sáng chế"],
            "kiểu dáng công nghiệp": ["sở hữu công nghiệp", "quyền đối với kiểu dáng công nghiệp"],
            "văn bằng bảo hộ": ["sở hữu công nghiệp", "đăng ký sở hữu công nghiệp"],
            "doanh nghiệp nhỏ và vừa": ["sme", "hỗ trợ doanh nghiệp"],
            "bảo hiểm xã hội": ["bhxh", "bảo hiểm thất nghiệp"],
            "hóa đơn": ["thuế", "chứng từ", "hóa đơn điện tử"],
            "công đoàn": ["kinh phí công đoàn", "lao động"],
            "đấu thầu": ["ưu đãi trong đấu thầu", "nhà thầu"],
            "thương mại điện tử": ["sàn giao dịch thương mại điện tử", "nền tảng số"],
            "tnhh": ["trách nhiệm hữu hạn"],
            "giấy phép": ["giấy chứng nhận"],
        }

    def expand(self, query: str) -> str:
        query_lower = query.lower()
        expanded_terms = []

        for key, values in self.synonyms.items():
            if key in query_lower:
                for value in values:
                    if value not in query_lower and value not in expanded_terms:
                        expanded_terms.append(value)

        if expanded_terms:
            print(f"[QueryExpander] Bổ sung từ đồng nghĩa: {expanded_terms}", flush=True)
            return query + " " + " ".join(expanded_terms)

        return query

    def _legal_anchor_expansion(self, query: str) -> list[str]:
        query_lower = query.lower()
        additions = []

        if any(kw in query_lower for kw in ["doanh nghiệp nhỏ và vừa", "sme", "khởi nghiệp", "ươm tạo"]):
            additions.extend([
                "Nghị định 80/2021/NĐ-CP",
                "Thông tư 52/2023/TT-BTC",
                "Luật hỗ trợ doanh nghiệp nhỏ và vừa",
            ])
        elif any(kw in query_lower for kw in ["thuế", "kê khai", "nộp phạt", "hóa đơn", "chứng từ", "khấu trừ"]):
            additions.extend([
                "Luật Quản lý thuế",
                "Nghị định 126/2020/NĐ-CP",
                "Nghị định 125/2020/NĐ-CP",
                "Nghị định 123/2020/NĐ-CP",
                "Thông tư 80/2021/TT-BTC",
                "Thông tư 105/2020/TT-BTC",
                "Thông tư 78/2021/TT-BTC",
            ])
        elif any(kw in query_lower for kw in ["đăng ký doanh nghiệp", "thành lập", "người đại diện", "cổ đông", "hội đồng quản trị"]):
            additions.extend([
                "Luật doanh nghiệp 59/2020/QH14",
                "Nghị định 01/2021/NĐ-CP",
            ])
        elif any(kw in query_lower for kw in ["lao động", "hợp đồng lao động", "sa thải", "bằng cấp", "giữ bản chính", "kỷ luật"]):
            additions.extend([
                "Bộ luật lao động 45/2019/QH14",
                "Nghị định 12/2022/NĐ-CP",
                "Nghị định 145/2020/NĐ-CP",
            ])
        elif any(kw in query_lower for kw in ["sở hữu công nghiệp", "nhãn hiệu", "sáng chế", "kiểu dáng công nghiệp", "văn bằng bảo hộ"]):
            additions.extend([
                "Luật Sở hữu trí tuệ",
                "Nghị định 65/2023/NĐ-CP",
                "Nghị định 99/2013/NĐ-CP",
            ])
        elif any(kw in query_lower for kw in ["thương mại điện tử", "sàn giao dịch", "nền tảng", "hộ kinh doanh"]):
            additions.extend([
                "Nghị định 52/2013/NĐ-CP",
                "Nghị định 85/2021/NĐ-CP",
                "Luật Quản lý thuế",
            ])
        elif any(kw in query_lower for kw in ["đấu thầu", "nhà thầu", "gói thầu"]):
            additions.extend([
                "Luật Đấu thầu",
                "Nghị định 24/2024/NĐ-CP",
                "Luật hỗ trợ doanh nghiệp nhỏ và vừa",
            ])

        return additions

    def get_query_variations(self, query: str) -> list[str]:
        variations = [query]
        additions = []

        expanded = self.expand(query)
        if expanded != query:
            suffix = expanded[len(query):].strip()
            if suffix:
                additions.append(suffix)

        additions.extend(self._legal_anchor_expansion(query))

        deduped = []
        seen = set()
        query_lower = query.lower()
        for item in additions:
            normalized = item.strip()
            if not normalized:
                continue
            if normalized.lower() in seen:
                continue
            if normalized.lower() in query_lower:
                continue
            seen.add(normalized.lower())
            deduped.append(normalized)

        if deduped:
            variation = query + " " + " ".join(deduped)
            print(f"[QueryExpander] Bổ sung truy vấn phụ: {deduped}", flush=True)
            variations.append(variation)

        return variations
