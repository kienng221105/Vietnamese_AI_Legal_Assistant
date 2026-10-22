"""Create a question-only test set from public Ministry of Justice Q&A records."""

from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "external" / "pbgdpl-vn-legal-qna" / "data" / "qa.jsonl"
OUTPUT_PATH = ROOT / "data" / "test" / "project_scope_questions.json"
REPORT_PATH = ROOT / "reports" / "test_question_collection_report.md"
AUDIT_PATH = ROOT / "data" / "test" / "project_scope_topic_audit.json"
SHUFFLE_SEED = 20260922

TOPIC_SELECTORS = [
    ("Dân sự", 260, lambda row: row["lv_name"] in {"Dân sự", "Bồi thường nhà nước"}),
    ("Hôn nhân, gia đình, trẻ em, bình đẳng giới", 150, lambda row: row["lv_name"] in {"Hôn nhân gia đình", "Con nuôi"}),
    ("Xây dựng, nhà ở, đô thị", 25, lambda row: row["lv_name"] == "Xây dựng, nhà ở, đô thị"),
    ("Bổ trợ tư pháp", 150, lambda row: row["lv_name"] in {"Công chứng", "Chứng thực", "Quản lý luật sư", "Giám định tư pháp", "Đấu giá tài sản"}),
    ("Hành chính tư pháp", 150, lambda row: row["lv_name"] in {"Hành chính tư pháp", "Hộ tịch", "Lý lịch tư pháp", "Quốc tịch", "Cư trú"}),
    ("Thi hành án", 150, lambda row: row["lv_name"] == "Thi hành án"),
    ("Tố tụng và các phương thức giải quyết tranh chấp", 115, lambda row: bool(re.search(r"trọng tài|tòa án|toà án|khởi kiện|hòa giải|hoà giải|tranh chấp|tố tụng", row["question_text"], re.I))),
]
ALLOWED_SOURCE_TOPICS = {
    None,
    "Dân sự",
    "Giao dịch đảm bảo",
    "Bồi thường nhà nước",
    "Hôn nhân gia đình",
    "Con nuôi",
    "Xây dựng, nhà ở, đô thị",
    "Công chứng",
    "Chứng thực",
    "Quản lý luật sư",
    "Giám định tư pháp",
    "Đấu giá tài sản",
    "Hành chính tư pháp",
    "Hộ tịch",
    "Lý lịch tư pháp",
    "Quốc tịch",
    "Cư trú",
    "Thi hành án",
}
OUT_OF_SCOPE = re.compile(
    r"dat dai|quyen su dung dat|so do|thua dat|thu hoi dat|chuyen muc dich su dung dat|manh dat|"
    r"hinh su|toi pham|pham toi|toi lua dao|cong an|bi cao|nguoi bi hai|ket an|an tu|phat tu|"
    r"gay thuong tich|xam hai suc khoe|danh nhau|tu ve|truy cuu trach nhiem hinh su|"
    r"lao dong|cong nhan|tien luong|thu viec|bao hiem xa hoi|dinh cong|nguoi su dung lao dong",
    re.I,
)
FOLLOW_UP_TOPIC_CORRECTIONS = {
    854: "Thi hành án",
    857: "Hôn nhân, gia đình, trẻ em, bình đẳng giới",
    860: "Hôn nhân, gia đình, trẻ em, bình đẳng giới",
    867: "Hành chính tư pháp",
    885: "Dân sự",
    888: "Dân sự",
    890: "Dân sự",
    894: "Dân sự",
    900: "Dân sự",
    932: "Thi hành án",
    1000: "Thi hành án",
}
FOLLOW_UP_REMOVALS = {862, 886, 944}
FINAL_TOPIC_CORRECTIONS = {
    145: "Xây dựng, nhà ở, đô thị", 455: "Xây dựng, nhà ở, đô thị",
    494: "Xây dựng, nhà ở, đô thị", 778: "Xây dựng, nhà ở, đô thị",
    1069: "Xây dựng, nhà ở, đô thị", 2100: "Xây dựng, nhà ở, đô thị",
    2159: "Xây dựng, nhà ở, đô thị", 2451: "Tố tụng và các phương thức giải quyết tranh chấp",
    1084: "Xây dựng, nhà ở, đô thị", 1267: "Xây dựng, nhà ở, đô thị",
}
FINAL_TEXT_REPLACEMENTS = {
    64: (("sử dụng đât", "sử dụng đất"),), 119: (("tranh chấp giũa", "tranh chấp giữa"),),
    196: (("khởi kiên", "khởi kiện"),), 259: (("quyết đinh", "quyết định"), ("chổng cũ", "chồng cũ")),
    367: (("đc", "được"), ("A nh", "Anh"), ("ko", "không")),
    391: (("vợ vhồng", "vợ chồng"), ("cùa", "của")), 411: (("từ chối.Vậy", "từ chối. Vậy"),),
    438: (("triệu động", "triệu đồng"),), 781: (("Câu hỏi ", ""), ("TÒa án", "Tòa án")),
    1854: (("bố tối", "bố tôi"), ("phản đổi", "phản đối")), 1961: (("giao giao", "giao"),),
    2056: (("có phài", "có phải"),), 2107: (("đó không", "đó không?"),),
    2140: (("nghĩa vụ nghĩa vụ", "nghĩa vụ"),), 2228: (("tlà hắc mắc", "thắc mắc"),),
    2245: (("mổi tháng", "mỗi tháng"),), 2292: (("làm nhưng thủ tục", "làm những thủ tục"),),
    2451: (("bồi trường", "bồi thường"),),
}
SECOND_TEXT_REPLACEMENTS = {
    10: (("Nghị định định 61", "Nghị định 61"),),
    122: (("có có phần", "có phần"),), 140: (("thì thì chị", "thì chị"),),
    158: (("Chứng minh dân dân", "Chứng minh nhân dân"),), 217: (("mức giá giá mới", "mức giá mới"),),
    415: (("vào năm năm 2007", "vào năm 2007"), ("làm làm thủ tục", "làm thủ tục")),
    423: (("quyền quyền, nghĩa vụ", "quyền, nghĩa vụ"),),
    461: (("thi hành hành án", "thi hành án"), ("nhà nhà cho ở nhờ", "nhà cho ở nhờ")),
    557: (("vợ chồng tôi tôi ly hôn", "vợ chồng tôi ly hôn"),),
    890: (("kết hôn với với người khác", "kết hôn với người khác"),),
    931: (("cấp giấy giấy chứng nhận", "cấp giấy chứng nhận"),),
    1049: (("Trưởng phòng Phòng Tư pháp", "Trưởng phòng Tư pháp"), ("đóng đấu", "đóng dấu")),
    1076: (("không không?", "không?"),), 1087: (("và và một ngôi nhà", "và một ngôi nhà"),),
    1141: (("thì có có được", "thì có được"),), 1336: (("chồng cũ cũ", "chồng cũ"),),
    1361: (("với với trường hợp", "với trường hợp"),), 1405: (("của của chị", "của chị"),),
    1467: (("có có một căn nhà", "có một căn nhà"),), 1567: (("sang tên tên xe", "sang tên xe"),),
    1713: (("thì thì mức", "thì mức"), ("đuợc", "được")),
    1842: (("có có chấp nhận", "có chấp nhận"),),
    1868: (("tài sản thừa thừa kế", "tài sản thừa kế"), ("Tờ khai khai thừa kế", "Tờ khai thừa kế")),
    1884: (("tiến hành hành thủ tục", "tiến hành thủ tục"),),
    1909: (("từ người người chồng", "từ người chồng"),),
    1975: (("Ca trực của của tôi", "Ca trực của tôi"),),
    2000: (("làm phiếu Phiếu lý lịch tư pháp", "làm Phiếu lý lịch tư pháp"),),
    2021: (("đấu giá giá viên", "đấu giá viên"),), 2102: (("mảnh mảnh đất", "mảnh đất"),),
    2113: (("tư tư vấn", "tư vấn"),), 2178: (("thủ tục để để bà ngoại", "thủ tục để bà ngoại"),),
    2223: (("em sẽ sẽ có quyền", "em sẽ có quyền"), ("Rât", "Rất")),
    2296: (("có có giấy đăng ký kết hôn", "có giấy đăng ký kết hôn"),),
    2367: (("chỉ có thể thể lập", "chỉ có thể lập"),),
    196: (("10tỷ", "10 tỷ"),), 387: (("dân s ự", "dân sự"),),
    393: (("h ành vi", "hành vi"),), 438: (("tạm định chỉ thi hành án", "tạm đình chỉ thi hành án"),),
    749: (("lam Chứng minh nhân dân", "làm Chứng minh nhân dân"), ("nơi sinh(từ", "nơi sinh (từ")),
    781: (("\" Mọi", "\"Mọi"),), 2245: (("(con tôi sinh năm 2004.", "(con tôi sinh năm 2004)"),),
}
FOLLOW_UP_REPLACEMENT_EXCLUSION = re.compile(
    r"ma túy|nghiện|tội|xử phạt|đe dọa|hành hung|xã hội đen|công ty tài chính|"
    r"ngân hàng|thất nghiệp|lãi suất",
    re.I,
)


def normalize(question: str) -> str:
    return re.sub(r"[^\w]", "", question.casefold())


def scope_text(question: str) -> str:
    normalized = unicodedata.normalize("NFD", question.casefold()).replace("đ", "d")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def apply_topic_audit(selected: list[dict[str, str]], seen: set[str]) -> tuple[list[dict[str, str]], dict[str, int]]:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    flagged = {item["id"]: item for item in audit["flagged_questions"]}
    retained: list[dict[str, str]] = []
    removed_by_topic: dict[str, int] = {}

    for item_id, item in enumerate(selected, start=1):
        finding = flagged.get(item_id)
        if finding is None:
            retained.append(item)
            continue
        if item["question"] != finding["question"]:
            raise RuntimeError(f"Audit không khớp câu hỏi có ID {item_id}.")
        if finding["status"] == "Không phù hợp 7 topic":
            removed_by_topic[item["topic"]] = removed_by_topic.get(item["topic"], 0) + 1
            continue
        if finding["status"] == "Nhầm topic":
            item = {**item, "topic": finding["suggested_topic"]}
        retained.append(item)

    if len(flagged) != len(audit["flagged_questions"]):
        raise RuntimeError("Audit chứa ID trùng lặp.")
    return retained, removed_by_topic


def replacement_is_eligible(topic: str, row: dict) -> bool:
    if topic == "Dân sự":
        return row["lv_name"] == "Dân sự"
    return next(selector for name, _, selector in TOPIC_SELECTORS if name == topic)(row)


def apply_follow_up_audit(
    selected: list[dict[str, str]], rows: list[dict], seen: set[str]
) -> list[dict[str, str]]:
    replacement_positions = []
    procedure_topic = "Tố tụng và các phương thức giải quyết tranh chấp"
    replacement_topic = "Dân sự"
    for item_id, item in enumerate(selected, start=1):
        if item_id in FOLLOW_UP_TOPIC_CORRECTIONS:
            if item["topic"] != procedure_topic:
                raise RuntimeError(f"Audit bổ sung không khớp topic hiện tại của ID {item_id}.")
            selected[item_id - 1] = {**item, "topic": FOLLOW_UP_TOPIC_CORRECTIONS[item_id]}
        elif item_id in FOLLOW_UP_REMOVALS:
            if item["topic"] != procedure_topic:
                raise RuntimeError(f"Audit bổ sung không khớp topic hiện tại của ID {item_id}.")
            replacement_positions.append(item_id - 1)

    replacement_index = 0
    for row in rows:
        if replacement_index == len(replacement_positions):
            break
        question = " ".join(row["question_text"].split())
        question_key = normalize(question)
        if (
            replacement_is_eligible(replacement_topic, row)
            and row["lv_name"] in ALLOWED_SOURCE_TOPICS
            and not OUT_OF_SCOPE.search(scope_text(question))
            and not FOLLOW_UP_REPLACEMENT_EXCLUSION.search(question)
            and question_key not in seen
        ):
            selected[replacement_positions[replacement_index]] = {
                "question": question,
                "topic": replacement_topic,
            }
            seen.add(question_key)
            replacement_index += 1
    if replacement_index != len(replacement_positions):
        raise RuntimeError("Không đủ câu Dân sự để thay thế câu ngoài phạm vi.")
    return selected


def add_land_questions(selected: list[dict[str, str]], rows: list[dict], seen: set[str]) -> list[dict[str, str]]:
    candidates = []
    for row in rows:
        question = " ".join(row["question_text"].split())
        question_key = normalize(question)
        if row["lv_name"] == "Đất đai" and question_key not in seen:
            candidates.append(question)
            seen.add(question_key)
    selected.extend({"question": question, "topic": "Đất đai"} for question in candidates)
    return selected


SOURCE_TOPIC_MAP = {
    "Dân sự": "Dân sự", "Giao dịch đảm bảo": "Dân sự",
    "Hôn nhân gia đình": "Hôn nhân, gia đình, trẻ em, bình đẳng giới",
    "Con nuôi": "Hôn nhân, gia đình, trẻ em, bình đẳng giới",
    "Công chứng": "Bổ trợ tư pháp", "Chứng thực": "Bổ trợ tư pháp",
    "Quản lý luật sư": "Bổ trợ tư pháp", "Giám định tư pháp": "Bổ trợ tư pháp", "Đấu giá tài sản": "Bổ trợ tư pháp",
    "Hành chính tư pháp": "Hành chính tư pháp", "Hộ tịch": "Hành chính tư pháp",
    "Lý lịch tư pháp": "Hành chính tư pháp", "Quốc tịch": "Hành chính tư pháp", "Cư trú": "Hành chính tư pháp",
    "Thi hành án": "Thi hành án", "Xây dựng, nhà ở, đô thị": "Xây dựng, nhà ở, đô thị",
    "Đất đai": "Đất đai",
}


def add_remaining_source_questions(selected: list[dict[str, str]], rows: list[dict], seen: set[str]) -> list[dict[str, str]]:
    for row in rows:
        topic = SOURCE_TOPIC_MAP.get(row["lv_name"])
        question = " ".join(row["question_text"].split())
        question_key = normalize(question)
        if topic and question_key not in seen and not OUT_OF_SCOPE.search(scope_text(question)):
            selected.append({"question": question, "topic": topic})
            seen.add(question_key)
    return selected


def add_commercial_arbitration_questions(selected: list[dict[str, str]], rows: list[dict], seen: set[str]) -> list[dict[str, str]]:
    pattern = re.compile(r"trọng tài.*tòa án|tòa án.*trọng tài", re.I | re.S)
    for row in rows:
        question = " ".join(row["question_text"].split())
        question_key = normalize(question)
        if (
            row["lv_name"] == "Doanh nghiệp, hợp tác xã"
            and pattern.search(question)
            and question_key not in seen
        ):
            selected.append({"question": question, "topic": "Tố tụng và các phương thức giải quyết tranh chấp"})
            seen.add(question_key)
    return selected


def apply_final_corrections(selected: list[dict[str, str]]) -> list[dict[str, str]]:
    for item_id, topic in FINAL_TOPIC_CORRECTIONS.items():
        selected[item_id - 1] = {**selected[item_id - 1], "topic": topic}
    for item_id, replacements in FINAL_TEXT_REPLACEMENTS.items():
        question = selected[item_id - 1]["question"]
        for old, new in replacements:
            question = question.replace(old, new)
        selected[item_id - 1] = {**selected[item_id - 1], "question": question}
    for item_id, replacements in SECOND_TEXT_REPLACEMENTS.items():
        question = selected[item_id - 1]["question"]
        for old, new in replacements:
            question = question.replace(old, new)
        selected[item_id - 1] = {**selected[item_id - 1], "question": question}
    return selected


def main() -> None:
    rows = [json.loads(line) for line in SOURCE_PATH.open(encoding="utf-8")]
    selected: list[dict[str, str]] = []
    seen = set()

    for topic, quota, selector in TOPIC_SELECTORS:
        topic_questions = []
        for row in rows:
            question = " ".join(row["question_text"].split())
            question_key = normalize(question)
            if (
                selector(row)
                and row["lv_name"] in ALLOWED_SOURCE_TOPICS
                and not OUT_OF_SCOPE.search(scope_text(question))
                and question_key not in seen
            ):
                topic_questions.append(question)
                seen.add(question_key)
                if len(topic_questions) == quota:
                    break
        if len(topic_questions) != quota:
            raise RuntimeError(f"{topic}: chỉ có {len(topic_questions)}/{quota} câu hỏi phù hợp.")
        selected.extend({"question": question, "topic": topic} for question in topic_questions)

    if len(selected) != 1_000:
        raise RuntimeError(f"Kỳ vọng 1.000 câu hỏi, nhận được {len(selected)}.")

    selected, removed_by_topic = apply_topic_audit(selected, seen)
    for topic, missing_count in removed_by_topic.items():
        for row in rows:
            question = " ".join(row["question_text"].split())
            question_key = normalize(question)
            if (
                replacement_is_eligible(topic, row)
                and row["lv_name"] in ALLOWED_SOURCE_TOPICS
                and not OUT_OF_SCOPE.search(scope_text(question))
                and question_key not in seen
            ):
                selected.append({"question": question, "topic": topic})
                seen.add(question_key)
                missing_count -= 1
                if missing_count == 0:
                    break
        if missing_count:
            raise RuntimeError(f"Không đủ câu thay thế cho chủ đề {topic}: thiếu {missing_count}.")

    if len(selected) != 1_000:
        raise RuntimeError(f"Sau audit, kỳ vọng 1.000 câu hỏi gốc, nhận được {len(selected)}.")
    selected = apply_follow_up_audit(selected, rows, seen)
    selected = add_land_questions(selected, rows, seen)
    selected = add_remaining_source_questions(selected, rows, seen)
    selected = add_commercial_arbitration_questions(selected, rows, seen)
    random.Random(SHUFFLE_SEED).shuffle(selected)
    selected = apply_final_corrections(selected)

    OUTPUT_PATH.write_text(
        json.dumps(
            [
                {
                    "id": index,
                    **question,
                    "relevant_docs": [],
                    "relevant_articles": [],
                }
                for index, question in enumerate(selected, start=1)
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    topic_counts: dict[str, int] = {}
    for item in selected:
        topic_counts[item["topic"]] = topic_counts.get(item["topic"], 0) + 1
    report = [
        "# Báo cáo thu thập câu hỏi cho tập test",
        "",
        "## Tập dữ liệu đầu ra",
        "",
        f"- Tệp: `{OUTPUT_PATH.relative_to(ROOT)}`.",
        f"- Số câu hỏi: {len(selected):,}.",
        "- Schema: `id`, `question`, `topic`.",
        "- Tập này chỉ chứa câu hỏi; không lưu đáp án, ngữ cảnh hay citation từ dataset nguồn.",
        "- Đã áp dụng audit chủ đề thủ công: 40 câu ngoài phạm vi bị loại, 56 câu được gán lại chủ đề.",
        "- Đã áp dụng audit bổ sung: 11 câu được gán lại chủ đề, 3 câu ngoài phạm vi được thay bằng câu Dân sự nguyên gốc.",
        "- Đã áp dụng vòng rà soát tiếp theo: sửa 41 lỗi chữ/OCR/dấu câu chắc chắn và chuyển 2 câu rõ ràng sang Xây dựng, nhà ở, đô thị.",
        "- Các trường hợp có thể làm đổi nghĩa pháp lý hoặc còn tranh luận (ID 62, 1062, 2039 và các câu liên ngành) được giữ nguyên để rà soát chuyên gia.",
        "",
        "## Nguồn đã sử dụng",
        "",
        "| Nguồn | Dữ liệu đã lấy | Số câu được chọn | Ghi chú |",
        "|---|---|---:|---|",
        f"| `tmquan/pbgdpl-vn-legal-qna` | `question_text` từ Q&A công khai của Cổng Phổ biến, giáo dục pháp luật thuộc Bộ Tư pháp | {len(selected):,} | Chỉ dùng câu hỏi và nhãn lĩnh vực; không dùng `answer_text`. |",
        "",
        "## Nguồn đã tải nhưng chưa đưa vào tập hiện tại",
        "",
        "| Nguồn | Lý do chưa sử dụng |",
        "|---|---|",
        "| `R2AIStage1DATA.json` | Thiên về doanh nghiệp, thuế, kế toán, lao động và sở hữu trí tuệ; không khớp ổn định với phạm vi hiện tại. |",
        "| `NamSyntax/Vietnamese-Legal-QA-RAG` | Đã rà toàn bộ 420 câu; các câu khớp từ khóa chủ yếu thuộc doanh nghiệp hoặc môi trường nên chưa đưa vào để bảo toàn phạm vi. |",
        "| `nguyenthanhasia/vn-legal-qa` | Dataset bị giới hạn quyền truy cập (Hugging Face 403); chưa tải được. |",
        "",
        "## Phân bố theo chủ đề",
        "",
        "| Chủ đề | Số câu |",
        "|---|---:|",
        *[f"| {topic} | {count} |" for topic, count in topic_counts.items()],
        "",
        "## Quy tắc lọc",
        "",
        "- Chỉ nhận bản ghi thuộc các nhãn nguồn tương ứng với 7 chủ đề dự án.",
        "- Loại Hình sự và Lao động/BHXH bằng cả nhãn nguồn và kiểm tra từ khóa sau chuẩn hóa Unicode.",
        "- Bổ sung toàn bộ câu Đất đai nguyên gốc không trùng từ cùng nguồn; không thay thế các câu thuộc chủ đề khác.",
        "- Chuẩn hóa khoảng trắng và loại câu trùng theo chuỗi câu hỏi đã casefold.",
        "- Giữ nguyên câu hỏi gốc; không sinh mới, không diễn đạt lại.",
        f"- Xáo trộn thứ tự bằng seed cố định `{SHUFFLE_SEED}`; ID được đánh lại sau khi xáo trộn.",
        "- Đối soát cuối: toàn bộ câu hỏi đầu ra khớp với `question_text` của dataset nguồn sau chuẩn hóa, không có câu tự sinh hoặc diễn đạt lại.",
        "",
        "## Lưu ý khi gán đáp án",
        "",
        "Câu hỏi nguồn được công bố trong giai đoạn 2007–2021, nên có thể chứa tiền đề pháp lý cũ. "
        "Ở bước gán đáp án, phải retrieval trên corpus hiện hành, gắn Điều nguồn và kiểm tra trạng thái hiệu lực; "
        "không dùng đáp án của dataset nguồn làm chuẩn.",
        "",
        "## Tái lập",
        "",
        "```powershell",
        "python scripts/collect_project_test_questions.py",
        "```",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Đã tạo {len(selected)} câu hỏi tại {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
