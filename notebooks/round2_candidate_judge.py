"""Build candidate-only Round 2 plans for OpenRouter Batch with native web search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import round2_judge_pipeline as legacy
except ImportError:
    from . import round2_judge_pipeline as legacy


OFFICIAL_DOMAINS = ["vbpl.vn", "vanban.chinhphu.vn", "congbao.chinhphu.vn"]
MAX_SEARCH_TOOL_CALLS = 2

SYSTEM_PROMPT = """Bạn sàng lọc độ liên quan của ứng viên Điều luật với câu hỏi pháp luật Việt Nam.
Bạn phải dùng công cụ web_search được cấp để tra cứu. Ưu tiên nguồn chính thức của Nhà nước.
Chỉ chọn ứng viên có trong input; không tự thêm văn bản hoặc Điều. Kiểm tra xem nội dung
nguồn tìm được có hỗ trợ trực tiếp câu hỏi không; mã văn bản hay tiêu đề giống nhau chưa đủ.
Nếu không tìm được nội dung Điều từ nguồn chính thức, kết quả mâu thuẫn, câu hỏi thiếu dữ
kiện, hoặc không thể kết luận đáng tin cậy, đặt need_review=true và nêu lý do.
Đây là bước sàng lọc, không tự kết luận hiệu lực pháp luật tại thời điểm lịch sử.

Trả đúng một JSON object gồm question_id, relevant_docs, relevant_articles, source_urls,
need_review, review_reasons. relevant_docs là danh sách law_id đã chọn;
relevant_articles là danh sách {law_id, article}. source_urls là URL nguồn chính thức
đã dùng. Chỉ trả về ứng viên có trong input. Nếu có liên quan nhưng chưa đủ chứng cứ,
đưa mục đó vào review_reasons thay vì khẳng định.
"""


def build_plan(questions_path: Path, result_paths: list[Path], selected_ids: list[int],
               max_candidates: int = 80, max_request_chars: int = 30000,
               reasoning_effort: str = "low", model_id: str = "openai/gpt-6-luna") -> dict[str, Any]:
    if not result_paths or not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Select unique question IDs and at least one model result")
    if min(max_candidates, max_request_chars) <= 0:
        raise ValueError("Limits must be positive")
    questions = legacy.load_questions(questions_path)
    unknown = set(selected_ids) - questions.keys()
    if unknown:
        raise ValueError(f"Unknown question IDs: {sorted(unknown)[:10]}")

    pool, pool_review = legacy.load_candidates(result_paths, set(selected_ids))
    doc_voters: dict[int, dict[str, set[str]]] = {qid: {} for qid in selected_ids}
    for path in result_paths:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        for row in rows:
            qid = row.get("id") if isinstance(row, dict) else None
            if qid not in doc_voters:
                continue
            output = row.get("model_output")
            docs = output.get("relevant_docs") if isinstance(output, dict) else None
            if not isinstance(docs, list):
                continue
            for doc in docs:
                if isinstance(doc, str) and doc.strip():
                    law_id = legacy.normalize_law_id(doc)
                    doc_voters[qid].setdefault(law_id, set()).add(path.stem)
    cases, requests = {}, []
    for qid in selected_ids:
        grouped = list(pool.get(qid, {}).values())
        docs = sorted(doc_voters[qid])
        articles = [{"law_id": candidate["law_id"], "article": candidate["article"],
                     "candidate_id": candidate["candidate_id"],
                     "votes": len(candidate["proposed_by"])} for candidate in grouped]
        reasons = list(dict.fromkeys(pool_review.get(qid, [])))
        if not docs and not articles:
            reasons.append("no_candidates_from_round1")
        if len(docs) + len(articles) > max_candidates:
            reasons.append(f"too_many_candidates:{len(docs) + len(articles)}")
        case = {"question": questions[qid], "candidate_docs": docs,
                "candidate_articles": articles, "review_reasons": reasons}
        cases[str(qid)] = case
        if len(docs) + len(articles) > max_candidates or not docs and not articles:
            continue
        payload = {"question_id": qid, "question": questions[qid]["question"],
                   "candidate_docs": docs,
                   "candidate_articles": [{"law_id": row["law_id"], "article": row["article"]}
                                          for row in articles]}
        request = {"custom_id": f"question-{qid}", "body": {
            "instructions": SYSTEM_PROMPT,
            "input": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            "reasoning": {"effort": reasoning_effort},
            "text": {"format": {"type": "json_object"}},
            "tools": [{"type": "web_search", "filters": {"allowed_domains": OFFICIAL_DOMAINS},
                       "search_context_size": "low"}],
            "tool_choice": "required",
            "max_tool_calls": MAX_SEARCH_TOOL_CALLS,
            "max_output_tokens": min(1200, max(300, 120 + 15 * len(articles))),
        }}
        if len(json.dumps(request, ensure_ascii=False).encode("utf-8")) > max_request_chars:
            case["review_reasons"].append(f"request_too_long:{len(json.dumps(request, ensure_ascii=False).encode('utf-8'))}")
        else:
            case["sent_candidate_ids"] = [row["candidate_id"] for row in articles]
            requests.append(request)
    plan = {"schema_version": 3, "mode": "candidate_only_web_search", "model": model_id,
            "reasoning_effort": reasoning_effort,
            "limits": {"max_candidates": max_candidates, "max_request_chars": max_request_chars,
                       "endpoint": "/v1/responses", "provider_only": ["openai"],
                       "max_search_tool_calls": MAX_SEARCH_TOOL_CALLS},
            "input_hashes": {str(path.resolve()): legacy.file_hash(path)
                             for path in [questions_path, *result_paths]},
            "selected_ids": selected_ids, "cases": cases, "requests": requests}
    plan["plan_sha256"] = hashlib.sha256(
        json.dumps(plan, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return plan


def save_plan(run_dir: Path, plan: dict) -> None:
    path = run_dir / "plan.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("This run directory contains a different plan; choose a new RUN_NAME")
    else:
        legacy.write_json_atomic(path, plan)
    pool = [{"id": qid, "question": plan["cases"][str(qid)]["question"]["question"],
             "candidate_docs": plan["cases"][str(qid)]["candidate_docs"],
             "candidate_articles": plan["cases"][str(qid)]["candidate_articles"],
             "review_reasons": plan["cases"][str(qid)]["review_reasons"]}
            for qid in plan["selected_ids"]]
    legacy.write_json_atomic(run_dir / "candidate_pool.json", pool)


def load_run(run_dir: Path, state_filename: str = "batch_state.json") -> tuple[dict, dict]:
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    state_path = run_dir / state_filename
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "plan_sha256": plan["plan_sha256"], "results": {}, "usage_by_question": {}, "inflight": None}
    if state.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("State does not match the frozen plan")
    return plan, state


def response_text(body: dict) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    texts = []
    for item in body.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                if isinstance(content.get("text"), str):
                    texts.append(content["text"])
    if not texts:
        raise ValueError("Responses API returned no output text")
    return "\n".join(texts)


def response_citations(body: dict) -> set[str]:
    citations = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            kind = value.get("type")
            url = value.get("url")
            if isinstance(url, str) and kind in {"url_citation", "url"}:
                citations.add(url)
            if kind == "web_search_call":
                action = value.get("action")
                sources = action.get("sources", []) if isinstance(action, dict) else []
                for source in sources:
                    if isinstance(source, dict) and isinstance(source.get("url"), str):
                        citations.add(source["url"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(body.get("output", []))
    return citations


def parse_judgment(question_id: int, body: dict, candidate_docs: set[str],
                   candidate_articles: set[tuple[str, str]]) -> dict:
    value = json.loads(response_text(body))
    if not isinstance(value, dict) or value.get("question_id") != question_id:
        raise ValueError("Incorrect question ID")
    docs, articles = value.get("relevant_docs"), value.get("relevant_articles")
    if not isinstance(docs, list) or not isinstance(articles, list):
        raise ValueError("Missing relevant_docs or relevant_articles list")
    review = value.get("need_review")
    reasons = value.get("review_reasons")
    source_urls = value.get("source_urls")
    if type(review) is not bool or not isinstance(reasons, list) or not all(isinstance(x, str) for x in reasons):
        raise ValueError("Invalid need_review or review_reasons")
    if not isinstance(source_urls, list) or not all(isinstance(x, str) for x in source_urls):
        invalid_source_shape = True
        source_urls = []
    else:
        invalid_source_shape = False
    if review and not reasons:
        reasons = ["judge_requested_review_without_reason"]
    invalid = []
    selected_docs = []
    for doc in docs:
        normalized_doc = legacy.normalize_law_id(doc) if isinstance(doc, str) else None
        if normalized_doc not in candidate_docs:
            invalid.append(f"unknown_relevant_doc:{doc}")
        elif normalized_doc not in selected_docs:
            selected_docs.append(normalized_doc)
    selected_articles = []
    seen = set()
    for item in articles:
        if not isinstance(item, dict) or not isinstance(item.get("law_id"), str) or not isinstance(item.get("article"), str):
            invalid.append("invalid_relevant_article")
            continue
        key = legacy.article_key(item["law_id"], item["article"])
        if key is None or key not in candidate_articles:
            invalid.append(f"unknown_relevant_article:{item['law_id']}:{item['article']}")
            continue
        if key not in seen:
            seen.add(key)
            selected_articles.append({"law_id": key[0], "article": key[1]})
    actual_citations = response_citations(body)
    approved_urls = []
    for url in source_urls:
        try:
            parsed_url = urlparse(url)
            hostname = (parsed_url.hostname or "").lower()
        except ValueError:
            invalid.append(f"unverified_or_nonofficial_source:{url[:160]}")
            continue
        official = any(hostname == domain or hostname.endswith("." + domain) for domain in OFFICIAL_DOMAINS)
        if url not in actual_citations or parsed_url.scheme != "https" or not official:
            invalid.append(f"unverified_or_nonofficial_source:{url[:160]}")
        elif url not in approved_urls:
            approved_urls.append(url)
    if invalid_source_shape:
        invalid.append("invalid_source_urls")
    if not approved_urls:
        invalid.append("no_verified_official_source")
    selected_docs = sorted(set(selected_docs) | {row["law_id"] for row in selected_articles})
    return {"relevant_docs": selected_docs,
            "relevant_articles": sorted(selected_articles, key=lambda row: (row["law_id"], row["article"])),
            "need_review": review or bool(invalid),
            "source_urls": approved_urls,
            "review_reasons": list(dict.fromkeys(reasons + invalid))}


def export_labels(run_dir: Path, state_filename: str = "batch_state.json") -> list[dict]:
    plan, state = load_run(run_dir, state_filename)
    output = []
    for qid in plan["selected_ids"]:
        case, result = plan["cases"][str(qid)], state["results"].get(str(qid), {})
        if result.get("status") == "judged":
            label = result["judgment"]
        else:
            label = {"relevant_docs": [], "relevant_articles": [], "source_urls": [], "need_review": True,
                     "review_reasons": [result.get("reason", "no_completed_judgment")]}
        reasons = list(dict.fromkeys(case["review_reasons"] + label["review_reasons"]))
        needs_review = label["need_review"] or bool(reasons) or not label["relevant_articles"]
        output.append({"id": qid, "question": case["question"]["question"],
                       "relevant_docs": label["relevant_docs"],
                       "relevant_articles": label["relevant_articles"],
                       "source_urls": label.get("source_urls", []),
                       "need_review": needs_review, "review_reasons": reasons})
    legacy.write_json_atomic(run_dir / "provisional_labels.json", output)
    return output
