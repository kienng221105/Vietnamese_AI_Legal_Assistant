"""Offline preparation and resumable OpenRouter batch judging for round 2.

The exported labels are provisional. This module never treats a model judgment as
legal verification or fetches a missing statute on the model's behalf.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


BATCH_URL = "https://openrouter.ai/api/v1/batches"
MODEL = "openai/gpt-6-luna"
WEB_SEARCH_USD_PER_CALL = 0.01
ARTICLE_RE = re.compile(r"(?<!\w)Điều\s+(\d+[A-Za-zĐđ]?)\b", re.IGNORECASE)
SYSTEM_PROMPT = """Bạn kiểm tra ứng viên căn cứ pháp luật Việt Nam cho bài toán truy hồi.
Chỉ dùng nội dung và metadata nguồn được cung cấp. Không dùng trí nhớ để xác nhận
số hiệu, nội dung hoặc hiệu lực của Điều. Phân biệt câu hỏi lịch sử với câu hỏi
về quy định tại ngày tham chiếu. Metadata hiệu lực là cấp văn bản, không chứng
minh hiệu lực từng Điều/khoản. Nếu thời điểm hoặc hiệu lực chưa rõ, ghi unclear.

Với mỗi candidate_id, chấm relevance 2 nếu Điều trực tiếp hỗ trợ ít nhất một ý
hỏi; 1 nếu chỉ là bối cảnh; 0 nếu không liên quan; null nếu chưa đủ căn cứ.
Trích nguyên văn một đoạn ngắn từ content cho mỗi relevance 2. Không thêm, sửa,
hoặc bỏ candidate_id. Ghi các ý hỏi chưa có căn cứ vào uncovered_issues.
Mỗi reason chỉ một câu ngắn; evidence_quote chỉ trích đoạn cần thiết.
Với relevance 0, 1 hoặc null, đặt evidence_quote là chuỗi rỗng nếu không có
đoạn trực tiếp; không trả null. reason phải là chuỗi cho mọi ứng viên.
Trả đúng một JSON object gồm question_id, judgments, uncovered_issues,
temporal_note. Mỗi judgment gồm candidate_id, relevance, applicability
(applicable/not_applicable/unclear), evidence_quote, supported_issues, reason.
"""


def normalize_law_id(value: str) -> str:
    return "".join(value.split()).rstrip("*").upper()


def normalize_article(value: str) -> str | None:
    matches = ARTICLE_RE.findall(value)
    if len(matches) != 1:
        return None
    return "Điều " + matches[0].lower()


def article_key(law_id: str, article: str) -> tuple[str, str] | None:
    if not isinstance(law_id, str) or not isinstance(article, str):
        return None
    law = normalize_law_id(law_id)
    number = normalize_article(article)
    return (law, number) if law and number else None


def candidate_id(question_id: int, key: tuple[str, str]) -> str:
    raw = json.dumps([question_id, *key], ensure_ascii=False)
    return f"q{question_id}-{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_questions(path: Path) -> dict[int, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError("Questions must be a JSON array")
    questions = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("id")) is not int or not isinstance(row.get("question"), str):
            raise ValueError("Invalid question row")
        if row["id"] in questions:
            raise ValueError(f"Duplicate question ID: {row['id']}")
        questions[row["id"]] = row
    return questions


def load_candidates(paths: list[Path], question_ids: set[int]) -> tuple[dict[int, dict], dict[int, list[str]]]:
    pooled: dict[int, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    review: dict[int, list[str]] = defaultdict(list)
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("Model result filenames must be unique")
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected JSON array: {path}")
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or type(row.get("id")) is not int:
                raise ValueError(f"Invalid outer result in {path.name}")
            question_id = row["id"]
            if question_id in seen:
                raise ValueError(f"Duplicate ID {question_id} in {path.name}")
            seen.add(question_id)
            if question_id not in question_ids:
                continue
            output = row.get("model_output")
            if not isinstance(output, dict) or output.get("id") != question_id:
                review[question_id].append(f"{path.stem}: invalid_model_output_or_id")
                continue
            docs, articles = output.get("relevant_docs"), output.get("relevant_articles")
            if not isinstance(docs, list) or not isinstance(articles, list):
                review[question_id].append(f"{path.stem}: invalid_label_arrays")
                continue
            if any(not isinstance(doc, str) or not doc.strip() for doc in docs):
                review[question_id].append(f"{path.stem}: invalid_document_entry")
            doc_ids = {normalize_law_id(doc) for doc in docs if isinstance(doc, str) and doc.strip()}
            used_doc_ids = set()
            for item in articles:
                if not isinstance(item, dict):
                    review[question_id].append(f"{path.stem}: invalid_article_entry")
                    continue
                law, article = item.get("law_id"), item.get("article")
                key = article_key(law, article)
                if key is None:
                    review[question_id].append(f"{path.stem}: unsupported_reference:{str(law)[:80]}:{str(article)[:80]}")
                    continue
                used_doc_ids.add(key[0])
                if key[0] not in doc_ids:
                    review[question_id].append(f"{path.stem}: article_law_missing_from_docs:{key[0]}")
                pooled[question_id].setdefault(key, {"candidate_id": candidate_id(question_id, key),
                                                      "law_id": key[0], "article": key[1], "proposed_by": []})
                voters = pooled[question_id][key]["proposed_by"]
                if path.stem not in voters:
                    voters.append(path.stem)
            for doc_id in doc_ids - used_doc_ids:
                review[question_id].append(f"{path.stem}: document_only_candidate:{doc_id}")
        missing = question_ids - seen
        for question_id in sorted(missing):
            review[question_id].append(f"{path.stem}: missing_question")
    return {qid: {key: value for key, value in sorted(group.items())} for qid, group in pooled.items()}, review


def load_evidence(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    units: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Citable corpus must be a JSON array")
    for row in rows:
        if row.get("citation_level") != "article" or row.get("quality_status") != "ready":
            continue
        key = article_key(row.get("law_id"), row.get("article"))
        if key is None:
            continue
        unit_id = row.get("parent_record_id") or row.get("record_id")
        if unit_id:
            units[key][str(unit_id)].append(row)

    evidence = {}
    for key, groups in units.items():
        assembled = []
        for records in groups.values():
            first = records[0]
            if first.get("unit_type") == "article_chunk":
                records.sort(key=lambda row: row.get("chunk_number") or 0)
                numbers = [row.get("chunk_number") for row in records]
                if numbers != list(range(1, len(records) + 1)):
                    continue
            elif len(records) != 1:
                continue
            content = "\n".join(str(row.get("content") or "") for row in records)
            if not content.strip():
                continue
            assembled.append({
                "content": content,
                "record_ids": [row["record_id"] for row in records],
                "source_urls": list(dict.fromkeys(url for row in records for url in row.get("source_urls", []))),
                "source_system": first.get("source_system"),
                "issue_date": first.get("issue_date"),
                "effective_from": first.get("effective_from"),
                "effective_to": first.get("effective_to"),
                "validity_tag": first.get("validity_tag"),
                "law_name": first.get("law_name"),
            })
        evidence[key] = assembled
    return evidence


def evidence_for(key: tuple[str, str], evidence: dict, max_chars: int) -> tuple[dict | None, str | None]:
    versions = evidence.get(key, [])
    if not versions:
        return None, "missing_or_incomplete_citable_article"
    by_text: dict[str, list[dict]] = defaultdict(list)
    for version in versions:
        by_text[" ".join(version["content"].split()).casefold()].append(version)
    if len(by_text) != 1:
        return None, "conflicting_article_text_versions"
    representatives = next(iter(by_text.values()))
    chosen = sorted(representatives, key=lambda item: (
        item["source_system"] == "vbpl_gateway", item["issue_date"] or "", item["record_ids"][0]
    ), reverse=True)[0]
    if len(chosen["content"]) > max_chars:
        return None, "article_too_long_for_configured_limit"
    if not chosen["source_urls"]:
        return None, "missing_source_url"
    return {
        "content": chosen["content"],
        "content_sha256": hashlib.sha256(chosen["content"].encode()).hexdigest(),
        "law_name": chosen["law_name"],
        "sources": [{field: version[field] for field in (
            "record_ids", "source_urls", "source_system", "issue_date", "effective_from", "effective_to", "validity_tag"
        )} for version in representatives],
    }, None


def make_request(question: dict, candidates: list[dict], as_of_date: str, reasoning_effort: str) -> dict:
    public_candidates = []
    for candidate in candidates:
        evidence = candidate["evidence"]
        versions = evidence["sources"]
        public_candidates.append({
            "candidate_id": candidate["candidate_id"],
            "law_id": candidate["law_id"],
            "article": candidate["article"],
            "evidence": {
                "content": evidence["content"],
                "law_name": evidence["law_name"],
                "source_url": next((url for version in versions for url in version["source_urls"]), None),
                "validity_tags": sorted({version["validity_tag"] or "unknown" for version in versions}),
                "issue_dates": sorted({version["issue_date"] for version in versions if version["issue_date"]}),
                "effective_from": sorted({version["effective_from"] for version in versions if version["effective_from"]}),
                "effective_to": sorted({version["effective_to"] for version in versions if version["effective_to"]}),
            },
        })
    payload = {"question_id": question["id"], "question": question["question"],
               "topic": question.get("topic"), "as_of_date": as_of_date, "candidates": public_candidates}
    completion_limit = min(6000, max(1200, 250 + 140 * len(public_candidates)))
    return {
        "custom_id": f"question-{question['id']}",
        "body": {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "reasoning": {"effort": reasoning_effort},
            "response_format": {"type": "json_object"},
            "prompt_cache_options": {"mode": "explicit"},
            "max_completion_tokens": completion_limit,
        },
    }


def request_cost_ceiling(request: dict, input_usd_per_mtok: float, output_usd_per_mtok: float) -> dict[str, float]:
    """Planning proxy, not a provider-guaranteed billing cap.

    UTF-8 byte count loosely proxies prompt tokens; 1,024 tokens cover chat framing.
    Output is bounded by its configured cap. Native search reserves each configured
    built-in tool call at the current search price. Actual usage comes from the
    completed batch; provider prices/settings must be rechecked before use.
    """
    prompt_bytes = len(json.dumps(request["body"], ensure_ascii=False).encode("utf-8"))
    input_proxy = prompt_bytes + 1024
    output_cap = request["body"].get("max_completion_tokens", request["body"].get("max_output_tokens", 0))
    input_cost = input_proxy * input_usd_per_mtok / 1_000_000
    output_cost = output_cap * output_usd_per_mtok / 1_000_000
    search_cost_reserve = (WEB_SEARCH_USD_PER_CALL * request["body"].get("max_tool_calls", 1)
                           if any(tool.get("type") == "web_search"
                                  for tool in request["body"].get("tools", [])) else 0.0)
    return {"input_token_proxy": input_proxy, "output_token_cap": output_cap,
            "input_planning_usd": input_cost, "output_planning_usd": output_cost,
            "web_search_cost_reserve_usd": search_cost_reserve,
            "planning_usd": input_cost + output_cost + search_cost_reserve}


def plan_cost_ceiling(plan: dict, input_usd_per_mtok: float, output_usd_per_mtok: float) -> dict[str, float]:
    estimates = [request_cost_ceiling(request, input_usd_per_mtok, output_usd_per_mtok)
                 for request in plan["requests"]]
    return {"requests": len(estimates),
            "input_token_proxy": sum(item["input_token_proxy"] for item in estimates),
            "output_token_cap": sum(item["output_token_cap"] for item in estimates),
            "input_planning_usd": sum(item["input_planning_usd"] for item in estimates),
            "output_planning_usd": sum(item["output_planning_usd"] for item in estimates),
            "web_search_cost_reserve_usd": sum(item["web_search_cost_reserve_usd"] for item in estimates),
            "planning_usd": sum(item["planning_usd"] for item in estimates)}


def build_plan(questions_path: Path, result_paths: list[Path], corpus_path: Path,
               selected_ids: list[int], as_of_date: str, max_candidates: int = 40,
               max_article_chars: int = 12000, max_request_chars: int = 80000,
               reasoning_effort: str = "low", model_id: str = MODEL) -> dict[str, Any]:
    if not result_paths or not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Select unique question IDs and at least one model result")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of_date):
        raise ValueError("as_of_date must use YYYY-MM-DD")
    date.fromisoformat(as_of_date)
    if min(max_candidates, max_article_chars, max_request_chars) <= 0:
        raise ValueError("Limits must be positive")
    questions = load_questions(questions_path)
    unknown = set(selected_ids) - questions.keys()
    if unknown:
        raise ValueError(f"Unknown question IDs: {sorted(unknown)[:10]}")
    pool, pool_review = load_candidates(result_paths, set(selected_ids))
    evidence = load_evidence(corpus_path)
    cases = {}
    requests = []
    for qid in selected_ids:
        candidates = list(pool.get(qid, {}).values())
        reasons = list(dict.fromkeys(pool_review.get(qid, [])))
        prepared = []
        if len(candidates) > max_candidates:
            reasons.append(f"too_many_candidates:{len(candidates)}")
        else:
            for candidate in candidates:
                source, problem = evidence_for((candidate["law_id"], candidate["article"]), evidence, max_article_chars)
                if problem:
                    reasons.append(f"{candidate['candidate_id']}:{problem}")
                else:
                    prepared.append({**candidate, "evidence": source})
        if not candidates:
            reasons.append("no_article_candidates")
        if len(candidates) <= max_candidates and prepared:
            request = make_request(questions[qid], prepared, as_of_date, reasoning_effort)
            request_size = len(json.dumps(request, ensure_ascii=False))
            if request_size > max_request_chars:
                reasons.append(f"request_too_long:{request_size}")
            else:
                requests.append(request)
        cases[str(qid)] = {"question": questions[qid], "candidates": candidates,
                           "sent_candidate_ids": [c["candidate_id"] for c in prepared]
                           if len(candidates) <= max_candidates and prepared and request_size <= max_request_chars else [],
                           "review_reasons": reasons}
    source_paths = [questions_path, *result_paths, corpus_path]
    plan = {"schema_version": 1, "model": model_id, "as_of_date": as_of_date,
            "reasoning_effort": reasoning_effort,
            "limits": {"max_candidates": max_candidates, "max_article_chars": max_article_chars,
                       "max_request_chars": max_request_chars},
            "input_hashes": {str(path.resolve()): file_hash(path) for path in source_paths},
            "selected_ids": selected_ids, "cases": cases, "requests": requests}
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return plan


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_plan(run_dir: Path, plan: dict) -> None:
    path = run_dir / "plan.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("This run directory contains a different plan; choose a new RUN_NAME")
        return
    write_json_atomic(path, plan)


def load_run(run_dir: Path, state_filename: str = "batch_state.json") -> tuple[dict, dict]:
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    state_path = run_dir / state_filename
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "plan_sha256": plan["plan_sha256"], "batches": [], "results": {}}
    if state.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("Batch state does not match the plan")
    return plan, state


def accounted_batch_cost(batch: dict) -> float:
    """Use reported charges, including upstream inference for BYOK batches."""
    if batch.get("status") not in {"completed", "failed", "expired", "cancelled"}:
        return batch["planning_usd"]
    usage = batch.get("usage")
    if not isinstance(usage, dict) or not isinstance(usage.get("cost"), (int, float)):
        return batch["planning_usd"]
    if not usage.get("is_byok"):
        return float(usage["cost"])
    upstream = usage.get("cost_details", {}).get("upstream_inference_cost")
    return float(usage["cost"] + upstream) if isinstance(upstream, (int, float)) else batch["planning_usd"]


def submit_next_batch(run_dir: Path, api_key: str, max_requests: int = 25, session=None,
                      *, budget_usd: float = 1.0, input_usd_per_mtok: float = 0.0625,
                      output_usd_per_mtok: float = 0.25) -> dict | None:
    import requests

    if max_requests <= 0 or min(budget_usd, input_usd_per_mtok, output_usd_per_mtok) <= 0:
        raise ValueError("Request count, budget, and planning prices must be positive")
    plan, state = load_run(run_dir)
    if any(batch["status"] == "submission_uncertain" for batch in state["batches"]):
        raise RuntimeError("A prior submission is uncertain; inspect the OpenRouter batches dashboard before retrying")
    submitted = {cid for batch in state["batches"] for cid in batch["custom_ids"]}
    pending = [request for request in plan["requests"] if request["custom_id"] not in submitted]
    if not pending:
        return None
    committed = sum(accounted_batch_cost(batch) for batch in state["batches"])
    remaining = budget_usd - committed
    selected = []
    planned_cost = 0.0
    for request in pending[:max_requests]:
        cost = request_cost_ceiling(request, input_usd_per_mtok, output_usd_per_mtok)["planning_usd"]
        if planned_cost + cost > remaining:
            break
        selected.append(request)
        planned_cost += cost
    if not selected:
        raise ValueError(f"Planning budget exhausted: ${committed:.4f} already committed against ${budget_usd:.4f}")
    entry = {"status": "submission_uncertain", "batch_id": None,
             "custom_ids": [request["custom_id"] for request in selected],
             "planning_usd": planned_cost}
    state["batches"].append(entry)
    write_json_atomic(run_dir / "batch_state.json", state)
    client = session or requests.Session()
    batch_request = {"endpoint": "/v1/responses" if plan.get("mode") == "candidate_only_web_search"
                     else "/v1/chat/completions", "model": plan["model"]}
    if plan.get("mode") == "candidate_only_web_search":
        batch_request["provider"] = {"only": ["openai"]}
    batch_request["requests"] = selected
    response = client.post(BATCH_URL, headers={"Authorization": f"Bearer {api_key}"},
                           json=batch_request, timeout=120)
    response.raise_for_status()
    body = response.json()
    if response.status_code != 202 or not body.get("id"):
        raise ValueError("Batch submission did not return a batch ID")
    entry.update({"status": body.get("status", "validating"), "batch_id": body["id"]})
    write_json_atomic(run_dir / "batch_state.json", state)
    return entry


def cost_report(state: dict) -> dict[str, float | int]:
    batches = state["batches"]
    usages = [batch["usage"] for batch in batches if isinstance(batch.get("usage"), dict)]
    result_usages = [record["usage"] for record in state.get("results", {}).values()
                     if isinstance(record.get("usage"), dict)]

    def usage_detail(usage: dict, name: str) -> int | None:
        for field in ("prompt_tokens_details", "input_tokens_details"):
            details = usage.get(field)
            if isinstance(details, dict) and isinstance(details.get(name), int):
                return details[name]
        return None

    read_counts = [usage_detail(usage, "cached_tokens") for usage in usages]
    write_counts = [usage_detail(usage, "cache_write_tokens") for usage in usages]
    search_counts = [usage.get("server_tool_use", {}).get("web_search_requests")
                     for usage in result_usages]
    search_counts = [value for value in search_counts if isinstance(value, int)]
    return {"batches": len(batches),
            "reported_or_reserved_usd": sum(accounted_batch_cost(batch) for batch in batches),
            "input_tokens_reported": sum(usage.get("prompt_tokens") or usage.get("input_tokens") or 0 for usage in usages),
            "output_tokens_reported": sum(usage.get("completion_tokens") or usage.get("output_tokens") or 0 for usage in usages),
            "web_search_requests_reported": sum(search_counts) if search_counts else None,
            "cache_read_tokens_reported": sum(read_counts) if read_counts and all(v is not None for v in read_counts) else None,
            "cache_write_tokens_reported": sum(write_counts) if write_counts and all(v is not None for v in write_counts) else None}


def kbench_output_limit(llm: Any, max_tokens: int) -> dict[str, int]:
    """Map one output cap to the two Kaggle Model Proxy backends."""
    backend = type(llm).__name__
    if backend == "GoogleGenAI":
        return {"max_output_tokens": max_tokens}
    if backend == "OpenAI":
        return {"max_completion_tokens": max_tokens}
    raise TypeError(f"Unsupported kbench model backend: {backend}")


def run_kbench(run_dir: Path, kbench: Any, llm: Any, max_new_requests: int = 5,
               max_run_usd: float = 1.0, state_filename: str = "kbench_state.json") -> dict:
    """Judge a bounded number of questions on Kaggle and checkpoint each reply."""
    if max_new_requests <= 0 or max_run_usd <= 0:
        raise ValueError("Request count and run budget must be positive")
    plan, state = load_run(run_dir, state_filename)
    if getattr(llm, "model", plan["model"]) != plan["model"]:
        raise ValueError("The selected kbench model differs from the frozen plan")
    if state.get("inflight"):
        raise RuntimeError(f"Uncertain previous model call: {state['inflight']}; inspect Kaggle run logs before continuing")
    state.setdefault("inflight", None)
    state.setdefault("usage_by_question", {})
    spent = sum(item["cost_usd"] for item in state["usage_by_question"].values()
                if isinstance(item.get("cost_usd"), (int, float)))
    if any(item.get("cost_usd") is None for item in state["usage_by_question"].values()):
        raise RuntimeError("Kaggle did not report cost for a previous call; check quota before continuing")
    pending = [request for request in plan["requests"]
               if request["custom_id"].removeprefix("question-") not in state["results"]]
    for request in pending[:max_new_requests]:
        if spent >= max_run_usd:
            break
        qid = request["custom_id"].removeprefix("question-")
        message = request["body"]["messages"][1]["content"]
        sent = json.loads(message)["candidates"]
        evidence_by_id = {candidate["candidate_id"]: candidate["evidence"] for candidate in sent}
        output_params = kbench_output_limit(llm, request["body"]["max_completion_tokens"])
        state["inflight"] = request["custom_id"]
        write_json_atomic(run_dir / state_filename, state)
        system_prompt = request["body"]["messages"][0]["content"]
        with kbench.chats.new(name=f"round2-{qid}", system_instructions=system_prompt) as chat:
            response = llm.prompt(message, reasoning=plan["reasoning_effort"],
                                  extra_api_params=output_params)
        usage = chat.usage
        input_cost = usage.input_tokens_cost_nanodollars
        output_cost = usage.output_tokens_cost_nanodollars
        cost = (input_cost + output_cost) / 1_000_000_000 if input_cost is not None and output_cost is not None else None
        state["usage_by_question"][qid] = {
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "input_cost_nanodollars": input_cost, "output_cost_nanodollars": output_cost,
            "cost_usd": cost,
        }
        try:
            parsed = parse_judgment(int(qid), {"choices": [{"finish_reason": "stop", "message": {"content": response}}]},
                                    set(evidence_by_id), evidence_by_id)
            state["results"][qid] = {"status": "judged", "judgment": parsed}
        except (ValueError, KeyError, TypeError) as error:
            state["results"][qid] = {"status": "needs_review", "reason": str(error), "raw_output": response}
        state["inflight"] = None
        write_json_atomic(run_dir / state_filename, state)
        if cost is None:
            break
        spent += cost
    return state


def parse_judgment(question_id: int, body: dict, expected_ids: set[str], evidence_by_id: dict) -> dict:
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Incomplete chat completion")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Model returned no JSON text")
    result = json.loads(content)
    if not isinstance(result, dict) or result.get("question_id") != question_id:
        raise ValueError("Incorrect question ID")
    judgments = result.get("judgments")
    if not isinstance(judgments, list):
        raise ValueError("Missing judgments list")
    validation_reasons = []
    uncovered = result.get("uncovered_issues")
    if not isinstance(uncovered, list) or not all(isinstance(value, str) for value in uncovered):
        validation_reasons.append("invalid_uncovered_issues")
        result["uncovered_issues"] = []
    if not isinstance(result.get("temporal_note"), str):
        validation_reasons.append("missing_temporal_note")
        result["temporal_note"] = ""
    seen = set()
    valid_judgments = []
    for index, original in enumerate(judgments):
        item = dict(original) if isinstance(original, dict) else None
        if not isinstance(item, dict) or not isinstance(item.get("candidate_id"), str):
            validation_reasons.append(f"invalid_judgment:{index}")
            continue
        cid = item["candidate_id"]
        if cid not in expected_ids:
            validation_reasons.append(f"unknown_candidate_id:{cid}")
            continue
        if cid in seen:
            validation_reasons.append(f"duplicate_candidate_id:{cid}")
            continue
        seen.add(cid)
        issues = []
        if item.get("relevance") not in (0, 1, 2, None) or type(item.get("relevance")) is bool:
            issues.append("invalid_relevance")
            item["relevance"] = None
        if item.get("applicability") not in ("applicable", "not_applicable", "unclear"):
            issues.append("invalid_applicability")
            item["applicability"] = "unclear"
        supported = item.get("supported_issues")
        if supported is None:
            item["supported_issues"] = []
        elif not isinstance(supported, list) or not all(isinstance(value, str) for value in supported):
            issues.append("invalid_supported_issues")
            item["supported_issues"] = []
        quote_value = item.get("evidence_quote")
        if quote_value is None:
            item["evidence_quote"] = ""
        elif not isinstance(quote_value, str):
            issues.append("invalid_evidence_quote")
            item["evidence_quote"] = ""
        reason_value = item.get("reason")
        if reason_value is None:
            item["reason"] = ""
        elif not isinstance(reason_value, str):
            issues.append("invalid_reason")
            item["reason"] = ""
        if not item["reason"].strip():
            issues.append("missing_reason")
        quote = " ".join(item["evidence_quote"].split()).casefold()
        source = " ".join(evidence_by_id[cid]["content"].split()).casefold()
        item["quote_verified"] = len(quote) >= 24 and quote in source
        item["decision"] = (
            "provisional_positive" if item["relevance"] == 2 and item["applicability"] == "applicable"
            and item["quote_verified"] and bool(item["supported_issues"])
            else "provisional_non_positive" if item["relevance"] in (0, 1) and not issues
            else "needs_review"
        )
        if item["decision"] == "needs_review" and not issues:
            issues.append("insufficient_positive_evidence_or_temporal_status")
        item["validation_reasons"] = issues
        valid_judgments.append(item)
    validation_reasons.extend(f"missing_candidate_id:{cid}" for cid in sorted(expected_ids - seen))
    result["judgments"] = valid_judgments
    result["validation_reasons"] = validation_reasons
    return result


def reparse_saved_judgments(run_dir: Path, state_filename: str = "kbench_state.json") -> dict[str, int]:
    """Revalidate saved model replies using the current parser without API calls."""
    plan, state = load_run(run_dir, state_filename)
    requests = {request["custom_id"].removeprefix("question-"): request for request in plan["requests"]}
    recovered = 0
    changed = False
    for qid, record in state["results"].items():
        request = requests.get(qid)
        if request is None:
            continue
        raw = record.get("raw_output") if record.get("status") == "needs_review" else record.get("judgment")
        if not isinstance(raw, (str, dict)):
            continue
        content = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        sent = json.loads(request["body"]["messages"][1]["content"])["candidates"]
        evidence_by_id = {candidate["candidate_id"]: candidate["evidence"] for candidate in sent}
        try:
            parsed = parse_judgment(int(qid), {"choices": [{"finish_reason": "stop", "message": {"content": content}}]},
                                    set(evidence_by_id), evidence_by_id)
        except (ValueError, KeyError, TypeError):
            continue
        if record.get("status") != "judged":
            recovered += 1
        if record.get("judgment") != parsed or record.get("status") != "judged":
            state["results"][qid] = {**record, "status": "judged", "judgment": parsed}
            state["results"][qid].pop("raw_output", None)
            state["results"][qid].pop("reason", None)
            changed = True
    if changed:
        write_json_atomic(run_dir / state_filename, state)
    return {"recovered_questions": recovered, "state_updated": int(changed)}


def collect_batches(run_dir: Path, api_key: str, session=None) -> dict:
    import requests

    plan, state = load_run(run_dir)
    requests_by_id = {request["custom_id"]: request for request in plan["requests"]}
    client = session or requests.Session()
    for batch in state["batches"]:
        if not batch.get("batch_id") or batch["status"] in {"completed", "failed", "expired", "cancelled"}:
            continue
        response = client.get(f"{BATCH_URL}/{batch['batch_id']}",
                              headers={"Authorization": f"Bearer {api_key}"}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        if payload.get("id") != batch["batch_id"]:
            raise ValueError("Unexpected batch ID")
        batch["status"] = payload["status"]
        batch["usage"] = payload.get("usage")
        if batch["status"] == "completed":
            received = set()
            for item in payload.get("results") or []:
                cid = item.get("custom_id")
                if cid not in batch["custom_ids"] or cid in received:
                    raise ValueError("Unexpected or duplicate result ID")
                received.add(cid)
                question_id = int(cid.removeprefix("question-"))
                request = requests_by_id[cid]
                try:
                    if item.get("error") or not item.get("response") or item["response"].get("status_code") != 200:
                        raise ValueError(f"Request failed: {item.get('error') or item.get('response', {}).get('status_code')}")
                    if plan.get("mode") == "candidate_only_web_search":
                        from . import round2_candidate_judge as candidate_judge
                        case = plan["cases"][str(question_id)]
                        article_keys = {(row["law_id"], row["article"])
                                        for row in case["candidate_articles"]}
                        parsed = candidate_judge.parse_judgment(
                            question_id, item["response"]["body"], set(case["candidate_docs"]), article_keys)
                    else:
                        sent = json.loads(request["body"]["messages"][1]["content"])["candidates"]
                        evidence_by_id = {candidate["candidate_id"]: candidate["evidence"] for candidate in sent}
                        parsed = parse_judgment(question_id, item["response"]["body"],
                                                set(evidence_by_id), evidence_by_id)
                    state["results"][str(question_id)] = {"status": "judged", "judgment": parsed,
                                                           "usage": item["response"]["body"].get("usage"),
                                                           "raw_body": item["response"]["body"]
                                                           if plan.get("mode") == "candidate_only_web_search" else None}
                except (ValueError, KeyError, TypeError) as error:
                    state["results"][str(question_id)] = {"status": "needs_review", "reason": str(error)}
            for cid in set(batch["custom_ids"]) - received:
                state["results"][cid.removeprefix("question-")] = {"status": "needs_review", "reason": "missing_batch_result"}
        write_json_atomic(run_dir / "batch_state.json", state)
    return state


def export_provisional(run_dir: Path, state_filename: str = "batch_state.json") -> list[dict]:
    plan, state = load_run(run_dir, state_filename)
    output = []
    for qid in plan["selected_ids"]:
        case = plan["cases"][str(qid)]
        result = state["results"].get(str(qid), {})
        reasons = list(case["review_reasons"])
        relevant_articles = []
        if result.get("status") == "judged":
            judgment = result["judgment"]
            reasons.extend(judgment.get("validation_reasons", []))
            by_id = {candidate["candidate_id"]: candidate for candidate in case["candidates"]}
            for item in judgment["judgments"]:
                reasons.extend(f"{item['candidate_id']}:{reason}" for reason in item.get("validation_reasons", []))
                if item["decision"] == "provisional_positive":
                    candidate = by_id[item["candidate_id"]]
                    relevant_articles.append({"law_id": candidate["law_id"], "article": candidate["article"]})
                elif item["decision"] == "needs_review":
                    reasons.append(f"{item['candidate_id']}:judge_needs_review")
            reasons.extend(f"uncovered_issue:{issue}" for issue in judgment["uncovered_issues"])
        else:
            reasons.append(result.get("reason", "no_completed_judgment"))
        relevant_articles.sort(key=lambda row: (row["law_id"], row["article"]))
        output.append({"id": qid, "question": case["question"]["question"],
                       "topic": case["question"].get("topic"),
                       "relevant_docs": sorted({row["law_id"] for row in relevant_articles}),
                       "relevant_articles": relevant_articles,
                       "review_status": "needs_review" if reasons or not relevant_articles else "provisional",
                       "review_reasons": list(dict.fromkeys(reasons))})
    write_json_atomic(run_dir / "provisional_labels.json", output)
    return output
