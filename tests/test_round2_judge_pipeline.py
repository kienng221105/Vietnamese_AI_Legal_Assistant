"""Offline regression checks for the round-2 judge notebook's reusable code."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from notebooks import round2_judge_pipeline as judge


def source_article(number: int, content: str, record_id: str) -> dict:
    return {
        "law_id": "1/2024/QH15", "law_name": "Văn bản mẫu", "article": f"Điều {number}",
        "content": content, "record_id": record_id, "unit_type": "article",
        "quality_status": "ready", "citation_level": "article",
        "source_urls": [f"https://vbpl.vn/van-ban/chi-tiet/--sample#{number}"],
        "source_system": "vbpl_gateway", "validity_tag": "in_force",
        "issue_date": "2024-01-01", "effective_from": "2024-02-01",
    }


def model_row(qid: int, number: int) -> dict:
    return {"id": qid, "model_output": {
        "id": qid, "relevant_docs": ["1/2024/QH15"],
        "relevant_articles": [{"law_id": "1/2024/QH15", "article": f"Điều {number}"}],
    }}


def fixture(root: Path, ambiguous: bool = False) -> tuple[Path, list[Path], Path]:
    questions = root / "questions.json"
    questions.write_text(json.dumps([
        {"id": 1, "question": "Tình huống 1?", "topic": "Dân sự"},
        {"id": 2, "question": "Tình huống 2?", "topic": "Dân sự"},
    ]), encoding="utf-8")
    results = []
    for name in ("a", "b"):
        path = root / f"{name}.json"
        path.write_text(json.dumps([model_row(1, 1), model_row(2, 2)]), encoding="utf-8")
        results.append(path)
    corpus = root / "citable.json"
    rows = [source_article(1, "Quy định về nội dung tình huống một.", "one"),
            source_article(2, "Nội dung thứ nhất của Điều hai.", "two")]
    if ambiguous:
        rows.append(source_article(2, "Nội dung khác của Điều hai.", "two-other"))
    corpus.write_text(json.dumps(rows), encoding="utf-8")
    return questions, results, corpus


class Response:
    def __init__(self, body: dict, status_code: int = 200):
        self.body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.body


class Session:
    def __init__(self, answer: dict):
        self.posts = []
        self.gets = []
        self.answer = answer

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return Response({"id": "batch-demo", "status": "validating"}, 202)

    def get(self, url, **kwargs):
        self.gets.append(url)
        return Response(self.answer)


class Round2JudgeTests(unittest.TestCase):
    def test_ambiguous_article_is_reviewed_and_model_does_not_see_votes(self):
        with tempfile.TemporaryDirectory(prefix="round2-test-", dir=Path.cwd()) as directory:
            questions, results, corpus = fixture(Path(directory), ambiguous=True)
            plan = judge.build_plan(questions, results, corpus, [1, 2], "2026-09-28")

        self.assertEqual([request["custom_id"] for request in plan["requests"]], ["question-1"])
        self.assertTrue(any("conflicting_article_text_versions" in reason
                            for reason in plan["cases"]["2"]["review_reasons"]))
        sent = plan["requests"][0]["body"]["messages"][1]["content"]
        self.assertNotIn("proposed_by", sent)
        self.assertNotIn("vote_count", sent)
        self.assertNotIn("record_ids", sent)
        self.assertEqual(plan["requests"][0]["body"]["prompt_cache_options"], {"mode": "explicit"})
        self.assertEqual(plan["requests"][0]["body"]["max_completion_tokens"], 1200)
        self.assertEqual(len(plan["cases"]["1"]["candidates"][0]["proposed_by"]), 2)

    def test_positive_requires_real_quote_and_all_candidate_ids(self):
        cid = "q1-candidate"
        evidence = {cid: {"content": "Nội dung trực tiếp hỗ trợ tình huống."}}
        output = {"question_id": 1, "judgments": [{
            "candidate_id": cid, "relevance": 2, "applicability": "applicable",
            "evidence_quote": "Đoạn trích không có thật", "supported_issues": ["Vấn đề chính"],
            "reason": "Có liên quan",
        }], "uncovered_issues": [], "temporal_note": "Đã xét thời điểm"}
        body = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}]}
        parsed = judge.parse_judgment(1, body, {cid}, evidence)
        self.assertEqual(parsed["judgments"][0]["decision"], "needs_review")
        output["judgments"][0]["evidence_quote"] = "Nội dung trực tiếp hỗ trợ tình huống."
        body["choices"][0]["message"]["content"] = json.dumps(output)
        self.assertEqual(judge.parse_judgment(1, body, {cid}, evidence)["judgments"][0]["decision"],
                         "provisional_positive")
        output["judgments"][0]["supported_issues"] = []
        body["choices"][0]["message"]["content"] = json.dumps(output)
        self.assertEqual(judge.parse_judgment(1, body, {cid}, evidence)["judgments"][0]["decision"],
                         "needs_review")
        output["judgments"][0]["supported_issues"] = ["Vấn đề chính"]
        output["judgments"][0]["reason"] = None
        body["choices"][0]["message"]["content"] = json.dumps(output)
        without_reason = judge.parse_judgment(1, body, {cid}, evidence)["judgments"][0]
        self.assertEqual(without_reason["decision"], "provisional_positive")
        self.assertIn("missing_reason", without_reason["validation_reasons"])
        partial = judge.parse_judgment(1, body, {cid, "missing"}, evidence)
        self.assertIn("missing_candidate_id:missing", partial["validation_reasons"])

    def test_null_quote_on_irrelevant_article_preserves_other_positive(self):
        positive, negative = "q1-positive", "q1-negative"
        evidence = {positive: {"content": "Điều này trực tiếp giải quyết nội dung tranh chấp."},
                    negative: {"content": "Quy định về một chủ đề khác."}}
        output = {"question_id": 1, "judgments": [
            {"candidate_id": positive, "relevance": 2, "applicability": "applicable",
             "evidence_quote": evidence[positive]["content"], "supported_issues": ["Tranh chấp"],
             "reason": "Hỗ trợ trực tiếp"},
            {"candidate_id": negative, "relevance": 0, "applicability": "unclear",
             "evidence_quote": None, "supported_issues": [], "reason": "Không liên quan"},
        ], "uncovered_issues": [], "temporal_note": "Đã xét"}
        body = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}]}
        parsed = judge.parse_judgment(1, body, {positive, negative}, evidence)
        self.assertEqual([row["decision"] for row in parsed["judgments"]],
                         ["provisional_positive", "provisional_non_positive"])
        self.assertEqual(parsed["judgments"][1]["evidence_quote"], "")

    def test_saved_invalid_response_can_be_reparsed_without_model_call(self):
        with tempfile.TemporaryDirectory(prefix="round2-test-", dir=Path.cwd()) as directory:
            root = Path(directory)
            questions, results, corpus = fixture(root)
            run_dir = root / "run"
            plan = judge.build_plan(questions, results, corpus, [1], "2026-09-28")
            judge.save_plan(run_dir, plan)
            cid = plan["cases"]["1"]["sent_candidate_ids"][0]
            raw = {"question_id": 1, "judgments": [
                {"candidate_id": cid, "relevance": 2, "applicability": "applicable",
                 "evidence_quote": "Quy định về nội dung tình huống một.",
                 "supported_issues": ["Tình huống 1"], "reason": "Hỗ trợ trực tiếp"},
                {"candidate_id": "unknown", "relevance": 0, "applicability": "unclear",
                 "evidence_quote": None, "supported_issues": [], "reason": "Không liên quan"},
            ], "uncovered_issues": [], "temporal_note": "Phù hợp"}
            judge.write_json_atomic(run_dir / "kbench_state.json", {
                "plan_sha256": plan["plan_sha256"], "batches": [],
                "results": {"1": {"status": "needs_review", "reason": "Missing quote or reason",
                                  "raw_output": json.dumps(raw)}},
            })
            self.assertEqual(judge.reparse_saved_judgments(run_dir)["recovered_questions"], 1)
            labels = judge.export_provisional(run_dir, "kbench_state.json")
            self.assertEqual(labels[0]["relevant_articles"],
                             [{"law_id": "1/2024/QH15", "article": "Điều 1"}])
            self.assertEqual(labels[0]["review_status"], "needs_review")
            self.assertTrue(any("unknown_candidate_id" in reason for reason in labels[0]["review_reasons"]))
            self.assertEqual(judge.reparse_saved_judgments(run_dir)["recovered_questions"], 0)

    def test_submit_collect_and_export_resume_without_duplicate_charge(self):
        with tempfile.TemporaryDirectory(prefix="round2-test-", dir=Path.cwd()) as directory:
            root = Path(directory)
            questions, results, corpus = fixture(root)
            plan = judge.build_plan(questions, results, corpus, [1], "2026-09-28")
            run_dir = root / "run"
            judge.save_plan(run_dir, plan)
            cid = plan["cases"]["1"]["sent_candidate_ids"][0]
            content = "Quy định về nội dung tình huống một."
            verdict = {"question_id": 1, "judgments": [{
                "candidate_id": cid, "relevance": 2, "applicability": "applicable",
                "evidence_quote": content, "supported_issues": ["Tình huống 1"], "reason": "Hỗ trợ trực tiếp",
            }], "uncovered_issues": [], "temporal_note": "Phù hợp"}
            batch = {"id": "batch-demo", "status": "completed", "usage": {"cost": 0.01},
                     "results": [{"custom_id": "question-1", "response": {"status_code": 200, "body": {
                         "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(verdict)}}]
                     }}, "error": None}]}
            session = Session(batch)
            self.assertEqual(judge.submit_next_batch(run_dir, "fake-key", session=session)["batch_id"], "batch-demo")
            self.assertIsNone(judge.submit_next_batch(run_dir, "fake-key", session=session))
            self.assertEqual(len(session.posts), 1)
            self.assertNotIn("fake-key", (run_dir / "batch_state.json").read_text(encoding="utf-8"))
            state = judge.collect_batches(run_dir, "fake-key", session=session)
            self.assertEqual(state["results"]["1"]["status"], "judged")
            self.assertEqual(judge.export_provisional(run_dir)[0]["review_status"], "provisional")
            self.assertEqual(judge.export_provisional(run_dir)[0]["relevant_docs"], ["1/2024/QH15"])

    def test_uncertain_submission_blocks_automatic_retry(self):
        with tempfile.TemporaryDirectory(prefix="round2-test-", dir=Path.cwd()) as directory:
            root = Path(directory)
            questions, results, corpus = fixture(root)
            run_dir = root / "run"
            judge.save_plan(run_dir, judge.build_plan(questions, results, corpus, [1], "2026-09-28"))

            class LostResponse:
                def post(self, url, **kwargs):
                    raise TimeoutError("response lost after submit")

            with self.assertRaises(TimeoutError):
                judge.submit_next_batch(run_dir, "fake-key", session=LostResponse())
            with self.assertRaisesRegex(RuntimeError, "submission is uncertain"):
                judge.submit_next_batch(run_dir, "fake-key", session=LostResponse())
            _, state = judge.load_run(run_dir)
            self.assertEqual(state["batches"][0]["status"], "submission_uncertain")

    def test_planning_budget_blocks_paid_submission_and_accounts_for_byok(self):
        with tempfile.TemporaryDirectory(prefix="round2-test-", dir=Path.cwd()) as directory:
            root = Path(directory)
            questions, results, corpus = fixture(root)
            run_dir = root / "run"
            plan = judge.build_plan(questions, results, corpus, [1], "2026-09-28")
            judge.save_plan(run_dir, plan)
            estimate = judge.plan_cost_ceiling(plan, 0.0625, 0.25)
            self.assertAlmostEqual(estimate["planning_usd"],
                                   estimate["input_planning_usd"] + estimate["output_planning_usd"])
            session = Session({})
            with self.assertRaisesRegex(ValueError, "Planning budget exhausted"):
                judge.submit_next_batch(run_dir, "fake-key", session=session,
                                        budget_usd=estimate["planning_usd"] / 2)
            self.assertEqual(session.posts, [])

        batch = {"planning_usd": 0.1, "status": "completed", "usage": {"cost": 0.01, "is_byok": True,
                 "cost_details": {"upstream_inference_cost": 0.03},
                 "prompt_tokens": 100, "completion_tokens": 20,
                 "prompt_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 5}}}
        self.assertAlmostEqual(judge.accounted_batch_cost(batch), 0.04)
        report = judge.cost_report({"batches": [batch]})
        self.assertEqual((report["cache_read_tokens_reported"], report["cache_write_tokens_reported"]), (10, 5))
        batch["status"] = "in_progress"
        self.assertAlmostEqual(judge.accounted_batch_cost(batch), 0.1)

    def test_kbench_checks_model_checkpoints_usage_and_stops_at_budget(self):
        with tempfile.TemporaryDirectory(prefix="round2-test-", dir=Path.cwd()) as directory:
            root = Path(directory)
            questions, results, corpus = fixture(root)
            run_dir = root / "run"
            plan = judge.build_plan(questions, results, corpus, [1, 2], "2026-09-28")
            judge.save_plan(run_dir, plan)
            cid = plan["cases"]["1"]["sent_candidate_ids"][0]
            answer = json.dumps({"question_id": 1, "judgments": [{
                "candidate_id": cid, "relevance": 2, "applicability": "applicable",
                "evidence_quote": "Quy định về nội dung tình huống một.",
                "supported_issues": ["Tình huống 1"], "reason": "Hỗ trợ trực tiếp",
            }], "uncovered_issues": [], "temporal_note": "Phù hợp"})

            class Chat:
                usage = SimpleNamespace(input_tokens=200, output_tokens=80,
                                        input_tokens_cost_nanodollars=100_000_000,
                                        output_tokens_cost_nanodollars=0)

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            class Chats:
                def new(self, **kwargs):
                    self.last = kwargs
                    return Chat()

            class OpenAI:
                model = judge.MODEL

                def __init__(self):
                    self.calls = []

                def prompt(self, message, **kwargs):
                    self.calls.append(kwargs)
                    return answer

            backend = SimpleNamespace(chats=Chats())
            model = OpenAI()
            state = judge.run_kbench(run_dir, backend, model, max_new_requests=2, max_run_usd=0.05)
            self.assertEqual(len(model.calls), 1)
            self.assertEqual(model.calls[0]["extra_api_params"], {"max_completion_tokens": 1200})
            self.assertEqual(backend.chats.last["system_instructions"], judge.SYSTEM_PROMPT)
            self.assertEqual(state["usage_by_question"]["1"]["cost_usd"], 0.1)
            self.assertEqual(judge.export_provisional(run_dir, "kbench_state.json")[0]["review_status"],
                             "provisional")
            self.assertEqual(len(judge.run_kbench(run_dir, backend, model, max_new_requests=2,
                                                  max_run_usd=0.05)["results"]), 1)

if __name__ == "__main__":
    unittest.main()
