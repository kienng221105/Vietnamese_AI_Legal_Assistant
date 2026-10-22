"""Regression checks for the candidate-only OpenRouter batch pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

from notebooks import round2_candidate_judge as candidate_judge
from notebooks import round2_judge_pipeline as transport


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
    def __init__(self, get_body: dict | None = None):
        self.posts = []
        self.gets = []
        self.get_body = get_body or {}

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return Response({"id": "batch-demo", "status": "validating"}, 202)

    def get(self, url, **kwargs):
        self.gets.append(url)
        return Response(self.get_body)


def fixture(root: Path) -> tuple[Path, list[Path]]:
    questions = root / "questions.json"
    questions.write_text(json.dumps([{"id": 1, "question": "Câu hỏi mẫu?"}], ensure_ascii=False),
                         encoding="utf-8")
    results = []
    for name in ("model_a", "model_b"):
        path = root / f"{name}.json"
        output = {"id": 1, "relevant_docs": ["91/2015/QH13"],
                  "relevant_articles": [{"law_id": "91/2015/QH13", "article": "Điều 1"}]}
        path.write_text(json.dumps([{"id": 1, "model_output": output}], ensure_ascii=False), encoding="utf-8")
        results.append(path)
    return questions, results


def response_body(question_id: int = 1) -> dict:
    url = "https://vbpl.vn/example"
    text = json.dumps({
        "question_id": question_id,
        "relevant_docs": ["91/2015/QH13"],
        "relevant_articles": [{"law_id": "91/2015/QH13", "article": "Điều 1"}],
        "source_urls": [url],
        "need_review": False,
        "review_reasons": [],
    }, ensure_ascii=False)
    return {"output": [
        {"type": "web_search_call", "action": {"sources": [{"type": "url", "url": url}]}},
        {"type": "message", "content": [{"type": "output_text", "text": text,
                                             "annotations": [{"type": "url_citation", "url": url}]}]},
    ], "usage": {"input_tokens": 100, "output_tokens": 50, "cost": 0.02,
                 "server_tool_use": {"web_search_requests": 2}}}


class CandidateJudgeTests(unittest.TestCase):
    def test_build_plan_deduplicates_and_enables_native_search(self):
        with tempfile.TemporaryDirectory(prefix="round2-candidate-", dir=Path.cwd()) as directory:
            questions, results = fixture(Path(directory))
            plan = candidate_judge.build_plan(questions, results, [1])

        case = plan["cases"]["1"]
        self.assertEqual(case["candidate_docs"], ["91/2015/QH13"])
        self.assertEqual(len(case["candidate_articles"]), 1)
        self.assertEqual(case["candidate_articles"][0]["votes"], 2)
        body = plan["requests"][0]["body"]
        self.assertEqual(body["tools"][0]["type"], "web_search")
        self.assertEqual(body["tools"][0]["filters"]["allowed_domains"], candidate_judge.OFFICIAL_DOMAINS)
        self.assertEqual(body["tool_choice"], "required")
        self.assertEqual(body["max_tool_calls"], 2)
        self.assertEqual(plan["limits"]["endpoint"], "/v1/responses")
        sent = json.loads(body["input"])
        self.assertNotIn("votes", json.dumps(sent))
        self.assertNotIn("model_a", body["input"])

    def test_parser_keeps_only_candidates_and_verified_official_citations(self):
        parsed = candidate_judge.parse_judgment(
            1, response_body(), {"91/2015/QH13"}, {("91/2015/QH13", "Điều 1")})
        self.assertFalse(parsed["need_review"])
        self.assertEqual(parsed["relevant_docs"], ["91/2015/QH13"])
        self.assertEqual(parsed["relevant_articles"], [{"law_id": "91/2015/QH13", "article": "Điều 1"}])
        self.assertEqual(parsed["source_urls"], ["https://vbpl.vn/example"])

        bad = response_body()
        bad["output"][1]["content"][0]["text"] = bad["output"][1]["content"][0]["text"].replace(
            "91/2015/QH13", "invented-law")
        rejected = candidate_judge.parse_judgment(
            1, bad, {"91/2015/QH13"}, {("91/2015/QH13", "Điều 1")})
        self.assertTrue(rejected["need_review"])
        self.assertFalse(rejected["relevant_articles"])

    def test_batch_submission_pins_openai_and_collector_exports_labels(self):
        with tempfile.TemporaryDirectory(prefix="round2-candidate-", dir=Path.cwd()) as directory:
            root = Path(directory)
            questions, results = fixture(root)
            run_dir = root / "run"
            plan = candidate_judge.build_plan(questions, results, [1])
            candidate_judge.save_plan(run_dir, plan)
            session = Session({"id": "batch-demo", "status": "completed",
                               "usage": {"cost": 0.02, "prompt_tokens": 100, "completion_tokens": 50},
                               "results": [{"custom_id": "question-1", "error": None,
                                            "response": {"status_code": 200, "body": response_body()}}]})
            submitted = transport.submit_next_batch(run_dir, "fake-key", 5, session=session)
            request = session.posts[0][1]["json"]
            self.assertEqual(request["endpoint"], "/v1/responses")
            self.assertEqual(request["provider"], {"only": ["openai"]})
            self.assertNotIn("fake-key", (run_dir / "batch_state.json").read_text(encoding="utf-8"))

            state = transport.collect_batches(run_dir, "fake-key", session=session)
            self.assertEqual(state["results"]["1"]["status"], "judged")
            self.assertEqual(transport.cost_report(state)["web_search_requests_reported"], 2)
            labels = candidate_judge.export_labels(run_dir)
            self.assertFalse(labels[0]["need_review"])
            self.assertEqual(labels[0]["source_urls"], ["https://vbpl.vn/example"])
            self.assertEqual(submitted["batch_id"], "batch-demo")

    def test_planning_estimate_reserves_search_tool_limit(self):
        with tempfile.TemporaryDirectory(prefix="round2-candidate-", dir=Path.cwd()) as directory:
            questions, results = fixture(Path(directory))
            plan = candidate_judge.build_plan(questions, results, [1])
        estimate = transport.plan_cost_ceiling(plan, 0.05, 0.25)
        self.assertEqual(estimate["web_search_cost_reserve_usd"], 0.02)
        self.assertGreaterEqual(estimate["planning_usd"], 0.02)


if __name__ == "__main__":
    unittest.main()
