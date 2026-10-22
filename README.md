# Vietnamese AI Legal Assistant

This repository is being rebuilt around a focused Vietnamese legal corpus.

## Active Data

- `data/current/legal_corpus.json`: chunked active corpus from repaired Phap Dien and VBPL records, including documented coverage additions.
- `data/current/legal_corpus_citable.json`: derived, citation-ready subset for the eight focus topics. `citation_level` is `article` or `document`; document chunks must not be cited as specific articles. Rebuild it whenever active changes.
- `data/current/manifest.json`: corpus schema, topic counts, and quality-status summary.
- `data/current/legal_corpus_unchunked.json`: ready-only snapshot reconstructed from the active corpus; one top-level record per `law_id` and article, with other text versions in `variants`. It is not updated automatically when active changes.
- `data/current/legal_corpus_full.json`: full article/document text from VBPL plus the legacy Phap Dien corpus, with one top-level record per `law_id` and article; `manifest_full.json` records coverage and freshness limits.

The active scope includes civil law, population/family/children/gender equality,
construction/housing/urban areas, land law, civil judicial support, judicial
administration, enforcement, and litigation/dispute resolution.
Supplemental documents outside these topics use `topic = "Ngoài phạm vi 8 chủ đề"`
and retain their inclusion reason in `topic_assignment_source`.
Each active record has `validity_tag` (`in_force`, `expired`,
`partially_expired`, `not_yet_effective`, `suspended`, or `unknown`) and
`validity_tag_source`. These describe document-level portal metadata, not the
validity of an individual article.

After changing the active corpus, repair conservative metadata and rebuild its
citation-ready subset:

```powershell
python scripts/repair_active_corpus.py --dry-run
python scripts/repair_active_corpus.py
```

The repair preserves all source records and content. It assigns stable IDs to
identifiable Phap Dien records, downgrades article records with missing or malformed
article numbers to `recoverable`, and marks duplicate recoverable document chunks.
Records with an unnumbered `law_id` such as `Không số` stay in active for search but
have `citation_level = none` and are excluded from the citable subset.

## Question labeling

Use `notebooks/label_questions_batch.ipynb` to label
`data/test/questions_with_difficulty.json` through OpenAI directly and/or Gemini
through OpenRouter using direct requests. Fill in `OPENAI_API_KEY` / `OPENAI_MODEL`,
`OPENROUTER_API_KEY` / `GEMINI_MODEL` (an OpenRouter `google/...` model ID), and
`PROMPT`. An API key set to `None` skips that provider; use `getpass` for hidden key
entry. Optional `OPENAI_PARAMS` and `GEMINI_PARAMS` omit values set to `None`.
Gemini parameters use OpenRouter Chat Completions names, such as `max_tokens`.
Use a regular model ID without the `:batch` suffix; see the
[OpenRouter quickstart](https://openrouter.ai/docs/quickstart).
Set `LIMIT` to the number of questions to try (default: 5), or `None` for all.
The API cell calls each question directly, prints progress, and saves
`labels_openai.json` and/or `labels_gemini.json` after each result. Rerunning it
starts over. Errors retain partial labels, and the other provider can continue.
Only fully completed providers are included in the merge step.
JSON replies wrapped in a single Markdown code fence are accepted. Empty or
invalid replies stop that provider and save the raw response to
`response_error_<provider>.txt` for inspection; invalid HTTP JSON bodies are
reported separately from invalid model output.
Its final cell merges `relevant_docs` and `relevant_articles` into the original
`data/test/questions.json` by ID, preserving other fields and untested questions,
and sorting by ID.
Results default to `data/test/batch_labeling/questions_completed_openai.json` and
`questions_completed_gemini.json` in the same directory;
each experiment overwrites these output files. Run the offline checks with
`python -B -m unittest discover -s tests -p test_label_questions_batch.py`.

### Round 2 judging

See [Round 2 judging](docs/round2_judging.md) for the local Jupyter notebook
that merges the seven Round 1 outputs into one candidate pool for manual LLM
judging with the prompt in `prompts/round2_manual_judge.md` and final human review.

## Legacy

- `data/legacy/2026-09-22_pre-domain-focus/`: historical data snapshot.
- `all_legacy/2026-09-22_pre-rebuild/`: previous application code, notebooks,
  tests, and runtime configuration.

Legacy artifacts are reference-only and must not be used by the rebuilt pipeline.

## Data Collection

- `notebooks/vbpl_gateway_crawler_v2.py`: crawler lấy URL văn bản trung ương từ sitemap VBPL và nội dung/metadata từ Gateway công khai.
- `notebooks/vbpl_gateway_crawler.ipynb`: notebook crawler cũ để tham khảo.

Crawler kiểm tra `robots.txt`, giới hạn URL trung ương bằng metadata cơ quan ban hành
ở ranh giới sitemap địa phương, giãn cách request một giây và checkpoint từng văn bản.
Cache ứng viên tách theo chế độ discovery. Một số ID Gateway có thể trả HTTP 500;
trạng thái này được giữ trong `data/raw/vbpl_gateway/checkpoint_v2.json`.

## Raw-record preparation

Split downloaded VBPL records before converting them into the article-level
corpus:

```powershell
python scripts/split_vbpl_records.py
```

The command creates `data/raw/vbpl_gateway/records_ready_for_conversion.jsonl` for records
with a non-empty text body, document identity, issuer, issue date, and portal validity
metadata. Records missing any required value are written to
`data/raw/vbpl_gateway/records_need_enrichment.jsonl`, with an `_enrichment`
field listing the missing paths.

Convert the ready records into an article-oriented corpus without replacing the
active corpus:

```powershell
python scripts/preprocess_vbpl_corpus.py
```

It writes document metadata and retrieval records to `data/processed/vbpl_gateway/`
and its preprocessing report to `reports/vbpl_preprocessing_report.md`. It also
writes unchunked source records to `data/processed/vbpl_gateway/legal_corpus_full.jsonl`.

Build the full-text corpus after full-only preprocessing:

```powershell
python scripts/preprocess_vbpl_corpus.py --full-only
python scripts/build_full_corpus.py
```

The full corpus retains expired and partially effective documents. Its primary
record is chosen deterministically from the available sources; other distinct
texts remain in `variants`. It is not a statement that the primary text is the
currently applicable version. Refresh sitemap discovery and audit missing
candidate URLs before treating coverage as complete.

Patch the chunked active corpus from records already crawled and preprocessed:

```powershell
python scripts/augment_active_from_full_processed.py --include-law-id 59/2020/QH14 --exclude-law-id 17/2026/QĐ-TTg
```

The script adds referenced laws and documents issued since the latest date in active,
keeps existing records, and leaves unparsed article headings out. It updates
`data/current/manifest.json` and `reports/active_corpus_patch_report.md`.

To add all processed source documents classified into the eight active topics
while excluding documents issued after the intended snapshot date:

```powershell
python scripts/augment_active_from_full_processed.py --recent-after 1900-01-01 --issue-date-through 2026-09-28
```

The early date selects the full processed source; existing active records are
preserved. Apply the validity-tag command below afterwards.

After rebuilding or patching active, refresh document-level validity tags:

```powershell
python scripts/tag_active_corpus_validity.py
```

The command matches legacy Phap Dien source `ItemID` and `law_id` against the
preprocessed Gateway documents. Unmatched records remain `unknown`; details are
recorded in `reports/active_corpus_validity_report.md`.

Để tiếp tục crawl từ checkpoint và tự xây lại corpus đầy đủ sau mỗi batch:

```powershell
python scripts/refresh_vbpl_corpus.py
```

Theo dõi `data/raw/vbpl_gateway/pipeline_status.json` và
`reports/full_corpus_report.md`. Corpus đầy đủ giữ cả văn bản hết hiệu lực để
phục vụ câu hỏi lịch sử; `validity_status` của nguồn Pháp điển cũ có thể chưa rõ.

Retry các ID trong `checkpoint_v2.json` và xây lại corpus đầy đủ nếu lấy thêm raw:

```powershell
python scripts/retry_failed_vbpl.py
```
