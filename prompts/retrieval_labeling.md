# Prompt gán nhãn truy hồi pháp luật

Dùng với `data/test/project_scope_questions_without_topic.json`, một **snapshot cố định** của `data/current/legal_corpus.json` và, nếu cần, `data/review/source_citation_candidates.jsonl`. Giao các ID theo lô nhỏ để người rà soát có thể kiểm tra từng kết quả. Thay `[KHOẢNG ID]` trước khi gửi.

## Prompt gửi cho bên gán nhãn

Bạn gán nhãn **truy hồi văn bản và Điều luật** cho các câu hỏi pháp luật Việt Nam. Hãy xử lý các ID: **[KHOẢNG ID]**.

Tôi cung cấp:

1. Tệp câu hỏi gồm `id`, `question`, `relevant_docs: []`, `relevant_articles: []`.
2. Một snapshot corpus có số hiệu `law_id`, số Điều `article`, nội dung và URL nguồn.
3. Nếu có, tệp ứng viên trích dẫn từ đáp án hỏi đáp cũ. **Ứng viên này có thể sai, thiếu hoặc hết hiệu lực; không được chép thẳng vào nhãn.**

Bạn được tra cứu văn bản pháp luật trên nguồn chính thức khi corpus thiếu. Với mỗi câu, xác định tất cả những văn bản và Điều **thực sự chứa căn cứ để giải quyết ít nhất một ý hỏi**. Đọc nội dung Điều và so với ý hỏi cụ thể trước khi gán; việc hai văn bản có cùng từ khóa hoặc một số hiệu được nhắc trong đáp án cũ không đủ để coi là liên quan.

Kiểm tra bối cảnh thời gian: nếu câu hỏi chỉ rõ một thời điểm, đánh giá theo thời điểm đó; nếu hỏi quy định hiện hành, kiểm tra văn bản/Điều còn áp dụng. Văn bản hết hiệu lực vẫn có thể phù hợp với câu hỏi lịch sử. Không tự đoán hiệu lực từ tên văn bản.

Quy tắc dấu `*`:

- Nếu đã xác minh một văn bản hoặc Điều từ **văn bản gốc chính thức** nhưng văn bản/Điều đó thiếu trong snapshot corpus, thêm `*` vào cuối `law_id`.
- Nếu văn bản có cả Điều đã có và Điều thiếu: `relevant_docs` ghi `law_id*` một lần; trong `relevant_articles`, chỉ Điều thiếu dùng `law_id*`.
- Chỉ dùng `*` sau khi đã tra được văn bản gốc và ghi URL kiểm chứng vào **nhật ký rà soát riêng**. Không dùng `*` chỉ vì corpus thiếu mà đáp án hỏi đáp cũ có nhắc số hiệu.
- Dấu `*` không thuộc số hiệu pháp luật. Khi so khớp giữa các trường, bỏ dấu `*` cuối chuỗi.

Nếu chưa xác minh được căn cứ, để các mảng của câu đó rỗng và ghi ID vào nhật ký cần rà soát. Mảng rỗng **không có nghĩa câu hỏi không có đáp án**. Không tạo số hiệu, số Điều, URL hoặc trích dẫn theo trí nhớ.

## Đầu ra bắt buộc

Xuất một tệp JSON dạng mảng. Mỗi ID được giao có đúng một object và **chỉ có bốn trường** sau:

```json
[
  {
    "id": 1,
    "question": "Giữ nguyên câu hỏi đầu vào",
    "relevant_docs": ["SO_HIEU_VAN_BAN"],
    "relevant_articles": [
      {"law_id": "SO_HIEU_VAN_BAN", "article": "Điều N"}
    ]
  }
]
```

Ví dụ trên chỉ minh họa cấu trúc; không sao chép giá trị giữ chỗ. Không thêm `topic`, điểm số liên quan, câu trả lời hoặc nhận xét vào tệp nhãn.

Quy tắc kiểm tra:

1. Giữ nguyên `id` và `question`, không bỏ sót hoặc trùng ID.
2. `relevant_docs` là các `law_id` duy nhất của văn bản có căn cứ liên quan. `relevant_articles` là các cặp `law_id`–`article` đã đọc và xác minh.
3. Mọi `law_id` trong `relevant_articles` phải có mặt trong `relevant_docs` sau khi bỏ dấu `*`; không lặp cùng một cặp văn bản–Điều.
4. Nếu chỉ xác minh được văn bản nhưng chưa xác minh được Điều, có thể điền `relevant_docs` và để `relevant_articles` rỗng; ghi ID đó vào nhật ký cần rà soát.
5. Không dùng đáp án nguồn cũ làm nhãn vàng. Sau khi gán, người rà soát phải kiểm tra lại các trường hợp `*`, văn bản hết hiệu lực một phần và câu có nhiều ý hỏi.

Nếu có thể tạo tệp, lưu đầu ra thành `retrieval_labels_<KHOẢNG_ID>.json` và nhật ký rà soát thành một tệp riêng. Nhật ký không thay đổi schema bốn trường của tệp nhãn.
