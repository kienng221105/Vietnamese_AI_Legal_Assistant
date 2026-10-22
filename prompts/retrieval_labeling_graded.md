# Prompt gán nhãn truy hồi pháp luật theo mức độ (lưu để dùng về sau)

Đính kèm tệp câu hỏi và một bản snapshot cố định của corpus, sau đó gửi nguyên nội dung từ phần **Prompt** trở xuống cho bên gán nhãn. Thay `[KHOẢNG ID CẦN GÁN]` bằng các ID của đợt làm việc.

## Prompt

Bạn là người gán nhãn cho bài toán **truy hồi văn bản pháp luật Việt Nam**. Nhiệm vụ của bạn là xác định văn bản và Điều liên quan đến từng câu hỏi. Không viết câu trả lời pháp lý.

### Đầu vào

1. Tệp câu hỏi JSON. Mỗi bản ghi có `id`, `question`, `topic`, `relevant_docs: []`, `relevant_articles: []`.
2. Một bản snapshot của corpus pháp luật. Corpus có `law_id`, `article`, `content`, URL nguồn và một phần metadata hiệu lực.
3. Các ID cần gán trong lượt này: **[KHOẢNG ID CẦN GÁN]**.

Giữ nguyên `id`, `question` và `topic` từ đầu vào. Mỗi câu dài có thể chứa nhiều ý hỏi và cần nhiều Điều ở nhiều văn bản.

### Cách tìm và kiểm tra căn cứ

1. Tìm ứng viên trong corpus. Bạn **được tra cứu thêm nguồn pháp luật chính thức** để tìm văn bản hoặc Điều mà corpus thiếu. Chỉ gán sau khi đã kiểm tra đúng số hiệu, số Điều và nội dung thực tế. Không suy đoán số Điều từ trí nhớ.
2. Xác định bối cảnh thời gian của câu hỏi. Nếu câu hỏi chỉ rõ thời điểm hoặc văn bản cũ, đánh giá theo bối cảnh đó. Nếu câu hỏi hỏi quy định hiện hành, kiểm tra hiệu lực. Không mặc định văn bản cũ vẫn áp dụng; cũng không tự loại văn bản hết hiệu lực khi câu hỏi hỏi về lịch sử. Hiệu lực và mức độ liên quan là hai thuộc tính khác nhau.
3. Nếu một phần câu hỏi mơ hồ về thời điểm, văn bản hoặc phạm vi pháp lý và chưa thể xác minh chắc chắn, dùng `needs_review`. Không ép phải tìm được căn cứ cho mọi câu.

### Mức độ liên quan của từng Điều đã xem

- `2` — **positive**: Điều cung cấp căn cứ trả lời trực tiếp ít nhất một ý hỏi. Câu nhiều ý có thể cần nhiều Điều mức `2`.
- `1` — **near positive**: Điều liên quan về bối cảnh nhưng bản thân nó không trả lời ý hỏi nào.
- `0` — **negative**: Đã đọc và xác nhận Điều không hữu ích cho câu hỏi, dù hệ thống có thể truy hồi nó vì từ ngữ hoặc chủ đề gần giống.

Điều **chưa được kiểm tra** là *unjudged*, không phải negative. Không cần tạo đủ số lượng negative cố định.

### Quy tắc đánh dấu thiếu corpus bằng `*`

Nếu một văn bản hoặc Điều mức `2` đã được xác minh trên nguồn pháp luật chính thức nhưng thiếu trong **snapshot corpus được giao**, thêm dấu `*` vào **cuối `law_id`**:

- Thiếu cả văn bản: dùng `law_id*`.
- Có văn bản nhưng không có record với trường `article` đúng Điều cần gán: dùng `law_id*` cho Điều đó. Nếu corpus chỉ có chunk toàn văn chứa chữ “Điều N” nhưng không có record cấp Điều tương ứng, vẫn xem là thiếu ở cấp Điều.
- Nếu cùng một văn bản có cả Điều đã có và Điều bị thiếu: trong `relevant_docs`, ghi `law_id*` **một lần**; trong `relevant_articles` và `judgments`, chỉ Điều bị thiếu dùng `law_id*`, Điều đã có giữ `law_id` không dấu `*`.
- Dấu `*` chỉ báo **khoảng thiếu của corpus**, không thuộc số hiệu văn bản chính thức và không biểu thị mức độ liên quan. Khi đối chiếu số hiệu giữa các trường, bỏ dấu `*` cuối chuỗi.
- Mỗi judgment mức `2` có `law_id*` bắt buộc có `source_url` dẫn đến nguồn chính thức đã dùng để kiểm chứng. Trong `reason`, nói rõ “thiếu văn bản” hoặc “thiếu Điều” trong corpus.
- Nếu chưa xác minh được văn bản/Điều từ nguồn đáng tin cậy, không tạo `law_id*`; dùng `needs_review`.

### Trạng thái của câu hỏi

- `labeled`: đã tìm và xác minh ít nhất một Điều hoặc văn bản mức `2`.
- `no_match`: đã kiểm tra đủ corpus và nguồn chính thức nhưng xác nhận không có căn cứ phù hợp để gán. Dùng trạng thái này thận trọng.
- `needs_review`: chưa đủ căn cứ để kết luận; ví dụ câu mơ hồ, nguồn không truy cập được, hiệu lực của Điều chưa rõ hoặc các nguồn mâu thuẫn.

Nếu chưa có căn cứ mức `2` được xác minh, để `relevant_docs` và `relevant_articles` là `[]`. Có thể ghi các ứng viên đã kiểm tra ở `judgments` mức `0` hoặc `1` nếu thực sự đã đọc chúng.

### Định dạng đầu ra

Xuất **một mảng JSON hợp lệ**. Mỗi ID được giao có đúng một bản ghi với cấu trúc:

```json
[
  {
    "id": 1,
    "question": "Giữ nguyên câu hỏi đầu vào",
    "topic": "Giữ nguyên chủ đề đầu vào",
    "annotation_status": "labeled",
    "relevant_docs": ["SO_HIEU_VAN_BAN"],
    "relevant_articles": [
      {
        "law_id": "SO_HIEU_VAN_BAN",
        "article": "Điều N"
      }
    ],
    "judgments": [
      {
        "law_id": "SO_HIEU_VAN_BAN",
        "article": "Điều N",
        "relevance": 2,
        "source_url": "URL đã kiểm tra",
        "reason": "Một câu ngắn nêu ý hỏi được Điều này hỗ trợ"
      }
    ]
  }
]
```

Đoạn JSON trên **chỉ minh họa cấu trúc**, không phải nhãn của câu hỏi thật. Không sao chép các giá trị giữ chỗ vào kết quả.

Quy tắc đầu ra:

1. `relevant_docs` chỉ gồm văn bản có ít nhất một judgment mức `2`. Mỗi số hiệu xuất hiện một lần; dùng dạng có `*` nếu có bất kỳ Điều positive nào của văn bản đó thiếu trong corpus.
2. `relevant_articles` chỉ gồm các judgment mức `2` đã xác định được Điều. `article` giữ đúng cách ghi trong văn bản, ví dụ `"Điều 5"`.
3. `judgments` chứa các cặp đã kiểm tra ở mức `2`, `1` hoặc `0`. Không ghi trùng một cặp `(law_id sau khi bỏ *, article)`.
4. Với văn bản đã xác định liên quan nhưng chưa xác định được Điều, chỉ ghi `relevant_docs` nếu có căn cứ mức `2` ở cấp văn bản; để `relevant_articles` rỗng và nêu lý do trong `judgments` với `article: null`.
5. Không điền câu trả lời, không sửa câu hỏi hoặc chủ đề, không bỏ sót ID. Không lấy đáp án cũ từ dataset câu hỏi làm căn cứ.
6. Chỉ xuất JSON, không kèm Markdown hoặc giải thích bên ngoài JSON. Nếu có thể ghi tệp, lưu thành `retrieval_labels_<KHOẢNG_ID>.json`.

### Tự kiểm tra trước khi bàn giao

- Số bản ghi đầu ra bằng số ID được giao; mỗi ID xuất hiện một lần.
- Mọi `relevant_articles` tương ứng với một judgment mức `2`.
- Mọi `law_id*` mức `2` đều có URL kiểm chứng và lý do thiếu corpus.
- Không có số hiệu, Điều hoặc URL do suy đoán.
- Các mục chưa kiểm tra không bị gán mặc định là negative.
