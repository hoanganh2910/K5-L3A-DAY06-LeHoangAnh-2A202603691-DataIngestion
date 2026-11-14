# Báo cáo cá nhân — Day 6

**Sinh viên:** Le Hoang Anh — **MSSV:** 2A202603691  
**Ngày chạy:** 13/11/2026

## Tóm tắt

Đề tài là truy xuất bằng chứng từ Luật Tổ chức chính quyền địa phương số 72/2025/QH15 bằng embedding đa ngôn ngữ và PostgreSQL + pgvector. Thí nghiệm so sánh ba chiến lược chunking/filter trên cùng corpus, model, metric và benchmark. Đây là pipeline **retrieval**: nó trả về đoạn nguồn, không sinh câu trả lời bằng LLM.

## Dữ liệu và pipeline

- **Corpus:** văn bản pháp luật công khai từ [Cổng thông tin văn bản Chính phủ](https://vanban.chinhphu.vn/?pageid=27160&docid=214553), PDF 61 trang; 61/61 trang trích xuất được text, tổng 131.709 ký tự sau chuẩn hóa. Manifest ghi nguồn, ngày truy cập và căn cứ/quyền sử dụng; cần đối chiếu yêu cầu môn học trước khi nộp lại PDF.
- **Xử lý:** `pypdf` đọc từng trang; chuẩn hóa Unicode NFC và whitespace; giữ `document_id`, tên nguồn và số trang. Metadata `chapter` được nhận diện từ tiêu đề chương và truyền tiếp cho các trang cho tới chương kế tiếp.
- **Giới hạn trích xuất:** PDF có chỗ chèn khoảng trắng vào giữa từ (ví dụ `nhâ n dân`). Chuẩn hóa whitespace không sửa được lỗi đó; so khớp evidence nguyên văn vì vậy có thể bỏ lỡ đoạn đúng hoặc làm chunk khó đọc.
- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, chạy local bằng `sentence-transformers` 5.7.0 / PyTorch 2.15.0; vector 384 chiều. Không gửi corpus tới API, chi phí API ước tính 0 USD (không tính điện/năng lực máy).
- **Vector store:** PostgreSQL 16.16, pgvector 0.8.7, `psycopg` 3.3.8. Tìm kiếm cosine exact; không bật ANN index. Kết quả giữ ID, nguồn, trang và metadata.
- **Metadata:** `document_id`, `source`, `page`, cùng filter `chapter`. Các giá trị chương lấy từ cấu trúc đề mục PDF, đối chiếu với page ranges trong manifest.

## Cấu hình và benchmark

| Strategy | Chunking | Filter |
|---|---|---|
| A — `A_fixed_500_50` | Cửa sổ 500 ký tự, overlap 50 | Không |
| B — `B_paragraph_500` | Gom theo đoạn, giới hạn 500 ký tự | Không |
| C — `C_paragraph_500_filtered` | Giống B, dùng lại cùng chunks/vectors | Chỉ áp dụng `chapter` khi câu hỏi có filter |

Benchmark có 20 câu, gồm 15 dev và 5 test; 9 câu có metadata filter (4 dev, 5 test). Validator xác nhận evidence tồn tại trên gold page và filter phù hợp với page range. C dùng filter do câu hỏi cung cấp, không dùng đáp án/gold source để lọc.

Quy tắc chọn chiến lược chưa được ghi thành tiêu chí trước khi xem kết quả. Phân tích dưới đây xem Evidence Hit@3 trên dev là chỉ báo retrieval chính, nhưng kết luận chỉ mang tính **thăm dò**; test nhỏ đã được chạy và xem, không dùng để khẳng định hiệu quả tổng quát. Không điều chỉnh tham số sau lượt chạy này.

## Kết quả retrieval

Hit yêu cầu đúng nguồn/trang gold và chứa evidence quote sau khi chuẩn hóa Unicode, khoảng trắng và chữ hoa/thường. Hit@1 xét vị trí đầu; Evidence Hit@3 xét ba kết quả đầu.

| Strategy | Split | Hit@1 | Evidence Hit@3 | Số chunks |
|---|---:|---:|---:|---:|
| A — fixed | Dev | 3/15 | 7/15 | 316 |
| A — fixed | Test | 3/5 | 4/5 | 316 |
| B — paragraph | Dev | 3/15 | 3/15 | 358 |
| B — paragraph | Test | 3/5 | 3/5 | 358 |
| C — paragraph + filter | Dev | 3/15 | 3/15 | 358 |
| C — paragraph + filter | Test | 3/5 | 4/5 | 358 |

| Strategy | Ký tự trung bình/chunk | Chunking (giây) | Embed tài liệu (giây) | Upsert (giây) | Search 20 câu (giây) |
|---|---:|---:|---:|---:|---:|
| A — fixed | 457,1 | 0,0005 | 8,0676 | 0,2467 | 0,5295 |
| B — paragraph | 366,9 | 0,0105 | 7,9395 | 0,2570 | 0,5038 |
| C — paragraph + filter | 366,9 | 0,0105 | 0,0000* | 0,1950 | 0,4728 |

Embedding 20 câu hỏi dùng chung cho cả ba strategy mất 0,3556 giây. Tổng một lượt pipeline là 27,636 giây, gồm khởi tạo model, xác thực dữ liệu, xử lý và ghi kết quả. `*` C dùng lại vectors của B nên thời gian embedding tăng thêm bằng 0 trong lượt chạy; không nên hiểu đây là chi phí embedding độc lập bằng 0.

### Metadata và đánh giá thủ công

- Trên top-3 của 5 câu test, kiểm tra tự động thấy mỗi strategy có đủ `source`, `page`, `document_id`, `chapter` và page/chapter khớp page ranges: **15/15 bản ghi mỗi strategy**. Đây là kiểm tra cấu trúc/đối chiếu máy, **chưa phải audit thủ công**.
- Với C, cả 5 câu test có filter; **15/15 kết quả top-3** tuân thủ chapter được yêu cầu (5/5 câu).
- Coherence và grounding cần người chấm thực hiện, không suy ra từ điểm retrieval. Hiện chưa có nhãn thủ công; các ô điểm còn trống trong [manual audit template](../../starter/results/manual_audit_template.json). File có 10 mẫu coherence theo trang và 5 câu test grounding cho mỗi strategy.

## Phân tích và kết luận

- **A so với B:** trên dev, A có Evidence Hit@3 cao hơn (7/15 so với 3/15), còn Hit@1 bằng nhau (3/15). Trên test A đạt 4/5 và B 3/5, nhưng chỉ có 5 câu nên chênh lệch này không đủ chứng minh ưu thế tổng quát. A tạo 316 chunks nhưng tổng ký tự chunk là 144.459 do overlap; B tạo 358 chunks, tổng 131.338 ký tự.
- **B so với C:** chunks và vectors giống nhau; khác biệt là C lọc theo metadata khi có filter. Dev đều 3/15 Evidence Hit@3; test C 4/5, B 3/5. C cải thiện q20 (B trượt, C tìm thấy evidence với filter `chapter_VII`), nhưng cả B và C trượt q18 dù top-3 của C đều đúng chương V: filter thu hẹp phạm vi, không bảo đảm tìm đúng evidence.
- **Ví dụ lỗi:** q04 với A có gold page 2 nhưng top-3 là trang 34, 42, 31. q18 với B/C có gold page 42 nhưng top-3 là 46, 44, 45. Các trường hợp này cần xem lại chunking, chất lượng PDF extraction và khả năng semantic retrieval trên văn bản pháp lý.
- **Lựa chọn tạm thời:** A có Evidence Hit@3 cao nhất trên dev, nên là baseline retrieval tạm thời. Chưa kết luận “tốt nhất” trước khi chấm coherence/grounding và xem xét các lỗi. C vẫn hữu ích khi cần ràng buộc truy xuất theo chương.
- **Hạn chế:** chỉ một văn bản và 5 câu test; exact evidence matching nghiêm ngặt, không tính bằng chứng tương đương; PDF có lỗi khoảng trắng; thời gian phụ thuộc máy và model cache. Kết quả không đại diện cho corpus khác.

## Lệnh tái lập

Chạy từ thư mục `starter/`, với PostgreSQL đang hoạt động:

```powershell
python -m unittest discover -s tests -v
python .\scripts\validate_benchmark.py
python .\scripts\check_setup.py
python .\scripts\check_embeddings.py
python .\scripts\check_store.py
python -m src.run
```

Trace đầy đủ: [retrieval_traces.json](../../starter/results/retrieval_traces.json).  
Bảng metric và tài nguyên: [summary.json](../../starter/results/summary.json).  
Audit thủ công cần hoàn tất trước khi xem báo cáo này là bản nộp cuối.
