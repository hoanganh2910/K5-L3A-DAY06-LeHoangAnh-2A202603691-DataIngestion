# Starter · PostgreSQL + pgvector

Bộ khởi động **tùy chọn** cho lab cá nhân. Docker Compose tạo PostgreSQL + pgvector local. Một số phần trong `src/` vẫn có TODO; ingestion và chunking đã được hiện thực, nhưng pipeline tổng thể chưa chạy được cho đến khi hoàn thiện embedding, store, evaluation và orchestration. Vẫn có thể chọn Qdrant, Chroma hoặc vector store tương đương theo [GUIDE.md](../GUIDE.md).

## Cấu trúc

```text
starter/
├── docker-compose.yml     # PostgreSQL 16 + pgvector, chỉ mở cổng local
├── init.sql               # bật extension khi tạo database mới
├── requirements.txt       # psycopg, pypdf, python-dotenv và Sentence Transformers
├── .env.example           # cấu hình mẫu; copy thành .env
├── scripts/
│   ├── check_setup.py     # kiểm tra kết nối và extension
│   └── validate_benchmark.py # kiểm tra manifest, split và evidence
├── data/
│   ├── manifest.json      # nguồn và metadata corpus
│   ├── benchmark.json     # 20 câu hỏi dev/test có gold evidence
│   └── ...pdf             # corpus đã chọn
├── src/
│   ├── models.py          # cấu trúc dữ liệu mẫu
│   ├── ingestion.py       # đọc PDF, chuẩn hóa và gắn metadata chương
│   ├── chunking.py        # fixed-size và theo đoạn
│   ├── embeddings.py      # model embedding đa ngôn ngữ chạy local
│   ├── store.py           # pgvector: upsert, tìm kiếm cosine và metadata filter
│   ├── evaluation.py      # TODO: chấm truy xuất, lưu trace
│   └── run.py             # TODO: kết nối các bước thành pipeline
└── results/               # lưu kết quả của bạn
```

## 1. Cài và kiểm tra database

Cần Docker Desktop hoặc Docker Engine + Compose, Python **3.11–3.14**, và mạng để tải image/dependencies lần đầu. Các lệnh sau chạy từ thư mục `starter/`:

```bash
cd starter
python3 -m venv .venv
```

Kích hoạt `.venv`: macOS/Linux dùng `source .venv/bin/activate`; Windows PowerShell dùng `.venv\Scripts\Activate.ps1`. Sau đó:

```bash
python -m pip install -r requirements.txt
# macOS/Linux: cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

Kỳ vọng: hiện tên database, phiên bản PostgreSQL và pgvector, sau đó `Setup OK`. Đây **chỉ là kiểm tra hạ tầng**, không phải kết quả benchmark. Docker dùng cổng `127.0.0.1:55433` và mật khẩu cố định **chỉ dùng local cho lab**; nếu cổng bận, sửa đồng thời port trong `docker-compose.yml` và `DATABASE_URL` trong `.env`. `init.sql` chỉ chạy khi volume mới được tạo; nếu database sẵn có chưa bật extension, xem log Docker và kiểm tra cấu hình thay vì xóa volume chứa dữ liệu.

Dừng container nhưng giữ dữ liệu:

```bash
docker compose -p day6_student_starter stop
```

## 2. Chọn embedding model

CP4 sử dụng model local `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` qua Sentence Transformers; model tự báo số chiều khi khởi tạo và cùng một model tạo vector cho document lẫn query. Lần chạy đầu cần mạng để tải model vào cache Hugging Face; sau đó không cần API key. Cài dependency bằng `python -m pip install -r requirements.txt`. Có thể đổi `EMBEDDING_MODEL` trong `.env` nếu truyền giá trị đó khi khởi tạo embedder; giữ nguyên model và ghi tên/phiên bản/số chiều trong báo cáo. Không gửi corpus tới DeepSeek: API DeepSeek được dành riêng cho tác vụ chat/generation nếu dự án có thêm bước trả lời.

## 3. Hoàn thiện pipeline

1. Corpus mẫu của bài hiện nằm trong `data/`; kiểm tra nguồn, quyền sử dụng và metadata trong `data/manifest.json`.
2. `src/ingestion.py` đọc từng trang PDF, chuẩn hóa Unicode/whitespace, giữ số trang và nhận diện metadata chương. `src/chunking.py` cung cấp fixed-size theo ký tự (có overlap) và chunk theo đoạn (có tách đoạn quá dài); cả hai giữ document ID, nguồn, trang và metadata.
3. `src/embeddings.py` tạo vector thật bằng model đa ngôn ngữ chạy local. `src/store.py` lưu chunk/vector bền vững trong PostgreSQL + pgvector, tìm kiếm theo cosine, lọc metadata bằng JSONB và thay thế chunk cũ khi nạp lại tài liệu. Sau khi Docker/PostgreSQL đang chạy và dependencies đã cài, kiểm tra CP3–CP5 và benchmark từ thư mục `starter/`:

   ```powershell
   python -m unittest discover -s tests
   python .\scripts\check_embeddings.py
   python .\scripts\check_store.py
   python .\scripts\validate_benchmark.py
   ```

   `check_store.py` dùng một strategy ngẫu nhiên và xóa riêng các hàng thử nghiệm đó khi hoàn tất; các strategy khác không bị ảnh hưởng. Benchmark hiện có 20 câu hỏi, 15 dev / 5 test, cùng gold evidence và filter metadata. Lần embedding đầu tiên có thể tải model vài phút.
4. Hoàn thiện các hàm TODO còn lại trong `src/`, lưu ID tài liệu, nguồn, trang/vị trí và ít nhất một trường filter có ích.
5. Xây ít nhất ba cấu hình; ví dụ A fixed-size, B theo đoạn, C như B nhưng có filter. Cần cùng corpus, model và benchmark; nạp lại không nhân đôi và xử lý chunk cũ khi dữ liệu đổi.
6. Lưu trace top-3 và báo cáo đủ năm tiêu chí theo [BENCHMARK.md](../BENCHMARK.md).
7. Sau khi hoàn thiện pipeline, chạy `python -m src.run` từ thư mục `starter/`. Hiện tại lệnh này chưa chạy được vì evaluation và orchestration còn TODO.

Lưu code, cấu hình, kết quả và lệnh chạy để người chấm tái lập; không commit key, dữ liệu mật hoặc dữ liệu không có quyền chia sẻ.
