# NB1 — Bằng chứng thực chạy

Ngày: 2026-10-07. Model: `unsloth/Qwen3.5-4B` (tier T4), chạy tokenizer trên CPU, không tải trọng số. Giữ model mặc định để dùng cùng model ở baseline và fine-tune. Dataset: 250 ticket CSKH tiếng Việt mặc định, JSON triage bốn trường; giữ dataset để tập trung kiểm chứng mask và phép so sánh của lab.

- `results/mask_proof.json`: toàn bộ câu trả lời nằm trong loss; câu hỏi không nằm trong loss. Mẫu đầu có 39/94 token được giám sát, `supervised_fraction=0.4149`, dưới ngưỡng 0.95.
- Chuỗi giải mã vùng loss có `</think>`, JSON đầy đủ và `<|im_end|>`. Đây là vùng assistant có marker của template, không phải chuỗi JSON thuần. Không khẳng định mask bằng đúng các ký tự JSON.
- `results/template_check.json`: nội dung reasoning thử nghiệm được giữ lại (`ok=true`, `body_present=true`).
- `results/token_stats.json`: p50=93, p95=98, p99=100, max=101; chọn `max_length=256` theo `suggested_max_length` của lab (có sàn 256). Đã áp dụng cho tier T4 để các bước train sau dùng cùng độ dài. Toàn bộ corpus hiện tại nằm dưới giới hạn này.
- Split seed 42: 225 train / 25 validation trong `data/split/`.

Môi trường: Python 3.12.0, transformers 5.19.0. Artefact sinh ra bị gitignore theo thiết kế repo; cần giữ kèm bài nộp. Chưa train, chưa chạy NB2 và chưa có phán quyết LoRA. Không đổi base model; nếu đổi model phải chạy lại NB1 và `scripts/check_mask_agreement.py` trước khi train.
