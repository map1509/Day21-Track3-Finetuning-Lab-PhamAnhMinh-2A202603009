# NB2 — Chuẩn bị đo, chưa có kết quả

Giữ base model `unsloth/Qwen3.5-4B`, tier T4 và toàn bộ tập eval. Prompt (b) gồm schema, các giá trị hợp lệ và ba ví dụ. Hai ví dụ bổ sung lấy từ corpus huấn luyện mặc định (ticket nồi chiên không dầu và balo laptop), không lấy nhãn từ eval. Mục đích là cung cấp thêm ví dụ cho intent và urgency; hiệu quả thực tế chưa được đo.

NB2 lưu SHA prompt, nội dung prompt, checksum SHA-256 hai tập eval và thời gian UTC vào `results/baselines_frozen.json`. Nếu target(b) <= target(a), NB2 lưu số đo rồi dừng pipeline trước train. Chỉ được cải thiện prompt và đo lại khi chưa train. Không suy đoán hoặc tự điền điểm baseline.

Môi trường local hiện chưa có PyTorch; GPU xác nhận qua registry là AMD Radeon 680M. Phiên làm việc không có công cụ truy cập runtime Colab. Vì vậy chưa chạy inference NB2, chưa tạo `baselines_frozen.json` và chưa chứng minh (b) > (a).

## Cách chạy đã đóng gói sẵn

Mở Colab → File → Upload notebook → chọn `colab/NB2_LOCAL_READY.ipynb` → chọn T4 GPU → Runtime → Run all. Notebook chứa source local và dữ liệu, không chứa `.env` hay credential, không cần commit/push. Nó cài dependencies, chạy đầy đủ NB1 + NB2 và tải `nb1_nb2_results.zip`, trong đó có dự đoán thô và bảng `NB2-MEASURED.md` sinh từ số đo thật. Nếu (b) <= (a), chạy ô cuối thủ công để tải bằng chứng rồi dừng trước train. Notebook này được sinh bằng `python scripts/build_nb2_portable.py`; sinh lại sau mỗi thay đổi source trước khi đo.

Để đo trên Colab T4, dùng phiên bản repo có các thay đổi local này: upload notebook và các file đã sửa vào checkout trong runtime, hoặc commit/push rồi cấu hình clone repo bài làm. Bootstrap hiện mặc định clone upstream, nên mở URL upstream không tự mang theo thay đổi local.

Trong checkout đúng phiên bản, chạy:

```python
import os, subprocess, sys
os.environ["COMPUTE_TIER"] = "T4"
os.environ.pop("EVAL_LIMIT", None)
subprocess.run([sys.executable, "scripts/colab_run.py", "nb1", "nb2"], check=True)
```

Kiểm tra model, đủ mẫu, checksum và target(b) > target(a) trong artefact trước khi chạy NB3. Nếu prompt, tokenizer hoặc model thay đổi thì chạy lại NB1 trước train. Tải `results/` về để giữ bằng chứng khi runtime kết thúc.
