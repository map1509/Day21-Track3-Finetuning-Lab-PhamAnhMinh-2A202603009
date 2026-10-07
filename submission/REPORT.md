# Lab 21 — Evaluation Report

Repo: PhamAnhMinh-2A202603009. Cập nhật: 2026-10-08.

## Setup và bằng chứng NB1

Model `unsloth/Qwen3.5-4B`, tier T4, fp16. Giữ corpus mặc định 250 ticket CSKH tiếng Việt → JSON triage để tập trung kiểm chứng mask và so sánh với prompt đủ mạnh, thay vì đổi đồng thời model và dữ liệu. Split seed 42: 225 train / 25 validation. Eval đầy đủ: 50 target và 15 regression; không dùng EVAL_LIMIT. p95=98 token, max=101, chọn max_length=256 theo gợi ý của lab có sàn 256.

Mask assistant-only giám sát 39/94 token trên mẫu minh họa, supervised_fraction=0.4149. Hai assert answer_is_supervised và question_is_masked đều true. Template giữ nội dung reasoning thử nghiệm. Vùng loss giải mã:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Vùng loss gồm JSON và marker của template, không phải JSON thuần. NB3 dùng labels pre-tokenize từ cùng hàm mask NB1 và kiểm tra Trainer giữ nguyên labels; không dựa vào cờ assistant_only_loss.

## Baseline NB2 đóng băng

Đóng băng lúc `2026-10-07T09:45:23.125350+00:00`, trước train. SHA prompt `70b2dfe897ba340f` và checksum eval khớp repo. Prompt (b) có schema và ba ví dụ; hai ví dụ bổ sung lấy từ corpus train, không lấy nhãn eval, nhằm minh họa thêm intent và urgency. Sau train không đổi prompt hay eval. Điểm được chấm lại từ dự đoán thô; latency giữ số đo runtime.

| Run | Target | Regression | Format | Latency ms/mẫu |
|---|---:|---:|---:|---:|
| (a) base + naive | 0,0000 | 0,7911 | 0,0000 | 3152,6 |
| (b) base + optimized | 0,8400 | 0,7911 | 1,0000 | 1131,9 |
| (c) correct LoRA + naive | 0,9650 | 0,6556 | 1,0000 | 1382,7 |

(b) thắng (a) trên thước đo JSON triage. Điểm 0 của (a) không đồng nghĩa base không hiểu tiếng Việt; output không đạt định dạng yêu cầu khiến metric bằng 0.

## NB3/NB4 — đối chứng cùng ngân sách

Cả bốn run dùng 30/30 step, 2 epochs, batch hiệu dụng 16, cùng model, split, mask và max_length. correct dùng toàn bộ linear của text decoder, r=16, alpha=32, LR=1e-4. Model có 24 lớp linear attention và 8 lớp full attention.

| Run | r | Alpha | Tham số train | LR | Train loss | Giây | VRAM GB | Target NB5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| correct | 16 | 32 | 32464896 | 1e-4 | 0,6264 | 397,0 | 8,78 | 0,965 |
| attn_only | 283 | 566 | 32456704 | 1e-4 | 0,5371 | 290,3 | 8,79 | 0,965 |
| wrong_lr | 16 | 32 | 32464896 | 1e-5 | 1,5715 | 446,7 | 8,78 | 0,000 |
| qlora | 16 | 32 | 32464896 | 1e-4 | 0,7064 | 546,5 | 3,85 | 0,945 |

**Vị trí vs rank:** attn_only đổi vị trí sang q,v và tăng rank để khớp ngân sách, sai lệch chỉ 0,0252%. Nó hòa correct ở target dù train loss thấp hơn. Không thể kết luận rank cao hoặc attention-only tốt hơn từ loss; một tác vụ và một seed không chứng minh ưu thế tổng quát.

**Learning rate:** wrong_lr chỉ giảm LR mười lần, loss trung bình cuối cao hơn và target/format đều bằng 0. Log chi tiết nằm trong wrong_lr_training_log.json; loss cuối không tự mô tả toàn bộ đường học. Nếu chỉ thấy loss giảm mà kết luận học tốt, sẽ bỏ qua output không đạt JSON trên eval; điểm tác vụ mới cho thấy run thất bại ở ngân sách này.

**QLoRA:** run đổi base sang 4-bit, giảm peak VRAM 4,93 GB, khoảng 56,2%. Target giảm 2 điểm phần trăm so với correct; format vẫn 100%, latency tăng từ 1382,7 lên 1703,3 ms. Kết quả thể hiện đánh đổi bộ nhớ/chất lượng trong lần đo này, chưa đủ khẳng định mọi tác vụ của dòng model nên tránh QLoRA. NB5 đánh giá adapter QLoRA trên base 4-bit đúng như khi train. Contrasts chỉ được chấm target/format/latency, không có regression riêng.

Xếp hạng target: **correct = attn_only > qlora > wrong_lr**.

| Run NB5 | Target | Format | Latency ms/mẫu | Regression |
|---|---:|---:|---:|---|
| correct | 0,965 | 1,000 | 1382,7 | 0,6556 |
| attn_only | 0,965 | 1,000 | 842,7 | Không đo |
| wrong_lr | 0,000 | 0,000 | 4948,0 | Không đo |
| qlora | 0,945 | 1,000 | 1703,3 | Không đo |

NB5 đo đủ bốn nhóm cho bản correct so với baseline (b); bảng contrasts dùng target để xếp hạng theo thiết kế notebook, không suy diễn regression chưa đo.

## Phán quyết NB5: FAIL

Target tăng 0,125, tức 12,5 điểm phần trăm; regression giảm 0,13556, tức khoảng 13,56 điểm phần trăm. Ngưỡng hồi quy cho phép là 0,02 nên verdict.passed=false. Tôi giữ nguyên tiêu chí và không sửa tập eval để chuyển kết luận sang PASS. Format của correct đạt 100%, nhưng định dạng hợp lệ trên ticket không bù được khả năng trả lời câu hỏi phổ thông bị suy giảm. Hai ca regression bên dưới cho thấy adapter trả JSON triage khi cần trả lời kiến thức ngoài miền. Đây là biểu hiện phù hợp với chuyên môn hóa quá mức, chưa chứng minh cơ chế quên kiến thức bên trong model. Latency cũng cao hơn baseline prompt tối ưu trong lần đo. valid_trace_rate=0 không chứng minh reasoning bị phá hủy vì train trên JSON thuần và inference tắt thinking. Kết luận là adapter cải thiện tác vụ hẹp nhưng không đạt cổng chất lượng tổng thể, nên chưa nên triển khai như assistant dùng chung. Thêm replay có thể là thí nghiệm tiếp theo, chưa phải kết quả đã thực hiện.

## Ví dụ định tính

Trên 50 ticket: **21 ca thắng, 29 ca hòa, 0 ca thua** baseline (b). Không tạo hai ca target thua khi dữ liệu không có. Bổ sung hai ca thua thật ở nhóm regression, ghi rõ nhóm và metric. Output/nhãn đầy đủ có trong qualitative.json và regression_losses.json.

| Nhóm / index | Input rút gọn | Chuẩn | Baseline (b) | correct | Nhận xét |
|---|---|---|---|---|---|
| target / 6 | Balo, đổi size, hỏi cho biết thôi, lần cuối mua ở đây | thap, tieu_cuc | trung_binh, trung_tinh; 0,50 | Đúng 4 trường; 1,00 | FT thắng |
| target / 13 | Tai nghe, vỡ khi nhận, không vội, bực mình | san_pham_loi, thap | hoi_thong_tin, trung_binh; 0,50 | Đúng 4 trường; 1,00 | FT thắng |
| target / 3 | Bình giữ nhiệt, chưa thấy tiền, khi nào tiện | hoan_tien, thap | hoi_thong_tin, trung_binh; 0,50 | hoan_tien, trung_binh; 0,75 | Thắng, vẫn sai urgency |
| target / 5 | Nồi chiên, thiếu phụ kiện, khi nào tiện | san_pham_loi, thap | van_chuyen, trung_binh; 0,50 | san_pham_loi, trung_binh; 0,75 | Thắng, vẫn sai urgency |
| target / 12 | Áo khoác, bị lỗi, khi nào tiện | urgency thap | trung_binh; 0,75 | trung_binh; 0,75 | Hòa, cả hai sai |
| regression / 9 | Một năm có bao nhiêu tháng? | Keyword 12 | Có 12; recall 1,00 | JSON triage; recall 0,00 | FT thua |
| regression / 13 | TP.HCM trước đây có tên là gì? | Keyword Sài Gòn | Có Sài Gòn; recall 1,00 | JSON intent/confidence; recall 0,00 | FT thua |

Keyword recall không kiểm chứng toàn bộ nội dung câu trả lời base. Các ca trên minh họa đúng tiêu chí đã đóng băng, không thay thế đánh giá factuality đầy đủ.

## Kết luận và điều học được

Thí nghiệm cho thấy fine-tune có thể cải thiện rõ một tác vụ hẹp mà vẫn không đạt yêu cầu triển khai tổng thể. Tôi không chọn đưa adapter correct vào assistant dùng chung ở trạng thái hiện tại. Nó đạt 96,5% field accuracy và giảm phụ thuộc vào prompt dài, nhưng đánh đổi khả năng trả lời những câu hỏi phổ thông đã được đưa vào cổng hồi quy trước train. Baseline prompt tối ưu đạt 84% mà không cần huấn luyện và giữ điểm regression cao hơn, nên có bằng chứng phù hợp hơn nếu hệ thống cần xử lý cả câu hỏi ngoài miền. Với dịch vụ chỉ làm triage, kết quả target đáng nghiên cứu tiếp, nhưng báo cáo này không thay tiêu chí đã đóng băng để gọi đó là PASS. Bảng đối chứng cho thấy rank cao trên attention không tự tạo ưu thế target, và loss thấp không tự chứng minh model tốt hơn. QLoRA tiết kiệm bộ nhớ nhưng giảm target và tăng thời gian trong lần đo. Cần thêm seed, tập eval lớn hơn và phân tích sai số trước khi khái quát. Latency phụ thuộc runtime và batch, không phải cam kết hiệu năng sản phẩm.

Ba điều học được: đọc labels thực tế thay vì tin tên cờ; khớp cả step và tham số để đối chứng có nghĩa; tăng target không thay thế kiểm tra regression. Nếu có thêm hai giờ, tôi sẽ thiết kế thí nghiệm mới với replay dữ liệu ngoài miền và giữ nguyên nguyên tắc đóng băng baseline trước train. NB6 chưa thực hiện, không khai điểm thưởng merge/hot-swap.
