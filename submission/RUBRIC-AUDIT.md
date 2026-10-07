# Đối chiếu rubric — 2026-10-08

Đã đọc toàn bộ rubric.md. Bảng này ghi bằng chứng, không dự đoán điểm của grader.

| Mục | Bằng chứng / trạng thái |
|---|---|
| 1.1 | Hai assert mask true; supervised_fraction=0.4149 <0.95. Có toàn bộ câu trả lời trong span, prompt bị che. |
| 1.2 | template_check.json giữ reasoning; report giải thích. |
| 1.3 | p95=98, max=101; max_length=256 theo sàn của hàm gợi ý; có lý do. |
| 1.4 | Adapter correct có đủ config/weights; số tensor khớp 32464896 tham số; runs.csv có loss/VRAM. |
| 2.1 | attn_only r=283; ngân sách lệch 0.0252%, dưới 5%; config và tensor khớp. |
| 2.2 | Cả bốn run ghi max_steps=actual_steps=30. |
| 2.3 | Vị trí với rank khớp ngân sách / LR chia 10 / base 4-bit; cùng split, model, mask, length và epoch budget. |
| 2.4–2.5 | Phân tích vị trí vs rank; thứ hạng theo target correct=attn_only>qlora>wrong_lr. |
| 3.1 | Baseline b=0.84>a=0; workflow đóng băng trước train; SHA prompt và byte checksum eval vẫn khớp. |
| 3.2 | Đủ bốn nhóm cho correct và baseline; contrasts theo thiết kế NB5 chỉ có target/format/latency. Report ghi rõ regression contrasts không đo. |
| 3.3 | FAIL do regression delta=-0.13556 dưới ngưỡng -0.02; điểm và phán quyết đã tính lại từ dự đoán thô. |
| 3.4 | 7 ví dụ trong report, gồm 2 ca thua thật ở regression. 50 target có 21 thắng/29 hòa/0 thua. Không thể đưa ra hai ca target thua không tồn tại; đã khai báo rõ nhóm và metric. |
| 4.1 | Có lựa chọn/lý do, mask, freeze, so sánh, verdict và điều học được. |
| 4.2 | Kết luận ≥150 từ (đã đếm 281 từ trước bổ sung bảng); có phân tích đánh đổi, giới hạn một seed và runtime. |
| 4.3 | Các điểm target/format/regression được tính lại; bảng loss/VRAM/step đối chiếu runs.csv; latency lấy số đo đã lưu. |
| 4.4 | Report có các bài học cụ thể. REFLECTION.md đã được soạn theo kết quả và quá trình chạy thực tế, kèm trong ZIP; người học cần đọc và chỉnh nếu chưa đúng trải nghiệm của mình. |
| B1–B5 | Chưa thực hiện các mục thưởng; không khai điểm thưởng. NB6 tùy chọn. |

Đã kiểm tra thêm: split seed42 khớp, không trùng nguyên văn input giữa corpus train và target eval; adapter không có tensor vision; manifest của ZIP khớp từng file. Kiểm tra không trùng input không thay thế kiểm chứng nhiễm dữ liệu ngữ nghĩa đầy đủ.

Gatekeeper không kiểm tra được tính cá nhân của phản tư, hoặc tự đảm bảo điểm rubric. Nội dung report cần được người học đọc và xác nhận. Không cần thay baseline, eval hoặc train lại chỉ vì phán quyết FAIL.
