# Reflection — Lab 21

Nhìn lại quá trình chạy NB1–NB5, điều đáng nhớ nhất là kết quả tốt trên ticket chưa đủ để kết luận model tốt hơn.

**1. Điều gì làm bạn ngạc nhiên nhất?**

Điều khiến tôi chú ý nhất là bản fine-tune đạt target 96,5%, cao hơn baseline đã tối ưu prompt là 84%, nhưng cuối cùng vẫn FAIL. Lý do là điểm regression giảm từ 79,11% xuống 65,56%. Khi xem câu hỏi “Một năm có bao nhiêu tháng?”, tôi thấy bản fine-tune trả về JSON phân loại ticket thay vì trả lời 12 tháng. Ví dụ này giúp tôi hiểu lỗi rõ hơn nhiều so với chỉ nhìn một con số regression giảm. Model làm tốt việc đã học, nhưng lại áp cách trả lời đó vào cả câu hỏi không liên quan.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Phần làm gián đoạn quá trình nhiều nhất là giữ và chuyển kết quả giữa các lần chạy Colab. Có lúc tôi chạy ô `wrong_lr` thì gặp lỗi không tìm thấy thư mục `/content/lab21_nb4_portable`, nên phải setup lại và khôi phục kết quả đã tải về. Tôi cũng phải kiểm tra nên giải nén ở đâu, file nào cần thay thế và ZIP cuối đã có đủ các run chưa. Tôi không có số đo để khẳng định đây là phần tốn nhiều phút nhất, nhưng đây là chỗ tôi phải quay lại xử lý nhiều lần. Qua đó tôi thấy việc lưu adapter và tải kết quả ngay sau mỗi run là một phần cần thiết của thí nghiệm, chứ không chỉ là bước dọn file cuối bài.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Điều tôi cần bỏ là cách suy luận “loss thấp hơn thì model tốt hơn”. Trong bài này, `attn_only` có train loss 0,5371, thấp hơn `correct` là 0,6264, nhưng cả hai cùng đạt target 96,5%. Chỉ nhìn loss thì tôi có thể chọn `attn_only` là run tốt nhất, trong khi điểm tác vụ chỉ cho thấy chúng hòa nhau. Tôi cũng không thể dùng việc target tăng để bỏ qua regression giảm. Sau lab, tôi thấy câu hỏi cần trả lời là model có đáp ứng toàn bộ yêu cầu đã đặt ra hay không, thay vì chỉ tìm một chỉ số đẹp để báo cáo.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để đọc code, kiểm tra cấu hình, chuẩn bị notebook Colab, nhập các file kết quả và đối chiếu report với artefact. Tôi trực tiếp chạy notebook trên Colab, tải ZIP và cung cấp đường dẫn kết quả. AI giúp giảm nhiều thao tác, nhưng tôi vẫn gặp những chỗ cần kiểm tra lại. Chẳng hạn, ban đầu tôi được hướng dẫn chạy tiếp ô NB4 mà hướng dẫn đó phụ thuộc vào việc runtime cũ vẫn còn; khi thư mục làm việc không còn, ô chạy bị lỗi và cần bước khôi phục. Lúc tạo notebook NB5, code định tính ban đầu cũng chỉ chọn các ca fine-tune có điểm thấp, chưa so từng ca với baseline (b). Sau khi bổ sung phép so sánh, kết quả mới cho thấy target không có ca thua và các ca thua thật nằm ở regression. Tôi thấy AI hữu ích nhất khi giúp kiểm chứng kết quả, nhưng lời hướng dẫn và kết luận của nó cũng cần được đối chiếu với trạng thái runtime và dữ liệu thật.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Tôi sẽ làm rõ model cần làm tốt việc gì và những khả năng nào không được giảm, rồi xây tập đánh giá trước. Sau đó tôi thử base model với prompt đủ rõ và một vài ví dụ, đo baseline và giữ lại kết quả trước khi train. Nếu prompt đã đáp ứng nhu cầu thì tôi chưa cần fine-tune. Nếu vẫn cần train, tôi sẽ kiểm tra loss mask và đặt tiêu chí chấp nhận từ đầu. Với kết quả của lab này, tôi chưa đưa adapter vào một trợ lý trả lời nhiều loại câu hỏi; tôi sẽ thử bổ sung dữ liệu ngoài miền trong một thí nghiệm mới rồi đánh giá lại, thay vì sửa tiêu chí để bản hiện tại được PASS.
