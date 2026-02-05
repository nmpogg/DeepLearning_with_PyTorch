việc khởi tạo các weight không ngẫu nhiên như ta tưởng -> nó đóng vai trò quan trọng
    + duy trì tính ốn định số học
    + liên quan đến việc chọn hàm kích hoạt
    + chọn sai có thể -> Vanishing and Exploding Gradients

**Vanishing Gradients**
hàm sigmoid gặp vấn đề khi lan truyền ngược qua nhiều lớp, trừ khi input ở vị trí xung quanh giá trị 0, đầu vào của nhiều hàm sigmoid gần = 0, đạo hàm của tích tổng thể có thể biến mất ( vì max của đạo hàm cũng chỉ là 0.25, nếu 0.25^15 thì cũng vanishing)

-> dùng ReLU nhung ít khả thi về mặt thần kinh hơn vì nếu x > 0, đạo hàm luôn = 1 -> 1 ^ n = 1

**Exploding Gradients**
xảy ra khi thuật toán GD không có cơ hội hội tụ do cách khởi tạo -> value lớn giữ nguyên nhân với nhau nhiều lần -> Exploding


**Symmetry - Sự đối xứng**
nếu khởi tạo trọng số giống nhau tất cả các quá trình từ forward đến backwward đều cho kết quả giống nhau -> không học được gì
-> cần random init

nhưng kĩ thuật dropout có thể xử lý việc này: cho một số nơ ron = 0 trong quá trình huấn luyện

**Một số pp khởi tạo**
- mặc định dùng random -> tốt trong thực tế với các bài toán trung bình
- khởi tạo Xavier: 
+ Dành cho: Hàm kích hoạt Sigmoid hoặc Tanh
+ Tư duy:
Nếu dùng Sigmoid/Tanh, vùng hoạt động tốt nhất là vùng tuyến tính gần 0 (Goldilocks zone).
Xavier tính toán phương sai dựa trên số lượng kết nối đầu vào ($fan_{in}$) và đầu ra ($fan_{out}$) để giữ tín hiệu ổn định.
+ Công thức (Phân phối chuẩn):
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{fan_{in} + fan_{out}}}\right)$$
+ Vấn đề: Phương pháp này giả định hàm kích hoạt là tuyến tính. Nó không hoạt động tốt với ReLU vì ReLU loại bỏ một nửa tín hiệu (phần âm).

**Khái quát hóa trong học sâu**
1. Nghịch lý về Tổng quát hóa (The Generalization Paradox)
- Thực tế: Các mạng nơ-ron sâu (Deep Neural Networks) thường có số lượng tham số khổng lồ (over-parameterized), lớn hơn nhiều so với số lượng mẫu dữ liệu. Chúng có khả năng ghi nhớ và khớp hoàn hảo (interpolate) tập dữ liệu huấn luyện (training error = 0).
- Mâu thuẫn với lý thuyết cổ điển: Theo lý thuyết thống kê truyền thống (như độ phức tạp VC), một mô hình quá phức tạp như vậy sẽ bị quá khớp (overfitting) nặng nề và thất bại trên tập kiểm thử. Tuy nhiên, trong thực tế, Deep Learning vẫn hoạt động rất tốt trên dữ liệu mới.
- Kết luận: Lý thuyết hiện tại chưa giải thích đầy đủ tại sao Deep Learning lại tổng quát hóa tốt đến vậy..
2. Sự thay đổi về quan niệm "Độ phức tạp" (Complexity)
- Góc nhìn cổ điển: Tăng độ phức tạp mô hình $\rightarrow$ Giảm Training Loss nhưng tăng Test Loss (Overfitting). Cần phải giảm bớt độ phức tạp.
- Góc nhìn Deep Learning: Đôi khi, việc làm cho mô hình phức tạp hơn nữa (tăng số lớp, số nơ-ron, train lâu hơn) lại giúp giảm lỗi tổng quát hóa.
- Hiện tượng "Double Descent": Khi tăng độ phức tạp, ban đầu lỗi kiểm thử tăng lên (overfitting), nhưng khi tiếp tục tăng độ phức tạp vượt qua một ngưỡng nào đó, lỗi lại bắt đầu giảm xuống.
3. Deep Learning dưới góc nhìn Phi tham số (Nonparametrics)
- Mặc dù mạng nơ-ron có tham số (weights), nhưng hành vi của chúng giống các mô hình phi tham số (như k-Nearest Neighbors - kNN).
- Giống như kNN ghi nhớ dữ liệu nhưng vẫn dự đoán được dựa trên khoảng cách, mạng nơ-ron khi đủ lớn (độ rộng vô hạn) sẽ hoạt động tương tự như các phương pháp Kernel (Neural Tangent Kernel). Điều này giúp giải thích phần nào khả năng nội suy dữ liệu của chúng.
4. Kỹ thuật Early Stopping (Dừng sớm)
Đây là một kỹ thuật quan trọng dựa trên hành vi học tập của mạng nơ-ron:
- Cơ chế: Mạng nơ-ron có xu hướng học các "mẫu sạch" (clean patterns) và cấu trúc chung của dữ liệu trước, sau đó mới học đến nhiễu (noise) hoặc các nhãn sai.
- Ứng dụng: Bằng cách dừng huấn luyện khi lỗi trên tập validation bắt đầu đi ngang hoặc tăng nhẹ (patience criterion), ta ngăn mô hình học thuộc lòng nhiễu, từ đó cải thiện khả năng tổng quát hóa.
- Lợi ích phụ: Tiết kiệm thời gian và chi phí tính toán.
5. Vai trò của Regularization (Điều chuẩn)
- Weight Decay (L2 Regularization): Vẫn được sử dụng phổ biến. Tuy nhiên, trong Deep Learning, nó không hẳn hoạt động bằng cách giới hạn khả năng của mô hình (vì mô hình vẫn có thể khớp 100% dữ liệu train), mà có thể nó giúp định hình một Inductive Bias (thiên kiến quy nạp) phù hợp, giúp mô hình tìm ra giải pháp tốt hơn trong không gian tham số.
- Inductive Bias: Vì không có thuật toán nào tốt nhất cho mọi dữ liệu (định lý "No free lunch"), sự thành công của Deep Learning phụ thuộc vào việc kiến trúc mạng (CNN, RNN, MLP...) có các giả định (bias) phù hợp với cấu trúc thực tế của dữ liệu hay không.

**Tóm lại: Trong Deep Learning, mục tiêu tối thượng không phải là tối ưu hóa hàm mất mát (Optimization) xuống 0 (vì điều này quá dễ), mà là làm sao để mô hình hoạt động tốt trên dữ liệu chưa từng thấy (Generalization). Các kỹ thuật như Early Stopping, Regularization, và việc lựa chọn kiến trúc đóng vai trò quan trọng hơn là việc cố gắng giảm thiểu độ phức tạp của mô hình theo cách truyền thống.**
