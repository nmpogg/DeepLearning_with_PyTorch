Softmax Regression = Linear Model + Softmax + Cross-Entropy Loss

hồi quy chỉ cần 1 output, phân loại cần C output

vấn đề là output sẽ có thể là dạng chữ : cat, dog, fish, mã hóa thế nào cho hiệu quả

nếu 0-cat, 1-dog, 2-fish -> mô hình hiểu nhầm là 2-fish sẽ có weight cho fish mạnh hơn -> dễ phân loại thành fish

-> onehot encoding 
cat	[1, 0, 0]
chicken	[0, 1, 0]
dog	[0, 0, 1]

mỗi output sẽ là 1 hàm linear:
o₁ = w₁ᵀx + b₁   (cat)
o₂ = w₂ᵀx + b₂   (chicken)
o₃ = w₃ᵀx + b₃   (dog)

-> dạng matrix : o = XW + b

vấn đề: output của mỗi nút ouput có thể âm, không mang tính chất xác suất khi tổng != 1

-> dùng softmax để chuẩn hóa : softmax(oᵢ) = exp(oᵢ) / Σ exp(oⱼ) vì xác suất của 1 output tỉ lệ thuận với hàm e mũ của output đó

chỉ cần dùng softmax trong lúc huấn luyện, cụ thể là tính loss entropy để đảm bảo tổng xs = 1

vì hai hàm x và e mũ x là đơn điệu, x lớn -> e mũ x cũng lớn
việc chỉ cần chọn giá trị e mũ x lớn nhất tương đương với việc chọn x lớn nhất


loss function

chúng ta muốn xác suất dự đoán đúng là cao nhất -> tức là P(y | x) = Π p(yᵢ | xᵢ) càng cao càng tốt
nhưng việc tối đa hóa 1 tích là rất phức tạp -> quy về bài toán tối thiểu hóa log
-> Loss = - log(p(correct_class))

sau một số bước đạo hàm thì cuối cùng loss sẽ về dạng giống các bài toán hồi quy = y_hat - y
