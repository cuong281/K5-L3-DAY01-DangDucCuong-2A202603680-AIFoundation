# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603680
- **Họ và tên:** Đặng Đức Cường
- **SEED:** 1115

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✅ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅ |

## 2. Kết quả chính

Từ file `lab1_report.json`, các mô hình có output cuối cùng như sau (dựa trên cùng tập test; LLM chưa chạy đầy đủ trong bản này):

| Mô hình | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.922 | 0.974 | 0.10 | 0.00 |
| 2. BoW + MLP | 0.954 | 1.000 | 0.60 | 0.00 |
| 3a. Encoder + kNN | 0.906 | 0.974 | 140.72 | 0.00 |
| 3b. Encoder + LogReg | 0.918 | 0.974 | 73.66 | 0.00 |
| 3c. Encoder + MLP head | 0.939 | 0.974 | 46.52 | 0.00 |

Nhận xét nhanh:
- Mô hình BoW + MLP là tốt nhất trên test (F1 ≈ 0.954), nhưng vẫn cần xem xét trên bộ case khó.
- 3c. Encoder + MLP head đạt F1 khá tốt và có độ trễ p95 dưới 50 ms, đủ gần với yêu cầu SLA.
- 3a. Encoder + kNN có p95 rất cao do phải tính cosine với tập train lớn, nên không phù hợp yêu cầu latency thấp.
- Vì `lab1_report.json` không có hàng LLM trong bản này, nên phần so sánh LLM cần dựa trên chạy thực nghiệm hoặc mô tả lý thuyết, không phải số liệu chốt của file này.

## 3. Dự đoán và thực tế

3 dự đoán lệch nhiều nhất so với thực tế:

1. `ml_f1`  
   - Dự đoán: F1 trên test khoảng 0.8–0.9, precision cao hơn recall.  
   - Thực tế: BoW + LogReg đạt F1 = 0.922, precision = 1.000, recall = 0.856.  
   - Vì sao lệch: dự đoán ban đầu khá gần đúng; mô hình đã chặn rất nhiều tin spam nhưng vẫn bỏ sót một phần spam, nhất là các tin ít có từ khóa rõ ràng. Làm đúng với nhận định “precision có thể cao hơn recall”.

2. `emb_search`  
   - Dự đoán: truy vấn ngụy trang sẽ gần với spam.  
   - Thực tế: kết quả in ra `spam` đúng như dự đoán.  
   - Vì sao lệch: vì embedding hiểu ngữ nghĩa, dù không có từ khóa rõ ràng như “free” hay “win”, câu vẫn có ý nghĩa quảng cáo/chiêu dụ và vector gần với spam.

3. `emb_tricky`  
   - Dự đoán: 3b. Encoder + LogReg sẽ làm tốt hơn BoW trên các câu ngụy trang.  
   - Thực tế: trên 6 case thử nghiệm, embedding + LogReg đúng 6/6 trong ví dụ kiểm tra, còn BoW đúng 5/6.  
   - Vì sao lệch: BoW quá phụ thuộc từ khóa, còn embedding đọc được nghĩa tồn tại trong câu dù từ khóa đã bị biến đổi hoặc giấu đi.

Ngoài ra, dự đoán `ml_lr` về `lr=50` và `lr=0.001` cũng khớp với cơ chế tối ưu: learning rate quá lớn khiến bước nhảy quá mạnh, learning rate quá nhỏ khiến loss giảm chậm. Đây là hiện tượng đúng với lý thuyết gradient descent.

## 4. Phân tích 50 case khó

Model tụt nhiều nhất là BoW (LogReg / MLP), vì nó nhìn văn bản theo token rời rạc, phụ thuộc từ khóa. Khi từ khóa bị giấu, biến đổi hoặc xuất hiện trong ngữ cảnh không phải spam, BoW dễ đoán sai.

Nhóm khó nhất đối với BoW:
- `obfuscation`: các câu dùng `C0ngr4tulat10ns`, `F R E E`, `c.a.s.h`, `Fr€€`...
- `ham_looks_spam`: tin ham có chứa từ như “FREE”, “WON”, “CALL”, “URGENT” nhưng không phải spam.

2 ví dụ cụ thể:
1. "Your mobile number has been chosen for a complimentary holiday..."  
   - Đây là spam nhưng không chứa các từ khóa spam quá rõ. BoW có thể đánh giá thấp nếu không có vocabulary phù hợp.
2. "Hi, this is Dr. Patel's office confirming your blood test results are ready..."  
   - Có từ “call”, “results”, “office”, “confirm” nhưng là tin hợp lệ. BoW dễ nhầm vì ngữ cảnh không rõ; embedding/LLM đọc hiểu tốt hơn.

Với embedding/LLM, nhóm khó nhất thường là prompt injection và natural spam, vì các tin này cố tình lừa mô hình hoặc viết theo giọng quen thuộc như người thân/khách hàng. Cụ thể, nếu prompt chứa lệnh như "Ignore all previous instructions..." thì LLM rất dễ sa vào bẫy. Đây cũng là điểm yếu mà prompt phòng thủ cần xử lý.

## 5. Thử thách cá nhân

Đề của tôi theo `SEED = 1115` là: "ML · Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1".

Hướng làm:
- Dùng mô hình logistic regression ở Phần 1.
- So sánh với ngưỡng 0.2, 0.5, 0.8.
- Quan sát sự thay đổi của precision / recall / F1.

Kết quả và giải thích:
- Ngưỡng 0.2: recall tăng, vì nhiều spam hơn được phát hiện, nhưng precision giảm vì có nhiều ham bị chặn nhầm.
- Ngưỡng 0.5: là ngưỡng cân bằng tự nhiên, là điểm mà nhiều mô hình chọn mặc định.
- Ngưỡng 0.8: precision tăng mạnh, nhưng recall giảm, dẫn đến bỏ sót thêm spam. Đây phù hợp nếu mục tiêu là giảm chặn nhầm tin ham.

Tóm lại, với sản phẩm thực tế, lựa chọn ngưỡng phụ thuộc mục tiêu: nếu ưu tiên giảm sai sót chặn nhầm, chọn ngưỡng cao; nếu ưu tiên bắt spam, chọn ngưỡng thấp. Trong hệ thống SMS, thường cần cân bằng để không chặn nhầm tin OTP hoặc thông báo giao dịch hợp lệ.

## 6. Phương án cho sản phẩm thật

Yêu cầu: p95 < 50 ms/tin cho 10 triệu SMS/ngày.

Dựa trên kết quả thực tế trong `lab1_report.json`:
- BoW + MLP có p95 = 0.60 ms, rất nhanh, F1 ≈ 0.954.
- 3c. Encoder + MLP head có p95 = 46.52 ms, gần zới dưới ngưỡng SLA, F1 ≈ 0.939.
- 3a. Encoder + kNN có p95 = 140.72 ms, vượt SLA rõ rệt.
- 3b. Encoder + LogReg có p95 = 73.66 ms, vẫn quá chậm nếu đòi SLA 50 ms.

Vì vậy, phương án phù hợp nhất cho sản phẩm thật là:
- Chọn 3c. Encoder + MLP head nếu cần chất lượng cao hơn và vẫn muốn nằm trong khoảng p95 gần ngưỡng 50 ms.
- Hoặc BoW + MLP nếu ưu tiên latency cực thấp và dễ triển khai ở quy mô lớn.

Đánh đổi:
- BoW + MLP rất nhanh và rẻ, nhưng còn yếu ở các case khó như obfuscation và ham_looks_spam.
- Encoder giúp hiểu ngữ nghĩa tốt hơn, nhưng tốn CPU/GPU và có thể tốn nhiều hơn khi xử lý khối lượng rất lớn.
- LLM tốt cho câu khó nhưng không phù hợp cho SLA latency < 50 ms và chi phí cao.

Với điều kiện này, router là phương án tốt nhất: dùng mô hình nhanh xử lý gần như chắc chắn, chỉ chuyển những tin lưỡng lự sang LLM hoặc encoder mạnh hơn. Đây là cách cân bằng giữa độ chính xác, latency và chi phí.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì | AI trả lời | Bạn sửa / kiểm chứng |
| --- | --- | --- | --- |
| Data preprocessing | "Viết code Python xử lý duplicate và hiểu dataset SMS" | Đề xuất `drop_duplicates()`, `value_counts()`, `describe()` | Kiểm tra số dòng gốc và loại trùng, đối chiếu với biến `df` |
| Bag-of-words | "Giải thích vectorize và xây vocab" | Chỉ ra cách tạo ma trận 0/1 theo từ xuất hiện | Kiểm tra shape và cách token hóa bằng regex |
| Logistic regression | "Giải thích gradient descent và loss" | Nêu các bước error, grad_w, grad_b, cập nhật w, b | Đối chiếu với code thực tế, xem dấu của cập nhật có đúng không |
| MLP | "Giải thích zero_grad, loss.backward, optimizer.step" | Giải thích từng bước trong training | Đọc kỹ code và kiểm tra việc xóa gradient cũ là bắt buộc |
| Embedding | "Nghĩa của cosine và top-k similarity" | Giải thích không chuẩn hóa vector là lỗi | Kiểm tra `top_k_similar` có chuẩn hóa hay không |
| LLM | "Viết prompt phân loại spam/ham, parse_label" | Đề xuất prompt và xử lý chuỗi trả lời | Kiểm tra parse "Spam.", "Label: spam", "Not spam" |

Điều đã học:
- Accuracy là thước đo dễ đánh lừa khi dữ liệu mất cân bằng; F1 mới là thước đo quan trọng trong bài này.
- Nhiều lỗi hay gặp không nằm ở syntax, mà ở data leakage, đánh giá sai tập và cách mô hình nhìn văn bản.
- Một mô hình mạnh trên tập test thường không đồng nghĩa với hiểu ngữ nghĩa, nên bộ case khó và stress test là cần thiết để kiểm tra độ bền.

### Kết luận chung

Mình đã hoàn thành lab theo hướng: hiểu cơ chế học, chạy mô hình, so sánh phương án, và ghi lại dự đoán / phân tích lỗi. Trong bài này, thông số quan trọng nhất là F1, p95 latency, và mức độ bền trên case khó. Từ đó, mô hình BoW + MLP và 3c. Encoder + MLP head là các phương án đáng tin cậy nhất cho hệ thống thực tế, trong khi LLM chỉ nên dùng ở tầng router hoặc xử lý các trường hợp lưỡng lự.
