# Machine Learning with Imbalanced Data

## 1. Giới thiệu các phương pháp phân lớp thiếu cân bằng

Các thuật toán học máy giả định rằng các tập dữ liệu có phân phối cân bằng. Điều này nghĩa là gần như có lượng quan sát tương tự từ tất cả các phân lớp mà chúng ta có trong tập dữ liệu. Tập dữ liệu mất cân bằng là những tập dữ liệu có nhiều thực thể hoặc quan sát của một phân lớp nhất định hơn là từ các lớp khác.

Vấn đề xảy ra khi một lớp có ít quan sát, thường gọi là lớp thiểu số (minority class), là rất khó để xây dựng các quy tắc hoặc ranh giới dự đoán để phân tách lớp nhỏ hoặc thiểu số khỏi các lớp khác. Do đó, các mẫu từ phân lớp thiểu số thường bị phân loại sai, bị phân loại thành bất kỳ mẫu nào khác trong nhóm đa số.

Một số bài toán tiêu biểu có dữ liệu mất cân bằng trong thực tế:

	- Phát hiện gian lận giao dịch thẻ tín dụng
	- Chuẩn đoán các bệnh trong y tế
	- Phát hiện thiết bị lỗi trong dây chuyền sản xuất 
	- Phát hiện tấn công mạng
	- Phát hiện tràn dầu sử dụng hình ảnh radar

- 

Một số thuật ngữ:

- **Class distribution**: Đề cập đến tỷ lệ các thực thể thuộc mỗi lớp

- **Imbalanced data-sets:** Tập dữ liệu mất cân bằng, có thể có một hoặc nhiều lớp thiểu số

- **Imbalance degree** (Mức độ mất cân bằng): Tỷ lệ kích thước mẫu của lớp thiểu số so với kích thước mẫu của lớp đa số. e.g: 1:100

  

## 2. Bản chất của các phân lớp mất cân bằng

Thực tế, một tập dữ liệu mất cân bằng không hoàn toàn có nghĩa là sẽ khó phân loại, như chúng ta sẽ thấy sau đây. Các yếu tố ảnh hưởng đến khả năng xác định các trường hợp hiếm gặp của tập phân loại trên thực tế là:

**Kích thước tập dữ liệu:** Dễ gặp vấn đề ở các tập dữ liệu có kích thước nhỏ hơn. Phân lớp mất cân bằng có thể không còn là vấn đề nếu dữ liệu này có kích thước đủ lớn.

**Khả năng phân tách lớp** thể hiện sự khác biệt rõ ràng của các quan sát từ các lớp khác nhau. Các lớp càng khác nhau thì càng dễ phân loại. Nếu các mẫu giữa các phân lớp chồng lên nhau thì thuật toán sẽ khó tìm ra các quy tắc hoặc ranh giới phân tách lớp này với lớp khác.

Một lớp riêng lẻ **gồm nhiều sub-cluster hoặc khái niệm như phân lớp không đồng nhất.** Có nhiều thành phần khác nhau tạo nên một lớp. Ngoài ra, các sub-cluster đó không phải lúc nào cũng chứa cùng một lượng ví dụ, và hiện tượng này được gọi là mất cân bằng trong lớp (within-class imbalance). Có một số sub-cluster trong lớp làm tăng độ phức tạp của lớp thiểu số, do đó khó tìm ra ranh giới phân tách lớp hơn.

## 3. Các phương pháp tổng quan làm việc với dữ liệu không cân bằng

Nhìn chung, có thể phân nhóm các phương pháp làm việc với dữ liệu mất cân bằng thành 3 nhóm chính:

**Phương pháp theo Data Level (cấp độ dữ liệu):** Cố gắng sửa đổi tập dữ liệu, thay đổi phân phối dữ liệu, để có nhiều quan sát hơn từ lớp thiểu số (oversampling) hoặc ít quan sát hơn từ lớp đa số (undersampling), do đó chúng ta có được một tỷ lệ tương tự từ mỗi lớp. Chúng ta có các phương pháp tạo dữ liệu mới trông giống lớp thiểu số như SMOTE cố gắng loại bỏ nhiễu hoặc những quan sát rất dễ phân loại để thuật toán có thể tập trung vào các trường hợp khó hơn.

**Phương pháp Cost-sensitive:** Áp dụng các hệ số khác nhau cho các lỗi mất mát khác nhau (tăng chi phí phân loại sai một trong các quan sát thiểu số lên cao hơn). Sau đó thuật toán sẽ cố gắng tối thiểu hóa những chi phí này theo các cách khác nhau. Như vậy, chúng ta đang thay đổi công thức tối thiểu hóa mà mô hình đang cố gắng khớp.

**Phương pháp Ensemble:** Chúng ta tạo nhiều bộ phân loại từ dữ liệu ban đầu, sau đó tổng hợp dự đoán và kết hợp các bộ phân loại giúp cải thiện khả năng tổng quát hóa (phần 5 của môn học).