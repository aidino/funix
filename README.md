# Assignment 2

- Môn học: Giới thiệu về Machine Learning - MLP301x
- Tên dự án: Phân tích cảm xúc và ví dụ về phân loại hình ảnh
- Download các nguồn tài liệu cần cho assignment [tại đây](https://drive.google.com/drive/folders/1uslCMtC2_9eCXgMdkxRXCZ4ICjygArSn).
- Các nguồn hữu ích
    - [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    - [Feed forward neural network](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)
    - [Imdb reviews data link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
    - [Dog-cat dataset](https://www.kaggle.com/c/dogs-vs-cats)
    - [Read an image with opencv](https://pythonexamples.org/python-opencv-read-image-cv2-imread/)
    - [Resize an image with opencv](https://pythonexamples.org/python-opencv-cv2-resize-image/#:~:text=To%20resize%20an%20image%20in,not%2C%20based%20on%20the%20requirement.)
    - [Python examples](https://pythonexamples.org/)
---

**Tiêu chí chức năng:**

| #  | Criterion                                                                                     | Map to LO               | Specification                                                                                                                                                   |  Weight | Mandatory? | Grading type  |
| -- | --------------------------------------------------------------------------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ---------- | ------------- |
| 1  | Download tập dữ liệu Imdb movie reviews                                                       | MLP301x\_o1             | \- Giải nén và xếp đúng thư mục dữ liệu tới thư mục dữ liệu assignment                                                                                          | 100%    | Yes        | Pass/Not Pass |
| 2  | Load tập dữ liệu Imdb movie reviews                                                           | MLP301x\_o1, 7          | \- Phải dùng hàm load\_train\_test\_imdb\_data với đúng tham số.                                                                                                | 100%    | Yes        | Pass/Not Pass |
| 3  | In ra shape của tập dữ liệu Imdb movie reviews                                                | MLP301x\_o6, 7          | \- Nên dùng các hàm head và shape.                                                                                                                              | 100%    | No         | Pass/Not Pass |
| 4  | Biến đổi các bài đánh giá thành các vectơ                                                     | MLP301x\_o7             | \- Nên dùng hàm transform của vectorizer.                                                                                                                       | 100%    | Yes        | Pass/Not Pass |
| 5  | Đánh giá mô hình cảm xúc                                                                      | MLP301x\_o7, 27         | \- Nên dùng hàm accuracy\_score để đánh giá mô hình<br>\- Để huấn luyện mô hình, học viên chỉ cần chạy cell training. Assignment này không yêu cầu mã hóa mode. | 100%    | Yes        | Pass/Not Pass |
| 6  | Báo cáo độ chính xác tập kiểm tra đánh giá phim                                               | MLP301x\_o1, 7, 27      | \- Trả lời quiz về độ chính xác của tập kiểm tra.<br>\- Độ chính xác dự kiến: 83.68%                                                                            | 100%    | Yes        | Pass/Not Pass |
| 7  | Báo cáo độ chính xác để cải thiện mô hình cảm xúc với tf-idf                                  | MLP301x\_o1, 7, 27      | \- Trả lời quiz về độ chính xác của tập kiểm tra.<br>\- Độ chính xác dự kiến: 88.68%                                                                            | 100%    | Yes        | Pass/Not Pass |
| 8  | Download tập dữ liệu Dog vs Cats từ Kaggle và giải nén, sao chép tới đúng thư mục             | MLP301x\_o1, 7, 27      | \- https://www.kaggle.com/c/dogs-vs-cats/data<br>\- Học viên cần tạo một tài khoản Kaggle để download tập dữ liệu.                                              | 100%    | Yes        | Pass/Not Pass |
| 9  | Load tập dữ liệu Dog vs Cats                                                                  | MLP301x\_o1, 7, 27      | Cần dùng hàm get\_cat\_dog\_data với đúng tham số.                                                                                                              | 100%    | Yes        | Pass/Not Pass |
| 10 | In ra shape của các biến nhãn dán và hình ảnh                                                 | MLP301x\_o6, 7, 27      | Nên dùng hàm shape.                                                                                                                                             | 100%    | No         | Pass/Not Pass |
| 11 | Tiền xử lý hình ảnh                                                                           | MLP301x\_o22, 7, 27     | Đảm bảo rằng học viên tái tạo lại đúng shape như trong file giải pháp và tỷ lệ điểm ảnh bằng cách chia cho 255.                                                 | 100%    | Yes        | Pass/Not Pass |
| 12 | In ra shape của tập dữ liệu huấn luyện, kiểm tra và đếm số lượng chó, mèo cho mỗi tập dữ liệu | MLP301x\_o22, 7, 27, 33 | Nên dùng hàm shape và sum.                                                                                                                                      | 100%    | No         | Pass/Not Pass |
| 13 | Đánh giá mô hình NN                                                                           | MLP301x\_o22, 7, 25, 29 | \- Nên dùng hàm score để đánh giá mô hình.<br>\- Để huấn luyện mô hình, học viên chỉ cần chạy cell training. Assignment này không yêu cầu mã hóa mode.          | 100%    | Yes        | Pass/Not Pass |
| 14 | Báo cáo độ chính xác của tập dữ liệu hình ảnh, huấn luyện và kiểm tra.                        | MLP301x\_o1, 7, 27, 32  | \- Trả lời các quiz về độ chính xác.<br>\- Độ chính xác huấn luyện dự kiến: 0.96<br>\- Độ chính xác kiểm tra dự kiến: 0.63                                      | 100%    | Yes        | Pass/Not Pass |
| 15 | (Tùy chọn) Viết code để đánh giá mô hình cảm xúc dùng f1-score                                | MLP301x\_o1, 7, 27      | f1 dự kiến: 0.83                                                                                                                                                | 50%     | No         | Pass/Not Pass |
| 16 | (Tùy chọn) Viết code để đánh giá mô hình cảm xúc cải thiện dùng f1-score                      | MLP301x\_o1, 7, 27      | f1 dự kiến: 0.89                                                                                                                                                | 50%     | No         | Pass/Not Pass |
| 17 | (Tùy chọn) Viết code để đánh giá mô hình mạng nơ-ron hình ảnh dùng f1-score                   | MLP301x\_o1, 7, 27      | f1 dự kiến: 0.61                                                                                                                                                | 50%     | No         | Pass/Not Pass |

**Tiêu chí phi chức năng:**

| # | Criterion               | Map to LO       | Specification                                                        |  Weight | Mandatory? | Grading type |
| - | ----------------------- | --------------- | -------------------------------------------------------------------- | ------- | ---------- | ------------ |
| 1 | Kết quả mô hình tốt hơn | MLP301x\_o7, 28 | Thêm 1 điểm cho mỗi kết quả mô hình có độ chính xác hoặc f1 cao hơn. | 50%     | No         | Scale        |