# Deep learning with Computer vision

Thị giác máy tính (computer vision) đề cập đến toàn bộ quá trình mô phỏng tầm nhìn của con người trong một bộ máy phi sinh học. Điều này bao gồm việc chụp ảnh ban đầu, phát hiện và nhận dạng đối tượng, nhận biết bối cảnh tạm thời giữa các cảnh và phát triển sự hiểu biết ở mức độ cao về những gì đang xảy ra trong khoảng thời gian thích hợp.

Mặc dù vẫn còn những trở ngại đáng kể trong con đường phát triển của thị giác máy tính đến “cấp độ con người”, các hệ thống Deep Learning đã đạt được tiến bộ đáng kể trong việc xử lý một số nhiệm vụ phụ có liên quan. Lý do cho sự thành công này một phần dựa trên sự phát triển của cấu trúc CNN - convolution neural network.

## Fundamental Convolutional Neural Network (CNN)

## Các kiến trúc CNN cơ bản

### Classic Network

![pic1](deeplearning_cv_tensorflow/pic1.png)

![pic2](deeplearning_cv_tensorflow/pic2.png)

*Note:* Kí hiệu `3x3 same` là một same convolution, nghĩa là 1 lớp convolution có padding sao cho output có size giống với input. Kích thước của kenel này ;à 3x3

(Không có chữ `same` thì thường là `valid` convolution, lớp này không có pading, và size của lớp output không giống size của lớp input)



![pic3](deeplearning_cv_tensorflow/pic3.png)



### Kiến trúc ResNets - Residual Networks

Những mạng nơ-ron rất rất sâu khá khó để huấn luyện vì sẽ gặp vấn đề vanishing/exploding gradient.

Ở phần này, chúng ta sẽ tìm hiểu về skip connection, giúp chúng ta lấy activation từ một layer và đưa nó vào một layer còn sâu hơn nhiều trong NN, điều này cho phép chúng ta huấn luyện NN rộng hơn với các layer lớn hơn 100.

#### Residual block - Khối dư

![pic4](deeplearning_cv_tensorflow/pic4.png)



#### Residual Network

![pic5](deeplearning_cv_tensorflow/pic5.png)

Các mạng này có thể đi sâu hơn mà không làm ảnh hưởng đến chất lượng. Trong các mạng NN chuẩn - mạng đơn giản - theo lý thuyết, nếu chúng ta đi sâu hơn thì sẽ tìm được giải pháp tốt hơn cho vấn đề của mình, nhưng khi gặp vấn đề vanishing/exploding gradient, chất lượng của mạng bị ảnh hưởng khi nó đi sâu hơn. Nhờ Residual Network, chúng ta có thể đi sâu hơn như mong muốn.

#### Tại sao ResNets lại hiệu quả

![pic6](deeplearning_cv_tensorflow/pic6.png)

#### Networks in Networks and 1x1 Convolutions



![pic7](deeplearning_cv_tensorflow/pic7.png)

![pic8](deeplearning_cv_tensorflow/pic8.png)

### Inception Network

#### Inception Network Motivation

Khi thiết kế CNN, chúng ta phải tự mình quyết định tất cả các layer. Chúng ta sẽ chọn 3 x 3 Conv hoặc 5 x 5 Conv hay có thể là max pooling layer. Chúng ta có rất nhiều lựa chọn.

Inception cho chúng ta biết điều gì? Tại sao không sử dụng tất cả chúng cùng một lúc?

Mô-đun Inception, phiên bản naive:

![pic9](deeplearning_cv_tensorflow/pic9.png)

Gợi ý rằng max-pool ở đây là như nhau. Đầu vào cho mô-đun inception là `28 x 28 x 192` và đầu ra là `28 x 28 x 256`. Chúng ta đã thực hiện tất cả các Conv và pool chúng ta có thể sẽ muốn và để NN tìm hiểu và quyết định xem nó muốn sử dụng cái nào nhất.

Vấn đề về chi phí tính toán trong mô hình Inception :

- - Chúng ta chỉ tập trung vào `5 x 5` Conv mà chúng ta đã thực hiện ở ví dụ trước.
  - Có 32 filter `5 x 5` giống nhau và đầu vào là `28 x 28 x 192`.
  - Đầu ra phải là `28 x 28 x 32`.

=> Tổng số phép nhân cần thiết ở đây là:

$\text{Số lượng đầu ra }* \text{Kích thước filter} * \text{Kích thước filter} * \text{Kích thước đầu vào} \Leftrightarrow 28 * 28 * 32 * 5 * 5 * 192 = 120 M $

Phép nhân $120 M$ vẫn còn là một vấn đề với máy tính hiện đại ngày nay. Nhờ sử dụng `1 x 1` convolution, chúng ta có thể giảm $120 M$  xuống chỉ còn $12 M$. Hãy xem nó hoạt động thế nào.

Sử dụng `1 X 1` convolution để giảm chi phí tính toán. Kiến trúc mới là:

- - Shape $X_0$ là `(28, 28, 192)`
  - Sau đó, chúng ta áp dụng 16  (1 x 1 Convolution)
  - Điều đó tạo ra $X_1$ có shape `(28, 28, 16)` ( gợi ý, chúng ta đã giảm kích thước ở đây).
  - Sau đó, áp dụng 32 (5 x 5 Convolution)
  - Điều đó tạo ra $X_2$ có shape (28, 28, 32)

Bây giờ, hãy tính số phép nhân:

- - Đối với Conv đầu tiên: $28 * 28 * 16 * 1 * 1 * 192 = 2,5 M$
  - Đối với Conv thứ hai: $28 * 28 * 32 * 5 * 5 * 16 = 10 M$

Vậy, tổng là khoảng 12.5 Mil, khá tốt so với 120 Mil. 1 x 1 Conv ở đây được gọi là Bottleneck BN. Như vậy 1 x 1 Conv sẽ không ảnh hưởng đến chất lượng.

Mô-đun Inception, phiên bản giảm kích thước:

![pic10](deeplearning_cv_tensorflow/pic10.png)

Ví dụ về mô hình inception trong Keras:

![pic11](deeplearning_cv_tensorflow/pic11.png)

#### Inception Network

Mạng inception gồm các block nối với nhau của mô-đun Inception. Cái tên inception được lấy từ một hình ảnh meme từ bộ phim Inception.

Đây là mô hình đầy đủ:

<img src="deeplearning_cv_tensorflow/pic12.png" alt="pic12" style="zoom:100%;" />

Đôi khi, `Max-Pool` block được sử dụng trước mô-đun inception để giảm kích thước của các đầu vào.

Có 3 nhánh Sofmax ở các vị trí khác nhau để đẩy mạng về phía mục tiêu của nó, đảm bảo các đặc trưng trung gian đủ tốt để mạng học được; và hóa ra softmax0 và sofmax1 mang có tác dụng điều chuẩn.

Kể từ khi phát triển mô-đun Inception, các tác giả và những người khác đã xây dựng các phiên bản khác của mạng này, như inception v2, v3 và v4. Ngoài ra, có một mạng đã sử dụng cả mô-đun inception và ResNet.

### Kiến trúc nâng cao

#### MobileNet

> MobileNet là một kiến trúc mạng mới thích hợp với các môi trường máy tính yếu hơn như các Mobile Phone

Khái niệm mới:

- Depthwise-separable convolutions: Tích chập tách biệt chiều sâu

![pic13](deeplearning_cv_tensorflow/pic13.png)



**Depthwise-separable convolution có 2 steps.**

![pic14](deeplearning_cv_tensorflow/pic14.png)

#### MobileNet Architecture

![pic15](deeplearning_cv_tensorflow/pic15.png)

#### EfficientNet

Với các MobileNet, làm sao để chúng ta thay đổi độ sâu của mạng tuỳ thuộc với phần cứng của devices?

Ví dụ như với những thiết bị có phần cứng yếu hơn, chúng ta sẽ phải đánh đổi độ sâu của mạng, mạng sẽ cho ra kết quả có độ chính xác thấp hơn nhưng tốc độ tính toán sẽ nhanh hơn.

Còn với những thiết bị mạnh hơn, chúng ta có thể tăng độ sâu của mạng, tăng độ phức tạp của mạng để có thể đưa ra được những kết quả chính xác hơn.

**EfficientNet** sẽ giúp chúng ta dynamic chuyện đấy!!!

Như chúng ta thấy trong các mạng CNN, thì có 3 yếu tố mà liên quan đến chi phí tính toán (compute cost), đó là: "Độ phân giải ảnh - **r**-resolution", "Độ sâu của mạng - **d** - depth", "Width của các layer trong mạng - **w**"

![pic16](deeplearning_cv_tensorflow/pic16.png)

![pic17](deeplearning_cv_tensorflow/pic17.png)



Vấn đề bây giờ là tìm một triển khai, cân bằng các giá trị **r-d-w**

*Nên tìm opensource triển khai EfficientNet*

## Những lời khuyên thiết thực trong việc xây dựng CNN

### Sử dụng Opensource

Chúng ta đã tìm hiểu rất nhiều kiến trúc NN và ConvNet.

Nhiều mạng nơ-ron trong số đó rất khó nhân rộng vì có một số chi tiết có thể không được trình bày trên tài liệu. Một số lý do khác như:

- - Learning decay.
  - Điều chỉnh tham số.

Nhiều nhà nghiên cứu học sâu đang mở nguồn cung cấp code của họ lên Internet trên các trang như Github.

Nếu bạn nhìn thấy một bài nghiên cứu và bạn muốn xây dựng dựa trên đó, trước tiên bạn nên tìm kiếm một triển khai mã nguồn mở cho tài liệu này.

Một số lợi ích của việc này là chúng ta có thể download triển khai mạng cùng với các tham số/trọng số của nó. Tác giả có thể đã sử dụng nhiều GPU và dành vài tuần để đạt được kết quả này, và kết quả đó xuất hiện ngay sau khi download.

### Transfer Learning

Nếu chúng ta đang sử dụng kiến trúc NN đã được huấn luyện trước đó, có thể sử dụng các tham số trọng số đã huấn luyện trước này thay vì khởi tạo ngẫu nhiên để giải bài toán. Nó giúp chúng tăng chất lượng của NN.

Các mô hình huấn luyện trước có thể đã được huấn luyện trên các tập dữ liệu lớn như ImageNet, Ms COCO hoặc pascal và tốn rất nhiều thời gian để tìm hiểu các tham số trọng số đó với các siêu tham số được tối ưu hóa. Điều này giúp chúng ta tiết kiệm rất nhiều thời gian.

**Hãy xem ví dụ sau:**

Giả sử chúng ta có bài toán phân loại mèo, trong đó có 3 lớp Tigger, Misty và neither. Chúng ta không có nhiều dữ liệu để huấn luyện NN trên những hình ảnh này.

Andrew khuyên rằng nên truy cập trực tuyến và download một NN tốt cùng với các trọng số của nó, loại bỏ layer kích hoạt softmax và đặt một layer của riêng mình và làm cho mạng chỉ học layer mới trong khi các trọng số layer khác bị cố định/đóng băng. Framework có các tùy chọn để đóng băng các tham số trong một số layer bằng cách sử dụng **trainable = 0** hoặc **freeze = 0**

Một trong những thủ thuật giúp tăng tốc quá trình huấn luyện là chạy NN đã huấn luyện trước mà không có layer softmax cuối và lấy một biểu diễn trung gian của hình ảnh rồi lưu chúng vào ổ đĩa. Sau đó sử dụng các biểu diễn này cho một mạng NN nông. Điều này giúp chúng ta tiết kiệm thời gian cần thiết để chạy hình ảnh qua tất cả các layer. Nó giống như chuyển đổi hình ảnh thành vectơ.

**Một ví dụ khác:**

Điều gì sẽ xảy ra nếu ở ví dụ trước có nhiều ảnh mèo. Chúng ta có thể đóng băng một vài layer từ đầu mạng đã huấn luyện trước và tìm hiểu các trọng số khác trong mạng. Một số ý tưởng khác là loại các layer không bị đóng băng và đưa các layer của riêng bạn vào đó.

**Một ví dụ khác:**

Nếu có đủ dữ liệu, chúng ta có thể tinh chỉnh tất cả các layer trong mạng đã huấn luyện trước nhưng không khởi tạo ngẫu nhiên các tham số, và hãy để nguyên các tham số đã tìm hiểu và học hỏi từ đó.

### Data Augmentation - tăng cường dữ liệu

Nếu dữ liệu được tăng lên thì NN sâu sẽ hoạt động tốt hơn. Tăng cường dữ liệu là một trong những kỹ thuật mà DL sử dụng để tăng chất lượng của NN sâu.

Hiện nay, phần lớn các ứng dụng thị giác máy tính cần nhiều dữ liệu hơn.

Một số phương pháp tăng cường dữ liệu được sử dụng cho các tác vụ thị giác máy tính bao gồm:

- - Mirroring (Phản chiếu).
  - Random cropping (Cắt xén ngẫu nhiên). Vấn đề với kỹ thuật này là chúng ta có thể cắt sai. Giải pháp là làm cho phần cắt đủ lớn.
  - Rotation (Xoay).
  - Shearing (Cắt).
  - Local warping (Làm cong cục bộ).
  - Color shifting (Chuyển màu).

Ví dụ: chúng ta thêm một số biến dạng vào R, G và B sẽ làm cho hình ảnh được xác định là giống với con người nhưng khác với máy tính.

Trên thực tế, giá trị thêm vào được lấy từ một số phân phối xác suất và những thay đổi này khá nhỏ.

Làm cho thuật toán mạnh mẽ hơn trong việc thay đổi màu sắc hình ảnh.

Thuật toán PCA color augmentation quyết định các thay đổi cần thiết một cách tự động.

Thực hiện các phép biến dạng trong quá trình huấn luyện:

Chúng ta có thể sử dụng một CPU thread khác để tạo các mini-batch bị bóp méo trong khi huấn luyện NN.

Tăng cường dữ liệu cũng có một số siêu tham số. Nên bắt đầu tìm một triển khai tăng cường dữ liệu mã nguồn mở rồi sử dụng nó hoặc tinh chỉnh các siêu tham số này.

### State of Computer Vision

Với một bài toán cụ thể, chúng ta có thể có ít hoặc rất nhiều dữ liệu.

Ví dụ, bài toán nhận dạng giọng nói (speech recognition) có lượng dữ liệu lớn, trong khi nhận dạng hình ảnh (image recognition) có lượng dữ liệu trung bình và phát hiện đối tượng (object detection) có lượng dữ liệu nhỏ.

Nếu bài toán có lượng lớn dữ liệu, các nhà nghiên cứu có xu hướng sử dụng:

- - Các thuật toán đơn giản hơn.
  - Ít kỹ thuật thủ công hơn.

Nếu không có nhiều dữ liệu như vậy, người ta có xu hướng thử kỹ thuật thủ công nhiều hơn cho bài toán “Hacks”, giống như việc chọn một kiến trúc NN phức tạp hơn. Do chúng ta không có nhiều dữ liệu trong nhiều bài toán thị giác máy tính nên nó phụ thuộc rất nhiều vào kỹ thuật thủ công. Chúng ta sẽ thấy rằng, vì phát hiện đối tượng có ít dữ liệu hơn nên sẽ biểu diễn kiến trúc NN phức tạp hơn.

Mẹo để thực hiện tốt các đánh giá xếp hạng/chiến thắng trong các cuộc thi:

**Ensembling.**
Huấn luyện một số mạng độc lập và tính trung bình kết quả đầu ra của chúng. Hợp nhất một số bộ phân loại.
Sau khi chọn được kiến trúc tốt nhất cho bài toán của mình, hãy khởi tạo ngẫu nhiên và huấn luyện chúng một cách độc lập. Điều này có thể giúp tăng 2%.
Tuy nhiên, điều này sẽ làm chậm quá trình tạo ra theo số lượng của các ensemble. Ngoài ra, nó chiếm nhiều dung lượng hơn vì nó lưu tất cả các mô hình trong bộ nhớ. Mọi người sử dụng nó trong các cuộc thi nhưng ít sử dụng nó trong thực tế.

**Multi-crop** ở thời điểm thử nghiệm **(Test Time Augmentation)**

Chạy bộ phân loại trên nhiều phiên bản của phiên bản thử nghiệm và kết quả trung bình. Có một kỹ thuật là 10 crops, hãy sử dụng điều này. Điều này có thể cho chúng ta kết quả tốt hơn.

**Sử dụng mã nguồn mở**

Sử dụng kiến trúc của các mạng được xuất bản trong tài liệu. Sử dụng các triển khai mã nguồn mở nếu có thể. Sử dụng các mô hình đã huấn luyện trước và tinh chỉnh trên tập dữ liệu.
