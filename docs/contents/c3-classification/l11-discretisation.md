# Discretization - Rời rạc hoá

## Introduction

>  Rời rạc hóa là quá trình biến đổi một biến liên tục thành biến rời rạc bằng cách tạo một tập hợp các khoảng liên tục trải dài trong phạm vi giá trị của biến. 

Rời rạc hóa (Discretisation) còn được gọi là binning, trong đó mỗi khoảng này cũng được gọi là bin. Rời rạc hóa có thể cải thiện chênh lệch giá trị của biến, cũng như xử lý ngoại lai. 

Chúng ta sẽ tiếp cận với 2 phương pháp tiếp cận rời rạc hóa, đó là **Supervised** (có giám sát) và **Unsupervised** (không giám sát).



## Rời rạc hoá không giám sát

### Rời rạc hoá sử dụng khoảng cách bằng nhau

Rời rạc hóa sử dụng khoảng cách bằng nhau chia phạm vi giá trị có thể của biến thành N bin hoặc các khoảng (interval). 
$$
width = \frac{\text{max value} - \text{min value}}{N}
$$
$N$: Number of bins or intervals

Các khoảng/bin này có khoảng cách (width) như nhau; khoảng cách được xác định bởi giá trị lớn nhất của biến, giá trị nhỏ nhất của biến và số khoảng mà chúng ta muốn tạo.

Phương pháp này:

- Không cải thiện chênh lệch giá trị
- Xử lý được outliers
- Tạo ra biến rời rạc
- Có kết hợp tốt với categorical encodings

### Rời rạc hoá sử dụng tần số bằng nhau

Rời rạc hóa sử dụng tần số bằng nhau chia phạm vi các giá trị có thể của biến thành một số khoảng mà mỗi khoảng lại chứa số lượng quan sát xấp xỉ nhau. Thông thường, để tính ranh giới cho từng khoảng này, chúng ta sẽ tính các **quantile** của biến.

Kỹ thuật này giúp cải thiện chênh lệch giá trị và nó cũng xử lý outlier khi chúng được phân bổ vào vùng đầu hoặc cuối. Việc cải thiện chênh lệch giá trị sẽ giúp các mô hình tuyến tính giả định chênh lệch đều hơn hoặc phân phối các giá trị của biến chuẩn hơn.

Phương pháp này:

- Có ảnh hưởng kèm theo là nó phân phối các giá trị đồng nhất hơn trên toàn bộ phạm vi giá trị của biến
- Cải thiện chênh lệch giá trị
- Xử lý được outlier khi chúng được phân bổ vào vùng đầu hay vùng cuối
- Tạo ra các biến rời rạc
- Có kết hợp tốt với categorical encodings

### Rời rạc hoá sử dụng K-means

Rời rạc hóa sử dụng K-means gồm việc áp dụng phân cụm K-means vào biến liên tục để thu được các cụm khác nhau, mỗi cụm tương ứng với một bin mà chúng ta sẽ sắp xếp các giá trị của biến. 

Phương pháp này:

- Không cải thiện chênh lệch giá trị
- Có thể xử lý được outliers, mặc dù outlier có thể có ảnh hưởng đến vị trí tâm của các cụm
- Tạo ra các biến rời rạc
- Có kết hợp tốt với categorical encodings

### Rời rạc hoá kết hợp với Categorical encoding

**Note:** Trên thực tế, cách hữu ích để mã hoá các **bin** này là sử dụng bộ mã hoá tạo ra mối quan hệ đơn điệu giữa **bin** và **target**



## Rời rạc hoá có giám sát

### Rời rạc hoá sử dụng Cây phân loại

Rời rạc hóa sử dụng cây phân loại là một thủ tục quyết định rời rạc có giám sát, trong đó chúng ta sử dụng decision tree (DT) để xác định bin tối ưu mà chúng ta cần sắp xếp các giá trị của biến. Khi DT đưa ra quyết định, nó sẽ chỉ định một quan sát cho một trong n lá cuối của DT. Vì DT có số lượng xác định các lá hoặc nút kết thúc nên nó sẽ biến đổi một biến liên tục thành một đầu ra rời rạc.

DT giúp chúng ta xử lý outlier. Kỹ thuật này cũng hữu ích với các mô hình tuyến tính vì nó tránh việc phải thực hiện hai bước như chúng ta đã làm trước đây, khi rời rạc hóa biến trước tiên rồi sau đó mới mã hóa biến để có được mối quan hệ đơn điệu (sử dụng DT giúp chúng ta có thể làm hai việc cùng một lúc).

### Rời rạc hoá theo Domain knowledge (Kiến thức chuyên ngành)

Chúng ta thường muốn chia các biến thành một tập hợp các khoảng xác định mà chúng ta xác định trước bằng cách sử dụng kiến thức chuyên ngành về nghiệp vụ hoặc lĩnh vực. Do vậy, chúng ta sẽ không áp dụng bất kỳ kỹ thuật nào khác mà chúng ta đã thảo luận cho tới giờ. Thay vào đó, chúng ta sẽ đưa ra các khoảng do chúng có ý nghĩa.

 

#### Tài liệu tham khảo

- [Discretisation: An Enabling Technique](http://www.public.asu.edu/~huanliu/papers/dmkd02.pdf)
- [Supervised and unsupervised discretisation of continuous features](http://ai.stanford.edu/~ronnyk/disc.pdf)
- [ChiMerge: Discretisation of Numeric Attributes](https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf)
- [Beating Kaggle the easy way](https://www.ke.tu-darmstadt.de/lehre/arbeiten/studien/2015/Dong_Ying.pdf)
- [Tips for honing logistic regression models](https://blog.zopa.com/2017/07/20/tips-honing-logistic-regression-models/)
- [ChiMerge discretisation algorithm](https://alitarhini.wordpress.com/2010/11/02/chimerge-discretization-algorithm/)
- [Score card development stages: Binning](https://plug-n-score.com/learning/scorecard-development-stages.htm#Binning)



<hr>

## Lab

### 10.1 Cardinality với bài toán phân loại

**Nhắc lại** Cardinality là số lượng các nhãn khác nhau trong một biến hạng mục.

Biến hạng mục mà có nhiều nhãn khác nhau gọi là biến đó có **high cardinality**

**High cardinality** có thể đối mặt với các vẫn đề sau:

- Các biến có quá nhiều nhãn có xu hướng chiếm ưu thế hơn so với những biến chỉ có một vài nhãn, đặc biệt là trong các thuật toán **cây**.

- Nhiều nhãn trong một biến có thể gây nhiễu với một ít thông tin (nếu có), do đó khiến các mô hình học máy dễ bị overfit.

- Một số nhãn có thể chỉ xuất hiện trong tập dữ liệu huấn luyện, không có trong tập kiểm tra, do đó các thuật toán học máy có thể quá khớp với tập huấn luyện.

- Ngược lại, một số nhãn có thể chỉ xuất hiện trong tập kiểm tra, do đó các thuật toán máy học không thể thực hiện phép tính với quan sát mới (không nhìn thấy).

Đặc biệt, **các phương thức cây có thể thiên về các biến có nhiều nhãn** (các biến với high cardinality). Do đó, chất lượng của chúng có thể bị ảnh hưởng bởi high cardinality.