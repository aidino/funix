## Introduction



**scalar**: số duy nhất.

**vector**: số có hướng (ví dụ: tốc độ gió với hướng).

**matrix**: mảng số 2-chiều.

**tensor**: mảng số n-chiều (trong đó: n là số bất kỳ, tensor 0-chiều là một số vô hướng, tensor 1-chiều là một vector).



### `tf.constant()`

Hằng số

```python
# Tạo một số vô hướng (tensor bậc 0)
scalar = tf.constant(7) # <tf.Tensor: shape=(), dtype=int32, numpy=7>
# Kiểm tra số chiều của tensor (ndim là số chiều)
scalar.ndim # 0

# Tạo một vectơ (nhiều hơn 0 chiều)
vector = tf.constant([10, 10])
# Kiểm tra số chiều của tensor của vectơ tensor
vector.ndim # 1

# Tạo một ma trận (nhiều hơn 1 chiều)
matrix = tf.constant([[10, 7],
                      [7, 10]])
matrix.ndim # 2

# Tạo một tensor
tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]],
                      [[13, 14, 15],
                       [16, 17, 18]]])
tensor.ndim # 3
```

Theo mặc định, TensorFlow tạo các tensor có kiểu dữ liệu `int32` hoặc `float32`.

```python
# Tạo một ma trận khác và xác định kiểu dữ liệu
another_matrix = tf.constant([[10., 7.],
                              [3., 2.],
                              [8., 9.]], dtype=tf.float16) # chỉ định kiểu dữ liệu với 'dtype'
```



### `tf.Variable()`

- **Variable - Constant**

```python
# Tạo tensor tương tự với tf.Variable() và tf.constant()
changeable_tensor = tf.Variable([10, 7])
unchangeable_tensor = tf.constant([10, 7])

# Sẽ có lỗi (yêu cầu phương thức .assign())
changeable_tensor[0] = 7 # Error

# Sẽ không có lỗi
changeable_tensor[0].assign(7)

# Sẽ có error (không thể thay đổi tf.constant())
unchangeable_tensor[0].assign(7) # Error


```

### `tf.random`

Random tensor là các tensor có kích thước bất kỳ chứa các số ngẫu nhiên.

```python
random_1 = tf.random.Generator.from_seed(42) # thiết lập seed cho khả năng tái tạo
random_1 = random_1.normal(shape=(3, 2)) # tạo tensor từ phân phối chuẩn

random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3, 2))

random_1 == random_2

--- result ---
array([[ True,  True],
        [ True,  True],
        [ True,  True]])>)

# Các random tensor mà chúng ta vừa tạo thực ra là số giả(chúng xuất hiện ngẫu nhiên nhưng thực sự không phải như vậy).
# Nếu thiết lập seed, chúng ta sẽ nhận được các số ngẫu nhiên (tương tự như `np.random.seed(42)` khi dùng NumPy).
# Thiết lập seed, giả sử ""hey, create some random numbers, but flavour them with X" (X là seed).

# Khi thay đổi seed, các số random sẽ khác nhau
```

### `tf.random.shuffle`

Tại sao chúng ta lại muốn trộn dữ liệu?

Giả sử chúng ta đang làm việc với 15,000 hình ảnh về mèo và chó, trong đó 10,000 hình ảnh đầu tiên là về mèo và 5,000 hình ảnh tiếp theo là về chó. Thứ tự này có thể ảnh hưởng đến cách mạng nơ-ron học (nó có thể overfit khi tìm hiểu thứ tự của dữ liệu), thay vào đó, chúng ta nên di chuyển dữ liệu xung quanh.

```python
not_shuffled = tf.Variable([[ 3,  4],
                            [ 2,  5],
                            [10,  7]])

tf.random.shuffle(not_shuffled) # Mỗi lần chạy sẽ ra một kết quả khác nhau

--- result ---
array([[ 2,  5],
       [ 3,  4],
       [10,  7]], dtype=int32)>

# Xáo trộn theo thứ tự tương tự mỗi lần sử dụng tham số seed (sẽ không thực sự như nhau)
tf.random.shuffle(not_shuffled, seed=42)

--- result ---
array([[10,  7],
       [ 2,  5],
       [ 3,  4]], dtype=int32)>

```

**Important note**

>  *Nếu cả global seed và operation seed đều được thiết lập: Cả 2 seed được sử dụng kết hợp để xác định trình tự ngẫu nhiên.*

```python
# Thiết lập global random seed
tf.random.set_seed(42)
# Thiết lập operation random seed
tf.random.shuffle(not_shuffled, seed=42)

--- result ---
array([[ 3,  4],
       [ 2,  5],
       [10,  7]], dtype=int32)>

# Thiết lập global random seed
tf.random.set_seed(42) # nếu biến đổi nó thành chú thích, chúng ta sẽ nhận được các kết quả khác nhau
# Thiết lập operation random seed
tf.random.shuffle(not_shuffled)

--- result ---
array([[ 2,  5],
       [10,  7],
       [ 3,  4]], dtype=int32)>

```

### `tf.ones`

```python
# Tạo tensor có các giá trị 1
tf.ones(shape=(3, 2))

--- result ---
array([[1., 1.],
       [1., 1.],
       [1., 1.]], dtype=float32)>
```

### `tf.zeros`

```python
# Tạo tensor có các giá trị 0
tf.zeros(shape=(3, 2))

--- result ---
array([[0., 0.],
       [0., 0.],
       [0., 0.]], dtype=float32)>
```



## Lấy thông tin từ tensor

Sẽ có những lúc chúng ta muốn lấy những thông tin khác nhau từ tensor của mình, cụ thể, chúng ta nên biết những thuật ngữ tensor sau:
* **Shape (hình dạng):** Độ dài (số phần tử) của từng chiều của tensor.
* **Rank (bậc):** Số chiều của tensor. Số vô hướng có rank (bậc) 0, vectơ có rank 1, ma trận có có rank 2 và tensor có rank n.
* **Axis (trục)** or **Dimension (chiều):** Các chiều cụ thể của tensor.
* **Size (kích thước):** Tổng số mục trong tensor.

Chúng ta sẽ đặc biệt sử dụng chúng khi sắp xếp shape của dữ liệu thành shape của mô hình. Ví dụ: đảm bảo shape của image tensor giống với shape của lớp đầu vào mô hình.

### `dtype`, `ndim`, `shape`, `size`

```python
# Tạo tensor bậc 4 (4 chiều)
rank_4_tensor = tf.zeros([2, 3, 4, 5])

# Lấy nhiều thuộc tính của tensor
print("Datatype of every element:", rank_4_tensor.dtype)
print("Number of dimensions (rank):", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (2*3*4*5):", tf.size(rank_4_tensor).numpy()) # .numpy() converts to NumPy array

--- result ---
Datatype of every element: <dtype: 'float32'>
Number of dimensions (rank): 4
Shape of tensor: (2, 3, 4, 5)
Elements along axis 0 of tensor: 2
Elements along last axis of tensor: 5
Total number of elements (2*3*4*5): 120
```

```python
# Lấy 2 mục đầu tiên của mỗi chiều
rank_4_tensor[:2, :2, :2, :2]

--- result ---
array([[[[0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.]]],

       [[[0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.]]]], dtype=float32)>

```

```python
# Tạo tensor bậc 2 (2 chiều)
rank_2_tensor = tf.constant([[10, 7],
                             [3, 4]])

# Lấy mục cuối của mỗi hàng
rank_2_tensor[:, -1]

--- result ---
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([7, 4], dtype=int32)>
```

### Thêm chiều với `tf.newaxis`

```python
# Thêm một chiều bổ sung (vào cuối)
rank_3_tensor = rank_2_tensor[..., tf.newaxis] # trong Python "..." nghĩa là "tất cả các chiều trước"
rank_2_tensor, rank_3_tensor # shape (2, 2), shape (2, 2, 1)

--- result ---
(<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
 array([[10,  7],
        [ 3,  4]], dtype=int32)>,
 <tf.Tensor: shape=(2, 2, 1), dtype=int32, numpy=
 array([[[10],
         [ 7]],
 
        [[ 3],
         [ 4]]], dtype=int32)>)
```

### Thêm chiều với `tf.expand_dims`

```python
tf.expand_dims(rank_2_tensor, axis=-1) # "-1" means last axis

--- result---
<tf.Tensor: shape=(2, 2, 1), dtype=int32, numpy=
array([[[10],
        [ 7]],

       [[ 3],
        [ 4]]], dtype=int32)>
```

## Thao tác tensor (các phép toán với tensor)

### `+`

```python
# Có thể cộng thêm giá trị vào tensor sử dụng toán tử cộng
tensor = tf.constant([[10, 7], [3, 4]])
tensor + 10

--- result ---
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[20, 17],
       [13, 14]], dtype=int32)>
```

**Note**: Do chúng ta dùng `tf.constant()` nên tensor ban đầu không đổi (phép cộng được thực hiện trên một bản sao).

```python
# Tensor ban đầu không đổi
tensor

--- result ---
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[10,  7],
       [ 3,  4]], dtype=int32)>
```

### `-`

```python
# Phép trừ
tensor - 10

--- result ---
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[ 0, -3],
       [-7, -6]], dtype=int32)>
```

### `tf.multiply`, `*`



```python
# Phép nhân (hay phép nhân theo từng phần tử)
tensor * 10

--- result ---
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[100,  70],
       [ 30,  40]], dtype=int32)>
```

**or**

Chúng ta cũng có thể sử dụng hàm TensorFlow tương đương. Sử dụng hàm TensorFlow (nếu có thể) có lợi thế là tăng tốc sau đó xuống dòng khi chạy như một phần của [đồ thị TensorFlow](https://www.tensorflow.org/tensorboard/graphs).

```python
# Sử dụng hàm TensorFlow tương đương của toán tử '*' (nhân)
tf.multiply(tensor, 10)

```

###  `tf.matmul()`, `@`

Một trong những phép toán cơ bản nhất của thuật toán học máy là [phép nhân ma trận](https://www.mathsisfun.com/algebra/matrix-multiplying.html). TensorFlow triển khai chức năng nhân ma trận này theo phương thức [`tf.matmul()`](https://www.tensorflow.org/api_docs/python/tf/linalg/matmul).

2 quy tắc nhân ma trận chính cần nhớ là:

1. Các dimension bên trong cần khớp nhau"
  * `(3, 5) @ (3, 5)` sẽ không hoạt động
  * `(5, 3) @ (3, 5)` sẽ hoạt động
  * `(3, 5) @ (5, 3)` sẽ hoạt động
2. Ma trận kết quả có shape của dimension bên ngoài:
 * `(5, 3) @ (3, 5)` -> `(5, 5)`
 * `(3, 5) @ (5, 3)` -> `(3, 3)`

> 🔑 **Lưu ý:** '`@`' là ký hiệu cho phép nhân trong Python.

```python
tf.matmul(tensor, tensor)

# or

# Phép nhân ma trận với toán tử '@' của Python
tensor @ tensor
```

![lining up dimensions for dot products](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/00-lining-up-dot-products.png)



![visual demo of matrix multiplication](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/00-matrix-multiply-crop.gif)



### `tf.reshape`

```python
# Ví dụ: reshape (3, 2) -> (2, 3)
tf.reshape(Y, shape=(2, 3))
```



### `tf.transpose`

```python
# Ví dụ chuyển vị (3, 2) -> (2, 3)
tf.transpose(X)
```

### `tf.tensordot`

Nhân các ma trận với nhau còn được gọi là tích vô hướng. Chúng ta có thể thực hiện thao tác `tf.matmul()` sử dụng [`tf.tensordot()`](https://www.tensorflow.org/api_docs/python/tf/tensordot).

```python
# Thực hiện tích vô hướng trên X và Y (cần X là chuyển vị)
tf.tensordot(tf.transpose(X), Y, axes=1)

--- result ---
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[ 89,  98],
       [116, 128]], dtype=int32)>
```



**Important note: ** 

Đầu ra của việc gọi `tf.reshape()` và `tf.transpose()` trên `Y` khác nhau dù chúng có shape tương tự.

Điều này có thể giải thích là do khả vi mặc định của từng phương thức:

* [`tf.reshape()`](https://www.tensorflow.org/api_docs/python/tf/reshape) - thay đổi shape của tensor đã cho (đầu tiên) rồi chèn các giá trị theo thứ tự chúng xuất hiện 

* [`tf.transpose()`](https://www.tensorflow.org/api_docs/python/tf/transpose) - hoán đổi thứ tự của các trục, mặc định trục cuối thành trục đầu, nhưng có thể thay đổi thứ tự bằng cách sử dụng [tham số `perm`](https://www.tensorflow.org/api_docs/python/tf/transpose).

Vậy chúng ta nên sử dụng cái nào?

Một lần nữa, hầu hết thời gian hoạt động (sẽ được thực hiện cho bạn khi chúng cần chạy, chẳng hạn như trong suốt quá trình huấn luyện mạng nơ-ron).

Nhưng nhìn chung, bất cứ khi nào tiến hành phép nhân ma trận và shape của hai ma trận không thẳng hàng, chúng ta sẽ không chuyển vị (không reshape) một trong số chúng để xếp cho chúng thẳng hàng.



### `tf.cast()`

Đôi khi chúng ta cần thay đổi kiểu dữ liệu mặc định của tensor.

Điều này thường xảy ra khi bạn muốn tính toán sử dụng độ chính xác thấp hơn (ví dụ: số thực dấu phẩy động 16 bit với số thực dấu phẩy động 32 bit).

Tính toán với độ chính xác thấp hơn rất hữu ích ở các thiết bị có dung lượng tính toán thấp hơn như thiết bị di động (vì càng ít bit thì càng yêu cầu dung lượng tính toán thấp hơn).

Chúng ta có thể thay đổi kiểu dữ liệu của tensor sử dụng [`tf.cast()`](https://www.tensorflow.org/api_docs/python/tf/cast).

```python
# Tạo tensor mới có kiểu dữ liệu mặc định (float32)
B = tf.constant([1.7, 7.4])

# Tạo tensor mới có kiểu dữ liệu mặc định (int32)
C = tf.constant([1, 7])
B, C

--- result ---
(<tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.7, 7.4], dtype=float32)>,
 <tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 7], dtype=int32)>)

# Thay đổi từ float32 thành float16 (giảm độ chính xác)
B = tf.cast(B, dtype=tf.float16)
B

--- result ---
<tf.Tensor: shape=(2,), dtype=float16, numpy=array([1.7, 7.4], dtype=float16)>


# Thay đổi từ int32 thành float32
C = tf.cast(C, dtype=tf.float32)
C

--- result ---
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 7.], dtype=float32)>
```

### `tf.abs()`

Đôi khi chúng ta muốn lấy giá trị tuyệt đối (tất cả các giá trị dương) của các phần tử trong tensor.

Để thực hiện, hãy dùng [`tf.abs()`](https://www.tensorflow.org/api_docs/python/tf/math/abs).

```python
# Tạo tensor có giá trị âm
D = tf.constant([-7, -10])
# Lấy giá trị tuyệt đối
tf.abs(D)

--- result ---
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([ 7, 10], dtype=int32)>
```



### `tf.reduce_min()`, `tf.reduce_max()`, `tf.reduce_mean()`, `tf.reduce_sum()`

Chúng ta có thể nhanh chóng kết tập (thực hiện phép tính trên toàn bộ tensor) các tensor để tìm giá trị nhỏ nhất, giá trị lớn nhất, giá trị trung bình và tổng của tất cả các phần tử.

Để thực hiện, chúng ta sử dụng phương thức kết tập có cú pháp `reduce()_[action]`, gồm:
* [`tf.reduce_min()`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_min) - tìm giá trị nhỏ nhất trong tensor.
* [`tf.reduce_max()`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_max) - tìm giá trị lớn nhất trong tensor (hữu ích khi tìm xác suất dự đoán cao nhất).
* [`tf.reduce_mean()`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean) - tìm giá trị trung bình của tất cả các phần tử trong tensor.
* [`tf.reduce_sum()`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum) - tìm tổng của tất cả các phần tử trong tensor.
* **Lưu ý:** mỗi loại này thường nằm trong một mô-đun `math`, chẳng hạn như `tf.math.reduce_min()` nhưng chúng ta cũng có thể sử dụng alias `tf.reduce_min()`.

```python
# Tạo tensor có 50 giá trị ngẫu nhiên trong khoảng 0-100
E = tf.constant(np.random.randint(low=0, high=100, size=50))

# Tìm giá trị nhỏ nhất
tf.reduce_min(E)

# Tìm giá trị lớn nhất
tf.reduce_max(E)

# Tìm tổng
tf.reduce_sum(E)

```

Chúng ta cũng có thể tìm độ lệch chuẩn ([`tf.reduce_std()`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_std)) và phương sai ([`tf.reduce_variance()`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_variance)) của các phần tử trong tensor bằng các phương thức tương tự.



### `tf.argmax()`, `tf.argmin()`

Làm sao để tìm vị trí mà tensor có giá trị lớn nhất?

Sẽ hữu ích nếu bạn muốn sắp xếp các nhãn (giả sử `['Green', 'Blue', 'Red']`) với tensor xác suất dự đoán (chẳng hạn `[0.98, 0.01, 0.01]`).

Trong trường hợp này, nhãn đã dự đoán (nhãn có xác suất dự đoán cao nhất) là `'Green'`.

Chúng ta có thể thực hiện tương tự với giá trị nhỏ nhất (nếu cần) với các phương thức sau:

* [`tf.argmax()`](https://www.tensorflow.org/api_docs/python/tf/math/argmax) - tìm vị trí của phần tử lớn nhất trong tensor đã cho.

* [`tf.argmin()`](https://www.tensorflow.org/api_docs/python/tf/math/argmin) - tìm vị trí của phần tử nhỏ nhất trong tensor đã cho.

```python
# Tạo tensor có 50 giá trị trong khoảng từ 0 đến 1
F = tf.constant(np.random.random(50))

# Tìm vị trí phần tử lớn nhất của F
tf.argmax(F)

# Tìm vị trí phần tử nhỏ nhất của F
tf.argmin(F)

# Tìm vị trí phần tử lớn nhất của F
print(f"The maximum value of F is at position: {tf.argmax(F).numpy()}") 
print(f"The maximum value of F is: {tf.reduce_max(F).numpy()}") 
print(f"Using tf.argmax() to index F, the maximum value of F is: {F[tf.argmax(F)].numpy()}")
print(f"Are the two max values the same (they should be)? {F[tf.argmax(F)].numpy() == tf.reduce_max(F).numpy()}")

--- result ---
The maximum value of F is at position: 35
The maximum value of F is: 0.9829797726410907
Using tf.argmax() to index F, the maximum value of F is: 0.9829797726410907
Are the two max values the same (they should be)? True
```



### `tf.squeeze()`

Chúng ta có thể dùng `tf.squeeze()` để loại các chiều đơn lẻ khỏi tensor (chiều có size 1).

[`tf.squeeze()`](https://www.tensorflow.org/api_docs/python/tf/squeeze) - loại tất cả các chiều có size là 1 khỏi tensor.

```python
# Tạo một tensor bậc 5 (5 chiều) có 50 số trong khoảng 0-100
G = tf.constant(np.random.randint(0, 100, 50), shape=(1, 1, 1, 1, 50))
G.shape, G.ndim

--- result ---
(TensorShape([1, 1, 1, 1, 50]), 5)

# Nén tensor G (loại tất cả các chiều 1)
G_squeezed = tf.squeeze(G)
G_squeezed.shape, G_squeezed.ndim

--- result ---
(TensorShape([50]), 1)
```

### `tf.one_hot()`

Chúng ta có thể sử dụng [`tf.one_hot()`](https://www.tensorflow.org/api_docs/python/tf/one_hot) để mã hóa one-hot một tensor gồm các chỉ số.

Chúng ta cũng nên chỉ định tham số `depth` (muốn mã hóa sâu bao nhiêu). 

```python
# Tạo một danh sách các chỉ số
some_list = [0, 1, 2, 3]

# Mã hóa one-hot chúng
tf.one_hot(some_list, depth=4)

--- result ---
<tf.Tensor: shape=(4, 4), dtype=float32, numpy=
array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]], dtype=float32)>
```

### `on_value`, `off_value`

Chúng ta cũng có thể chỉ định giá trị cho `on_value` và `off_value` thay vì `0` và `1` mặc định.

```python
# Specify custom values for on and off encoding
tf.one_hot(some_list, depth=4, on_value="We're live!", off_value="Offline")

--- result ---
<tf.Tensor: shape=(4, 4), dtype=string, numpy=
array([[b"We're live!", b'Offline', b'Offline', b'Offline'],
       [b'Offline', b"We're live!", b'Offline', b'Offline'],
       [b'Offline', b'Offline', b"We're live!", b'Offline'],
       [b'Offline', b'Offline', b'Offline', b"We're live!"]], dtype=object)>
```

### `tf.square()`, `tf.sqrt()`, `tf.math.log()`

Chúng ta có thể thực hiện nhiều phép toán khác ở một số giai đoạn.

Chẳng hạn:

* [`tf.square()`](https://www.tensorflow.org/api_docs/python/tf/math/square) - tính bình phương của mọi giá trị trong tensor.

* [`tf.sqrt()`](https://www.tensorflow.org/api_docs/python/tf/math/sqrt) - tính căn bậc hai của mọi giá trị trong tensor. (**lưu ý:** các phần tử phải là kiểu float, nếu không sẽ sai).

* [`tf.math.log()`](https://www.tensorflow.org/api_docs/python/tf/math/log) - tính log tự nhiên của mọi giá trị trong tensor (các phần tử phải là float).



```python
# Tạo tensor mới
H = tf.constant(np.arange(1, 10))

--- result ---
<tf.Tensor: shape=(9,), dtype=int64, numpy=array([1, 2, 3, 4, 5, 6, 7, 8, 9])>

# Bình phương nó
tf.square(H)

--- result ---
<tf.Tensor: shape=(9,), dtype=int64, numpy=array([ 1,  4,  9, 16, 25, 36, 49, 64, 81])>

# Tính căn bậc hai (lỗi), không phải là số nguyên
tf.sqrt(H)

--- result ---
InvalidArgumentError: Value for attr 'T' of int64 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128

# Đổi H thành float32
H = tf.cast(H, dtype=tf.float)
# Tính căn bậc hai
tf.sqrt(H)

# Tính log (đầu vào cần là float)
tf.math.log(H)
```



### `.assign()`, `.add_assign()`

Những tensor được tạo với `tf.Variable()` có thể thay đối tại chỗ với các phương thức sau:

* [`.assign()`](https://www.tensorflow.org/api_docs/python/tf/Variable#assign) - gán một giá trị khác cho một chỉ mục cụ thể của tensor variable.

* [`.add_assign()`](https://www.tensorflow.org/api_docs/python/tf/Variable#assign_add) - thêm vào một giá trị hiện có và gán lại nó ở một chỉ mục cụ thể của tensor variable.

```python
# Tạo tensor variable
I = tf.Variable(np.arange(0, 5))
I

--- result ---
<tf.Variable 'Variable:0' shape=(5,) dtype=int64, numpy=array([0, 1, 2, 3, 4])>

# Gán cho giá trị cuối cùng một giá trị mới là 50
I.assign([0, 1, 2, 3, 50])

--- result ---
<tf.Variable 'UnreadVariable' shape=(5,) dtype=int64, numpy=array([ 0,  1,  2,  3, 50])>

# Thêm 10 vào mỗi phần tử trong I
I.assign_add([10, 10, 10, 10, 10])


--- result ---
<tf.Variable 'UnreadVariable' shape=(5,) dtype=int64, numpy=array([10, 11, 12, 13, 60])>

```

## Tensor and Numpy

Chúng ta đã thấy một số ví dụ về tensor với mảng NumPy như sử dụng mảng NumPy để tạo tensor.

Có thể chuyển đổi tensor thành mảng NumPy bằng:

* `np.array()` - chuyển một tensor để chuyển đổi thành một mảng n-chiều (kiểu dữ liệu chính của NumPy).

* `tensor.numpy()` - gọi một tensor để chuyển thành một mảng n-chiều.

Điều này hữu ích vì nó khiến các tensor lặp lại và cho phép chúng ta sử dụng bất kỳ phương thức NumPy nào trên đó.

```python
# Tạo tensor từ mảng NumPy
J = tf.constant(np.array([3., 7., 10.]))
---
<tf.Tensor: shape=(3,), dtype=float64, numpy=array([ 3.,  7., 10.])>

# Chuyển đổi tensor J thành mảng NumPy với np.array()
np.array(J), type(np.array(J))
---
(array([ 3.,  7., 10.]), numpy.ndarray)

# Chuyển đổi tensor J thành mảng NumPy với .numpy()
J.numpy(), type(J.numpy())
---
(array([ 3.,  7., 10.]), numpy.ndarray)
```

Theo mặc định, tensor có `dtype=float32`, trong khi mảng NumPy có `dtype=float64`.

Điều này là do mạng nơ-ron (thường được tạo với TensorFlow) có thể hoạt động tốt với độ chính xác thấp hơn (32 bit hơn là 64 bit).

```python
# Tạo tensor từ NumPy và từ một mảng
numpy_J = tf.constant(np.array([3., 7., 10.])) # sẽ là float64 (do NumPy)
tensor_J = tf.constant([3., 7., 10.]) # sẽ là float32 (do mặc định là TensorFlow)
numpy_J.dtype, tensor_J.dtype
---
(tf.float64, tf.float32)
```



## `@tf.function`

Trong quá trình tìm hiểu TensorFlow, chúng ta có thể gặp các hàm Python có decorator [`@tf.function`](https://www.tensorflow.org/api_docs/python/tf/function).

Nếu chưa rõ về Python decorator, hãy đọc [hướng dẫn của RealPython về decorator](https://realpython.com/primer-on-python-decorators/).

Tóm lại, decorator sửa đổi một hàm không bằng cách này thì bằng cách khác.

Trong trường hợp sử dụng decorator `@tf.function`, nó biến hàm Python thành một đồ thị TensorFlow có thể gọi được. Đây là một cách nói hoa mỹ, nếu bạn viết hàm Python của riêng mình và trang bị nó với `@tf.function`, thì khi bạn xuất code (để chạy trên thiết bị khác), TensorFlow sẽ cố chuyển đổi nó thành một phiên bản nhanh (hơn) của chính nó (bằng cách biến nó thành một phần của đồ thị tính toán).

Để biết thêm chi tiết, hãy đọc hướng dẫn [Better performance with tf.function](https://www.tensorflow.org/guide/function).

```python
# Tạo một hàm đơn giản
def function(x, y):
  return x ** 2 + y

x = tf.constant(np.arange(0, 10))
y = tf.constant(np.arange(10, 20))
function(x, y)
---
<tf.Tensor: shape=(10,), dtype=int64, numpy=array([ 10,  12,  16,  22,  30,  40,  52,  66,  82, 100])>

# Tạo một hàm tương tự và trang bị với tf.function
@tf.function
def tf_function(x, y):
  return x ** 2 + y

tf_function(x, y)
---
<tf.Tensor: shape=(10,), dtype=int64, numpy=array([ 10,  12,  16,  22,  30,  40,  52,  66,  82, 100])>
```

Nếu không thấy khác biệt giữa hai hàm trên (hàm được trang bị và hàm không được trang bị) tức là bạn đã đúng.

Phần lớn sự khác biệt xảy ra ẩn sau. Một trong những khác biệt chính là tăng tốc độ code tiềm năng khi có thể.



## Tìm truy cập vào GPU

Chúng ta có thể kiểm tra xem có truy cập vào GPU nào không bằng [`tf.config.list_physical_devices()`](https://www.tensorflow.org/guide/gpu).

```python
print(tf.config.list_physical_devices('GPU'))
```

Nếu ở trên xuất ra một mảng trống (hoặc không có gì), điều này có nghĩa là chúng ta không có quyền truy cập vào GPU (hoặc ít nhất là TensorFlow không thể tìm thấy nó).

Nếu chạy trong Google Colab, chúng ta có thể truy cập GPU bằng cách vào **Runtime -> Change Runtime Type -> Select GPU** (**lưu ý:** sau khi thực hiện điều này, notebook sẽ khởi động lại và bất kỳ biến nào mà chúng ta đã lưu sẽ bị mất).

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
---example
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

🔑 **Lưu ý:** *Nếu chúng ta có quyền truy cập vào GPU, TensorFlow sẽ tự động sử dụng nó bất cứ khi nào có thể.*



## 📖 Tài liệu tham khảo

* Đọc qua [danh sách TensorFlow Python API](https://www.tensorflow.org/api_docs/python/), chọn một cái mà chúng ta chưa tìm hiểu trong notebook này, thiết kế ngược (tự viết code tài liệu) và tìm hiểu xem nó có tác dụng gì.
* Thử tạo một chuỗi các hàm tensor để tính toán các hóa đơn tạp hóa gần đây nhất (không cần tên các mặt hàng, chỉ cần giá cả ở dạng số).
  * Chúng ta sẽ tính toán hóa đơn tạp hóa theo tháng và năm thế nào khi sử dụng tensor?
* Xem qua hướng dẫn [TensorFlow 2.x quick start for beginners](https://www.tensorflow.org/tutorials/quickstart/beginner) (đảm bảo tự gõ toàn bộ code dù bạn không hiểu nó).
  * Còn hàm nào mà chúng ta sử dụng ở đây khớp với những gì chúng ta sử dụng ở đó không? Những cái nào giống nhau? Những cái nào chúng ta chưa từng thấy trước đây?
* Xem video ["What's a tensor?"](https://www.youtube.com/watch?v=f5liqUk0ZTw) - giới thiệu trực quan tuyệt vời về các khái niệm mà chúng ta đã đề cập trong notebook này.
