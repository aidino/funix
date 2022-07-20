# Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

## Practical aspects of Deep Learning

### Train/Dev/Test sets

- Việc có được những Hyperparameter trong ngay những lần thử nghiệm đầu tiên là điều không thể, do đó, khi giải quyết một vấn đề, chúng ta sẽ luôn xoay vòng qua các bước: `Idea ==> Code ==> Experiment`. Chúng ta sẽ phải đi qua vòng lặp này nhiều lần để tìm ra các hyperparameter của mình.

  <img src="images/ml_hyperparameter_turning.png" style="zoom:70%;" />

-  Dữ liệu sẽ được chia làm ba phần

  - Training set (phải là tập lớn nhất)
  - Cross Validation / Dev set
  - Test set

  Dữ liệu sẽ được huấn luyện trên `Training set` , sau đó dùng `Cross Validation/Dev set` để tối ưu hoá các Hyperparameters càng nhiều càng tốt. Sau khi mô hình đã sẵn sàng, dùng tập thử nghiệm để đánh giá.

- Tỉ lệ phân chia các tập:

  - Nếu kích thước tập dữ liệu từ 100 đến 1.000.000 => `Train/Dev/Test = 60/20/20`
  - Nếu kích thước tập dữ liệu lớn hơn 1.000.000 ==> `Train/Dev/Test = 98/1/1 or 99.5/0.25/0.25`

- Cần đảm bảo Dev set và test set có cùng một phân phối

  Ví dụ: Trong bài toán phân loại mèo thì, nếu training set là ảnh lấy từ mạng Internet, trong khi đó ảnh trong Dev/Test set lại là ảnh người dùng đưa lên, chúng sẽ không khớp. Vậy tốt hơn hết là nên đảm bảo rằng dev set và test set là trên cùng một phân phối

- Hoàn toàn có thể chỉ dùng Dev set mà không cần Test set.

### Bias/Variance

![](images/bias-and-variance.png)

- **Bias** là sai số trên Training/Dev sets
- **Phương sai (Variance)** là sai số trên Test set. 
- Nếu mô hình không phù hợp (Logistic Regression of non liner data) sẽ có **Bias cao**
- Nếu mô hình Overfit thì sẽ có **Variance cao**
- mô hình ổn khi có độ cân bằng giữa **Bias** và **Variance**

Ngoài việc dùng đồ thị thì có môt cách khác để đánh giá được **Bias** và **Variance**

- High variance (Overfiting)
  - Training error: 1%
  - Dev error: 11%
- High Bias (Underfiting)
  - Training error: 15%
  - Dev error: 14%
- High bias (underfiting) and High variance (overfitting)
  - Traning error: 15%
  - Test error: 30%
- Best
  - Training error: 0.5%
  - Test error: 1%

Các giả định này đều đến từ sau số mức độ con người là 0%, tất cả lỗi đánh giá phải lấy lỗi con người làm cơ sở.

Lỗi con người ở đây là những bài toán mà ngay cả khi con người cũng có khả năng nhầm lẫn, chúng ta cần lấy con số đó làm tiêu chuẩn để đánh giá Bias và Variance xem chúng ta đang bị overfiting hay underfiting.



### Công thức cơ bản cho Machine Learning

- Với các bài toán có Bias cao (Sai số trên Training set cao)
  - Cố gắng làm cho Neural network của bạn lớn hơn (Size of hidden units / number of layers)
  - Thử nghiệm một model mới phù hợp với Data của bạn hơn
  - Sử dụng các thuật toán tối ưu hoá khác nhau
- Với các bài toán có Variance cao (Sai số trên test set)
  - Cần nhiều dữ liệu hơn
  - Thử regularization
  - Thử nghiệm model khác phù hợp với Data của bạn hơn
- Thời gian đầu của machine learning, chúng ta thường hướng tới sự cân bằng giữa Variance và Bias. Nhưng ngày nay, khi mà chúng ta đã có thêm nhiều công cụ để giải quyết vấn đề về Bias và Variance, nên nó sẽ thực sự hữu ích khi sử dụng Deep Learning.
- Đào tạo một neural network lớn hơn không bao giờ gây hại.

### Regularization

- Thêm Regularization vào trong NN sẽ giúp nó giảm phương sai (overfiting)

- Norm 1: $\parallel W \parallel = \text{SUM}(\left| W[i,j] \right|)$ # Sum of absolute values of all `w`

- Norm 2: $\parallel W \parallel ^{2} = \text{SUM}(\left| W[i,j]^2 \right|)$ # Sum of all `w` squared

  Ngooài ra, $\parallel W \parallel ^{2} = W^TW$  # $W$ is a vector

- Regularization for logistic regression:

  - Normal cost function: $J(w,b) = \frac{1}{m} \sum_{i=1}^{m}\mathcal{L(y_i, \hat{y}_i)}$
  - The L2 regularization version: $J(w,b) = \frac{1}{m} \sum_{i=1}^{m}\mathcal{L(y_i, \hat{y}_i)} + \frac{\lambda}{2m}\sum_{i=1}^{m}w_i^2$
  - The L1 regularization version: $J(w,b) = \frac{1}{m} \sum_{i=1}^{m}\mathcal{L(y_i, \hat{y}_i)} + \frac{\lambda}{2m}\sum_{i=1}^{m}|w_i|$
  - L1 regularization làm cho rất nhiều các $w$ tiến đến 0, do đó làm cho model size nhỏ hơn.
  - L2 thường xuyên được sử dụng hơn, nó không đưa $w$ về không, nhưng nó đưa $w$ tiến gần đến 0
  - $\lambda$ làm một **hyperparameter**

- Regularization for Neural Network:
  - Normal cost function: $J(W_1,b_1, ... , W_L, b_L) = \frac{1}{m} \sum_{i=1}^{m}\mathcal{L(y_i, \hat{y}_i)}$
  
  - The L2 regularization version: $J(W_1,b_1, ... , W_L, b_L) = \frac{1}{m} \sum_{i=1}^{m}\mathcal{L(y_i, \hat{y}_i)} + \frac{\lambda}{2m}\sum_{i=1}^{m}w_i^2$
  
  - Backpropagation in old way: `dw[1] = (from back propagation)`
  
  - In new way: `d[w1] = (from back propagation) + lambda/m * w[1] `
  
  - So, plugin in weight update step
    $$
    \begin{array}{rcl}
    W[1] & = & W[1] - \text{learning rate} * dw[1] \\
     & = & W[1] -  \text{learning rate} * ((\text{from back propagation}) + \frac{\lambda}{m} * w[1])\\
     & = & (1 - \text{learning rate} * \frac{\lambda}{m}) * w[1] - \text{learning rate} * (\text{from back propagation})
    \end{array}
    $$
    

### Why regularization reduces overfitting?

- Intution 1:
  - Nếu $\lambda$ quá lớn, rất nhiều $w$ sẽ tiến đến 0, điều này sẽ làm cho mạng NN đơn giản hơn (Nó sẽ gần gần đến với hồi quy logistic)
  - Nếu $\lambda$ đủ tốt, nó sẽ chỉ giảm một số trọng số nhất định khiến mạng NN không bị quá tải, cũng như tránh Overfitting

- Intution 2 (Với Tanh Activation):
  - Nếu $\lambda$ quá lớn, $w$ sẽ nhỏ gần bằng 0, do đó sẽ sử dụng phần tuyến tính của Tanh Activation, do đó chúng sẽ dần chuyển từ kích hoạt không tuyến tính sang kích hoạt gần tuyến tính, do đó làm cho NN trở thành một bộ phân loại gần như Tuyến tính
  - Nếu $\lambda$ đủ tốt, nó sẽ tạo ra kích hoạt tanh gần tuyến tính ở một vài node, do đó nó ngăn cản đc việc overfiting

### Dropout Regularization

- Andrew Ng sử dụng L2 Regularization trong hầu hết các trường hợp
- Dropout regularization là việc loại bỏ một số unit trong 1 layer dựa vào xác suất
- Một kỹ thuật phổ biến được dùng đến là "Inverted dropout"

```python
keep_prob = 0.8   # 0 <= keep_prob <= 1
l = 3  # this code is only for layer 3
# the generated number that are less than 0.8 will be dropped. 80% stay, 20% dropped
d3 = np.random.rand(a[l].shape[0], a[l].shape[1]) < keep_prob

a3 = np.multiply(a3,d3)   # keep only the values in d3

# increase a3 to not reduce the expected value of output
# (ensures that the expected value of a3 remains the same) - to solve the scaling problem
a3 = a3 / keep_prob       
```

- `d[l]` cho cả forward and backward propagation trong 1 vòng lặp là giống nhau, nhưng giữa các vòng lặp, `d[l]` sẽ khác nhau
- Tại thời điểm test, không sử dụng dropout, nếu implement dropdown tại thời điểm test, nó sẽ bị nhiễu cho các dự đoán.

### Understanding Dropout
- Dropout sẽ ngẫu nhiên loại bỏ một số đơn vị trong NN, do đó với mỗi lần lặp lại, bạn sẽ làm việc với một NN nhỏ hơn, nó sẽ có tác dụng tốt
- Vì trên mỗi layer, việc loại bỏ các unit là ngẫu nhiên, do đó weight sẽ phải dàn trải, mà không phụ thuộc hoàn toàn vào một weight nào.
- Mỗi layer, có thể có `keep_prob` là khác nhau, nên số unit được giữ lại ở mỗi lớp có thể là khác nhau
- Ở layer đầu tiên,  `keep_prob` phải là 1 hoặc gần 1, vì dĩ nhiên là chúng ta không muốn bỏ đi các features
- Các nhà nghiênc cứu thường dung Dropout với computer vision vì chúng có kích thước đàu vào rất lớn và hầu như không bao giờ đủ dữ liệu, do đó overfit là vấn đề thường gặp.
- Một nhược điểm của Dropout là cost function không xác định rõ (không biết những unit nào sẽ được bỏ đi trong 1 vòng lặp) do đó khó gỡ lỗi
  - Giải quyết vấn đề này, thì bạn nên tắt dropout (đặt `keep_prob = 1`) sau đó chạy test xem J có giảm một cách đơn điệu hay không.

### Một số phương pháp Regularization khác

- Data augmentation - Tăng dữ liệu
  - Ví dụ, dữ liệu trong Computer Vision
    - Bạn có thể lật tất cả các ảnh theo chiều ngang, điều này sẽ giúp bạn có nhiều dữ liệu hơn
    - Bạn cũng có thể áp dụng một vị trí và xoay ngẫu nhiên để có nhiều dữ liệu hơn
  - Ví dụ trong OCR, bạn có thể đặt các phép quay, phép biến dạng ngẫu nhiên cho các chữ số, chữ cái
  - Dữ liệu mới này có thể sẽ không tốt bằng dữ liệu độc lập thực, nhưng vẫn có thể được dùng như một kỹ thuật regularization

- Early Stopping
![](images/early_stopping.png)

---- https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/2-%20Improving%20Deep%20Neural%20Networks#dropout-regularization ---







## Code

### Khởi tạo trọng số.

- Khởi tạo ngẫu nhiên

```python
def initialize_parameters_random(layers_dims):
    """
    Đối số:
    layer_dims -- mảng python (list) chứa kích thước của mỗi lớp.
    
    Trả về:
    parameters -- dictionary của python chứa các tham số "W1", "b1", ..., "WL", "bL":
                    W1 -- ma trận trọng số có shape (layers_dims[1], layers_dims[0])
                    b1 -- vectơ bias có shape (layers_dims[1], 1)
                    ...
                    WL -- ma trận trọng số có shape (layers_dims[L], layers_dims[L-1])
                    bL -- vectơ bias có shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)               # seed này đảm bảo các số là ngẫu nhiên
    parameters = {}
    L = len(layers_dims)            # số nguyên thể hiện số lớp
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters
```

- Khởi tạo HE

```python
def initialize_parameters_he(layers_dims):
    """
    Đối số:
    layer_dims -- mảng python (list) chứa kích thước của mỗi lớp.
    
    Trả về:
    parameters -- dictionary của python chứa các tham số "W1", "b1", ..., "WL", "bL":
                    W1 -- ma trận trọng số có shape (layers_dims[1], layers_dims[0])
                    b1 -- vectơ bias có shape (layers_dims[1], 1)
                    ...
                    WL -- ma trận trọng số có shape (layers_dims[L], layers_dims[L-1])
                    bL -- vectơ bias có shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # số nguyên thể hiện số lớp
     
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
    return parameters
```



### Regularization

```python
def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    """
   Triển khai mạng nơ-ron 3 lớp: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Đối số:
    X -- dữ liệu đầu vào có shape (kích thước đầu vào, số ví dụ)
    Y -- vectơ true "label" (1 cho chấm xanh / 0 cho chấm đỏ), có shape (kích thước đầu ra, số ví dụ)
    learning_rate -- tốc độ học của tối ưu hóa
    num_iterations -- số lần lặp của vòng lặp tối ưu
    print_cost -- nếu True, in ra cost sau mỗi 100 lần lặp
    lambd -- siêu tham số điều chuẩn, số vô hướng 
    keep_prob - xác suất duy trì một nơ-ron hoạt động trong khi drop-out, số vô hướng.
    
    Trả về:
    parameters -- các tham sô mà mô hình đã tìm hiểu. Có thể sử dụng để dự đoán sau đó.
    """
        
    grads = {}
    costs = []                            # theo dõi cost
    m = X.shape[1]                        # số ví dụ
    layers_dims = [X.shape[0], 20, 3, 1]
    
    # Khởi tạo dictionary parameters.
    parameters = initialize_parameters(layers_dims)

    # Vòng lặp (gradient descent)

    for i in range(0, num_iterations):

        # Lan truyền xuôi: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        
        # Hàm chi phí
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
            
        # Lan truyền ngược.
        assert(lambd==0 or keep_prob==1)    # có thể sử dụng cả L2 regularization và dropout, 
                                            # nhưng ở lab này, chúng ta sẽ chỉ khám phá lần lượt từng cái
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
        # Cập nhật tham số.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # In ra loss sau mỗi 1000 lần lặp
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
    
    # vẽ biểu đồ của cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
```

- L2 Regularization

Cách tiêu chuẩn để tránh overfitting được gọi là **L2 regularization**. Nó bao gồm việc sửa đổi một cách thích hợp hàm mất mát, từ:

$$J = -\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small  y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} \tag{1}$$

Tới:

$$J_{regularized} = \small \underbrace{-\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} }_\text{cross-entropy cost} + \underbrace{\frac{1}{m} \frac{\lambda}{2} \sum\limits_l\sum\limits_k\sum\limits_j W_{k,j}^{[l]2} }_\text{L2 regularization cost} \tag{2}$$

```python
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Triển khai hàm chi phí với L2 regularization. Xem công thức (2) ở trên.
    
    Đối số:
    A3 -- hậu kích hoạt, đầu ra của lan truyền xuôi có shape (kích thước đầu ra, số ví dụ)
    Y -- vectơ "true" label, có shape (kích thước đầu ra, số ví dụ)
    parameters -- dictionary của python chứa các tham số của mô hình 
    
    Trả về:
    cost - giá trị của hàm mất mát có điều chuẩn (công thức (2))
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = compute_cost(A3, Y) # cho phần cross-entropy của cost
   
    L2_regularization_cost = (lambd / (2 * m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
   
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost
```

**Backward propagation** : Các thay đổi chỉ liên quan đến dW1, dW2 và dW3. Đối với mỗi thứ, bạn phải thêm gradient của số hạng regularization 

($\frac{d}{dW} ( \frac{1}{2}\frac{\lambda}{m}  W^2) = \frac{\lambda}{m} W$).

```python
def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Triển khai lan truyền ngược của mô hình cơ sở đã thêm L2 regularization.
    
    Đối số:
    X -- tập dữ liệu đầu vào có shape (kích thước đầu vào, số ví dụ)
    Y -- vectơ "true" có shape (kích thước đầu ra, số ví dụ)
    cache -- đầu ra cache từ forward_propagation()
    lambd -- siêu tham số điều chuẩn, số vô hướng
    
    Trả về:
    gradients -- Một dictionary với gradient descent liên quan tới từng tham số, biến kích hoạt và tiền kích hoạt 
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y

    dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd / m) * W3
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd / m) * W2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T) + (lambd / m) * W1
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
```



### Dropout

Cuối cùng, **dropout** là một kỹ thuật điều chuẩn dành riêng cho deep learning.
**Nó sẽ tắt ngẫu nhiên một số nơ-ron trong mỗi lần lặp lại.** Hãy xem 2 video sau để hiểu rõ!


To understand drop-out, consider this conversation with a friend:
- Friend: "Why do you need all these neurons to train your network and classify images?". 
- You: "Because each neuron contains a weight and can learn specific features/details/shape of an image. The more neurons I have, the more featurse my model learns!"
- Friend: "I see, but are you sure that your neurons are learning different features and not all the same features?"
- You: "Good point... Neurons in the same layer actually don't talk to each other. It should be definitly possible that they learn the same image features/shapes/forms/details... which would be redundant. There should be a solution."


<center><video width="620" height="440" src="images/dropout1_kiank.mp4" type="video/mp4" controls></video></center>

<center> <u> Hình 2 </u>: Drop-out ở lớp ẩn thứ 2</center>
Tại mỗi lần lặp, bạn tắt (= đặt thành 0) từng nơ-ron của lớp với xác suất $1 - keep\_prob$ hoặc giữ nguyên với xác suất $keep\_prob$ (50% ở đây). Các nơ-ron bị loại bỏ không đóng góp vào việc huấn luyện trong cả quá trình truyền xuôi và truyền ngược của lặp lại.

<center><video width="620" height="440" src="images/dropout2_kiank.mp4" type="video/mp4" controls></video></center>

<center> <u> Hình 3 </u>: Drop-out ở lớp ẩn thứ nhất và thứ 3.</center>
Lớp thứ $1$: tắt trung bình 40% nơ-ron. Lớp thứ $3$: tắt trung bình 20% nơ-ron.

Khi đóng một số nơ-ron, bạn sẽ thực sự sửa đổi mô hình của mình. Ý tưởng đằng sau drop-out là ở mỗi lần lặp lại, bạn huấn luyện một mô hình khác nhau chỉ sử dụng một tập hợp con các nơ-ron. Nếu không có dropout, các nơ-ron trở nên ít nhạy cảm hơn với sự kích hoạt của một nơ-ron cụ thể khác, bởi vì nơ-ron khác đó có thể bị tắt bất cứ lúc nào.



Bạn muốn tắt một số nơ-ron ở lớp thứ nhất và lớp thứ hai. Để làm điều đó, bạn sẽ thực hiện 4 Bước:
1. Trong bài giảng, chúng ta đã tạo một biến $d^{[1]}$ có shape là $a^{[1]}$ sử dụng `np.random.rand()` để lấy ngẫu nhiên các số từ 0 đến 1. Ở đây, bạn sẽ sử dụng triển khai vector hóa, vì vậy hãy tạo ma trận ngẫu nhiên $D^{[1]} = [d^{[1](1)} d^{[1](2)} ... d^{[1](m)}] $ có cùng chiều với $A^{[1]}$.
2. Thiết lập mỗi mục nhập của $D^{[1]}$ thành 0 với xác suất (`1-keep_prob`) hoặc 1 với xác suất (`keep_prob`), bằng cách lập ngưỡng giá trị trong $D^{[1]}$ thích hợp. Gợi ý: đặt tất cả các mục của ma trận X thành 0 (nếu mục nhập nhỏ hơn 0,5) hoặc 1 (nếu mục nhập lớn hơn 0,5), bạn sẽ thực hiện: `X = (X < 0.5)`. Lưu ý rằng 0 và 1 tương ứng với False và True.
3. Đặt $A^{[1]}$ thành $A^{[1]} * D^{[1]}$. (Bạn có thể đang tắt một số nơ-ron). Bạn có thể coi $D^{[1]}$ như một mask, để khi nó được nhân với một ma trận khác, nó sẽ tắt một số giá trị.
4. Chia $A^{[1]}$ cho `keep_prob`. Làm như vậy sẽ bảo đảm rằng kết quả của cost cũng sẽ có giá trị mong đợi như khi không có drop-out. (Kỹ thuật này được gọi là inverted dropout.)

```python
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Triển khai lan truyền xuôi: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Đối số:
    X -- tập dữ liệu đầu vào, có shape (2, số ví dụ)
    parameters -- dictionary của python chứa các tham số "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- ma trận trọng số có shape (20, 2)
                    b1 -- vectơ bias có shape (20, 1)
                    W2 -- ma trận trọng số có shape (3, 20)
                    b2 -- vectơ bias có shape (3, 1)
                    W3 -- ma trận trọng số có shape (1, 3)
                    b3 -- vectơ bias có shape (1, 1)
    keep_prob - xác suất duy trì một nơ-ron hoạt động trong khi drop-out, số vô hướng
    
    Trả về:
    A3 -- giá trị kích hoạt cuối, đầu ra của lan truyền xuôi, có shape (1,1)
    cache -- tuple, thông tin được lưu trữ để tính toán lan truyền xuôi
    """
    
    np.random.seed(1)
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
 
    # Bước 1: khởi tạo ma trận D1 = np.random.rand(..., ...)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    # Bước 2: chuyển các entry của D1 thành 0 hoặc 1 (dùng keep_prob làm ngưỡng)                                       
    D1 = (D1 < keep_prob).astype(int)  
    # Bước 3: dừng các nơ-ron của A1                                       
    A1 = A1 * D1 
    # Bước 4: chia tỷ lệ giá trị các nơ-ron chưa dừng                                        
    A1 = A1 / keep_prob                                         

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = (D2 < keep_prob).astype(int)
    A2 = A2 * D2
    A2 = A2 / keep_prob
    
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache
```

Truyền ngược với dropout thực sự khá dễ dàng. Bạn sẽ phải thực hiện 2 bước:

1. Trước đây bạn đã đóng một số nơ-ron trong quá trình truyền ngược, bằng cách áp dụng mask $ D ^ {[1]} $ cho `A1`. Trong truyền ngược, bạn sẽ phải đóng các nơ-ron tương tự bằng cách áp dụng lại cùng mask $ D ^ {[1]} $ cho `dA1`.

2. Trong quá trình truyền xuôi, bạn đã chia `A1` cho `keep_prob`. Do dó trong truyền ngược, bạn sẽ phải chia lại `dA1` cho `keep_prob` (giải thích tính toán là nếu $ A ^ {[1]} $ được chia tỷ lệ bởi cùng `keep_prob`, thì đạo hàm của nó là $ dA ^ {[1 ]} $ cũng được chia tỷ lệ bởi cùng một `keep_prob`).

```python
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Triển khai lan truyên ngược của mô hình cơ sở đã thêm dropout.
    
    Đối số:
    X -- tập dữ liệu đầu vào, có shape (2, số ví dụ)
    Y -- vectơ "true" label, có shape (kích thước đầu ra, số ví dụ)
    cache -- đầu ra cache từ forward_propagation_with_dropout()
    keep_prob - xác suất duy trì một nơ-ron hoạt động trong khi drop-out, số vô hướng
    
    Trả về:
    gradients -- Một dictionary với gradient descent liên quan tới từng tham số, biến kích hoạt và tiền kích hoạt 
    """
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
    # Bước 1: Áp dụng mask D2 để dừng các nơ-ron tương tự như trong lan truyền xuôi
    dA2 = dA2 * D2   
    # Bước 2: chia tỷ lệ giá trị các nơ-ron chưa dừng           
    dA2 = dA2 / keep_prob              
    
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    
    # Bước 1: Áp dụng mask D1 để dừng các nơ-ron tương tự như trong lan truyền xuôi
    dA1 = dA1 * D1    
    # Bước 2: chia tỷ lệ giá trị các nơ-ron chưa dừng          
    dA1 = dA1 / keep_prob              
    
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
```

