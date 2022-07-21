# Optimization algorithms

// câu hỏi: Andrew Ng có nói về việc size của mini batch trong phương pháp Mịni-Batch Gradient Decent cần phù hợp với CPU, GPU, cụ thể là như thế nào?

## 1 - Gradient Descent

Phương pháp tối ưu hóa đơn giản trong machine learning là gradient descent (GD). Khi bạn thực hiện các bước gradient đối với tất cả $ m $ mẫu dữ liệu (samples) trên mỗi bước, nó còn được gọi là Batch Gradient Descent.

$$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} \tag{1}$$

$$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} \tag{2}$$

```python
def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Cập nhật các tham số sử dụng gradient descent 1 bước
    
    Đối số:
    parameters -- dictionary của python chứa các tham số cần cập nhật:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- dictionary của python chứa các gradient để cập nhật từng tham số:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- tốc độ học, số vô hướng.
    
    Trả về:
    parameters -- dictionary của python chứa các tham số đã cập nhật 
    """

    L = len(parameters) // 2 # số lớp trong mạng nơ-ron

    # Cập nhật quy luật cho từng tham số
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters
```



## 2 - Mini-Batch Gradient descent

Có 2 bước:

- **Shuffle (Xáo trộn)**: Tạo phiên bản xáo trộn của tập huấn luyện (X, Y) như hình dưới đây. Mỗi cột X và Y đại diện cho một ví dụ huấn luyện. Lưu ý rằng xáo trộn ngẫu nhiên được thực hiện đồng bộ giữa X và Y. Như vậy, sau khi xáo trộn cột thứ $ i $ của X là ví dụ tương ứng với nhãn thứ $ i $ trong Y. Bước xáo trộn đảm bảo rằng các ví dụ sẽ được chia ngẫu nhiên thành các mini-batch khác nhau.



<img src="images/kiank_shuffle.png" style="width:550px;height:300px;">



- **Partition (Phân vùng)**: Phân vùng xáo trộn (X, Y) thành các mini-batch có kích thước `mini_batch_size` (ở đây là 64). Lưu ý rằng số lượng ví dụ huấn luyện không phải lúc nào cũng chia hết cho `mini_batch_size`. Mini-batch cuối cùng có thể nhỏ hơn, nhưng bạn không cần phải lo lắng về điều này. Khi mini-batch cuối nhỏ hơn toàn bộ `mini_batch_size` sẽ giống như sau:



<img src="images/kiank_partition.png" style="width:550px;height:300px;">





Lưu ý rằng mini-batch cuối cùng có thể nhỏ hơn `mini_batch_size=64`. Giả sử $\lfloor s \rfloor$ đại diện cho $ s $ được làm tròn xuống số nguyên gần nhất (đây là `math.floor(s)` trong Python). Nếu tổng số ví dụ không phải là bội số của `mini_batch_size=64` thì sẽ có $\lfloor \frac{m}{mini\_batch\_size}\rfloor$ mini-batch với đầy đủ 64 ví dụ và số ví dụ trong mini-batch cuối cùng sẽ là ($m-mini_\_batch_\_size \times \lfloor \frac{m}{mini\_batch\_size}\rfloor$). 

```python
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Tạo danh sách các minibatch ngẫu nhiên từ from (X, Y)
    
    Đối số:
    X -- dữ liệu đầu vào, có shape (kích thước đầu vào, số ví dụ)
    Y -- vectơ true "label" (1 cho chấm xanh / 0 cho chấm đỏ), có shape (1, số ví dụ)
    mini_batch_size -- kích thước của các mini-batch, số nguyên
    
    Trả về:
    mini_batches -- danh sách đồng bộ (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # đảm bảo các minibatch "ngẫu nhiên" như nhau
    m = X.shape[1]                  # số ví dụ huấn luyện
    mini_batches = []
        
    # Bước 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))
    #print ("shape of shuffled_Y: " + str(shuffled_Y.shape))

    # Bước 2: Partition (shuffled_X, shuffled_Y). Trừ trường hợp cuối.
    num_complete_minibatches = math.floor(m/mini_batch_size) # số minibatch của size mini_batch_size trong partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Xử lý trường hợp cuối  mini-batch cuối < mini_batch_size)
    if m % mini_batch_size != 0:
        
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
```

## 3 - Momentum

Vì mini-batch gradient descent cập nhật tham số chỉ sau khi nhìn thấy một tập hợp con các ví dụ, hướng của bản cập nhật có một số phương sai và do đó, đường dẫn được thực hiện bởi mini-batch gradient descent sẽ "dao động" theo hội tụ. Sử dụng momentum có thể làm giảm các dao động này.



Momentum tính đến các gradient trong quá khứ để cập nhật một cách trơn tru. Chúng ta sẽ lưu trữ 'hướng' của các gradient trước đó trong biến $ v $. Về mặt hình thức, đây sẽ là giá trị trung bình có trọng số theo cấp số nhân của gradient ở các bước trước đó. Bạn cũng có thể coi $ v $ là "vận tốc" của một quả bóng lăn xuống dốc, xây dựng tốc độ (và momentum) theo hướng của gradient/độ dốc của ngọn đồi.





<img src="images/opt_momentum.png" style="width:400px;height:250px;">

<center> <u><font color='red'>**Hình 3**</u><font color='red'>: Các mũi tên màu đỏ chỉ hướng được thực hiện bởi một bước của mini-batch gradient descent. Các điểm màu xanh dương hiển thị hướng của gradient (đối với mini-batch hiện tại) ở mỗi bước. Thay vì chỉ tuân theo gradient, chúng ta để gradient ảnh hưởng đến $ v $ và sau đó thực hiện một bước theo hướng của $v$.<br> <font color='black'> </center>

```python
v["dW" + str(l+1)] = ... *#(mảng numpy của 0 có shape tương tự như parameters["W" + str(l+1)])*
v["db" + str(l+1)] = ... *#(mảng numpy của 0 có shape tương tự như parameters["b" + str(l+1)])*
```



**Lưu ý** rằng trình lặp l bắt đầu từ 0 trong vòng lặp for trong khi các tham số đầu tiên là v ["dW1"] và v ["db1"] (đó là "1" ở chỉ số trên). Đây là lý do tại sao chúng ta chuyển từ l sang l + 1 trong vòng lặp `for`.

```python
def initialize_velocity(parameters):
    """
    Khởi tạo velocity làm dictionary của python với:
                - khóa: "dW1", "db1", ..., "dWL", "dbL" 
                - giá trị: mảng numpy của 0 có shape tương tự như các gradient/tham số tương đương.
    Đối số:
    parameters -- dictionary của python chứa các tham số.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Trả về:
    v -- dictionary của python chứa velocity hiện tại.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 2 # số lớp trong mạng nơ-ron
    v = {}
    
    # Khởi tạo velocity
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        
    return v
  
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Cập nhật tham số sử dụng Momentum
    
    Đối số:
    parameters -- dictionary của python chứa các tham số:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- dictionary của python chứa cac gradient cho từng tham số:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- dictionary của python chứa velocity hiện tại:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- siêu tham số momentum, số vô hướng
    learning_rate -- tốc độ học, số vô hướng
    
    Trả về:
    parameters -- dictionary của python chứa các tham số đã cập nhật
    v -- dictionary của python chứa các velocity đã cập nhật
    """

    L = len(parameters) // 2 # số lớp trong mạng nơ-ron
    
    # Cập nhật momentum cho từng tham số
    for l in range(L):
        
        # tính velocity
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1 - beta) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1 - beta) * grads["db" + str(l+1)]
        # cập nhật tham số
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]
        
    return parameters, v
```

## 4 - Adam

Adam là một trong những thuật toán tối ưu hóa hiệu quả nhất để huấn luyện mạng nơ-ron. Nó kết hợp các ý tưởng từ RMSProp (được mô tả trong bài giảng) và Momentum.

**Adam hoạt động như thế nào?**

1. Nó tính trung bình có trọng số theo cấp số nhân của các gradient trước và lưu trữ nó trong các biến $ v $ (trước bias correction) và $ v ^ {correction} $ (với bias correction).
2. Nó tính trung bình có trọng số theo cấp số nhân của các bình phương của các gradient trước và lưu trữ nó trong các biến $ s $ (trước bias correction) và $ s ^ {corrected} $ (với bias correction).
3. Nó cập nhật các thông số theo một hướng dựa trên việc kết hợp thông tin từ "1" và "2".

Quy tắc cập nhật là với $l = 1, ..., L$: 

$$\begin{cases}
v_{dW^{[l]}} = \beta_1 v_{dW^{[l]}} + (1 - \beta_1) \frac{\partial \mathcal{J} }{ \partial W^{[l]} } \\
v^{corrected}_{dW^{[l]}} = \frac{v_{dW^{[l]}}}{1 - (\beta_1)^t} \\
s_{dW^{[l]}} = \beta_2 s_{dW^{[l]}} + (1 - \beta_2) (\frac{\partial \mathcal{J} }{\partial W^{[l]} })^2 \\
s^{corrected}_{dW^{[l]}} = \frac{s_{dW^{[l]}}}{1 - (\beta_1)^t} \\
W^{[l]} = W^{[l]} - \alpha \frac{v^{corrected}_{dW^{[l]}}}{\sqrt{s^{corrected}_{dW^{[l]}}} + \varepsilon}
\end{cases}$$
trong đó:
- t đếm số bước đã thực hiện của Adam
- L là số lớp
- $\beta_1$ and $\beta_2$ là các hyperparameter kiểm soát 2 giá trị trung bình có trọng số theo cấp số nhân.
- $\alpha$ là learning rate
- $\varepsilon$ là một số rất nhỏ để tránh chia hết cho 0

Như thường lệ, chúng ta sẽ lưu trữ tất cả các tham số trong dictionary `parameters`  

```python
def initialize_adam(parameters) :
    """
    Khởi tạo v và s làm 2 dictionary của python với:
                - khóa: "dW1", "db1", ..., "dWL", "dbL" 
                - giá trị: mảng numpy của 0 có shape tương tự như các gradient/tham số tương đương.
    
    Đối số:
    parameters -- dictionary của python chứa các tham số.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Trả về: 
    v -- dictionary của python chứa trung bình có trọng số theo cấp số nhân của gradient
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- dictionary của python chứa trung bình có trọng số theo cấp số nhân của bình phương gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2 # số lớp trong mạng nơ-ron
    v = {}
    s = {}
    
    # Khởi tạo v, s. Đầu vào: "parameters". Đầu ra: "v, s".
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    
    return v, s
  
  
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Cập nhật tham số sử dụng Adam
    
    Đối số:
    parameters -- dictionary của python chứa các tham số:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- dictionary của python chứa các gradient cho từng tham số:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- biến Adam, trung bình động của gradient đầu tiên, python dictionary
    s -- biến Adam, trung bình động của gradient bình phương, python dictionary
    learning_rate -- tốc độ học, số vô hướng.
    beta1 -- Exponential decay hyperparameter cho các ước tính thời điểm đầu tiên
    beta2 -- Exponential decay hyperparameter cho các ước tính thời điểm thứ hai
    epsilon -- hyperparameter ngăn chia cho 0 trong cập nhật Adam

    Trả về:
    parameters -- dictionary của python chứa các tham số đã cập nhật 
    v -- biến Adam, trung bình động của gradient đầu tiên, python dictionary
    s -- biến Adam, trung bình động của gradient bình phương, python dictionary
    """
    
    L = len(parameters) // 2                 # số lớp trong mạng nơ-ron
    v_corrected = {}                         # Khởi tạo ước tính thời điểm đầu tiên, python dictionary
    s_corrected = {}                         # Khởi tạo ước tính thời thứ hai, python dictionary
    
    # Thực hiện cập nhật Adam trên tất cả các tham số
    for l in range(L):
        # Trung bình động của gradient. Đầu vào: "v, grads, beta1". Đầu ra: "v".
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]

        # Tính bias-corrected first moment estimate. Đầu vào: "v, beta1, t". Đầu ra: "v_corrected".
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1**t)

        # Trung bình động của bình phương gradient. Đầu vào: "s, grads, beta2". Đầu ra: "s".
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * grads["dW" + str(l+1)]**2
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * grads["db" + str(l+1)]**2

        # Tính bias-corrected second raw moment estimate. Đầu vào: "s, beta2, t". Đầu ra: "s_corrected".
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2**t)

        # Cập nhật tham số Đầu vào: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Đầu ra: "parameters".
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)

    return parameters, v, s
```



**Model**

```python
def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):
    """
    mô hình mạng nơ-ron 3 lớp có thể chạy ở các chế độ tối ưu khác nhau.
    
    Đối số:
    X -- dữ liệu đầu vào, có shape (2, số ví dụ)
    Y -- vectơ true "label" (1 cho chấm xanh / 0 cho chấm đỏ), có shape (1, số ví dụ)
    layers_dims -- python list, chứa kích thước của từng lớp
    learning_rate -- tốc độ học, số vô hướng.
    mini_batch_size -- kích thước của mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter cho ước tính gradient trước đó
    beta2 -- Exponential decay hyperparameter cho ước tính gradient bình phương trước đó
    epsilon -- hyperparameter ngăn chia cho 0 trong các cập nhật của Adam
    num_epochs -- số epoch
    print_cost -- True sẽ in ra cost sau mỗi 1000 epoch

    Trả về:
    parameters -- python ditionary chứa các tham số đã cập nhật
    """

    L = len(layers_dims)             # số lớp trong mạng nơ-ron
    costs = []                       # theo dõi cost
    t = 0                            # khởi tạo bộ đếm cần cho cập nhật Adam
    seed = 10                        # Với mục đích chấm điểm, các "random" minibatch của bạn cần tương tự với của chúng tôi
    
    # Khởi tạo tham số
    parameters = initialize_parameters(layers_dims)

    # Khởi tạo optimizer
    if optimizer == "gd":
        pass # không yêu cầu khởi tạo cho gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    # Vòng lặp tối ưu
    for i in range(num_epochs):
        
        # Xác định các random minibatch. Tăng seed để xáo trộn lại tập dữ liệu sau mỗi epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            # Chọn một minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Lan truyền xuôi
            a3, caches = forward_propagation(minibatch_X, parameters)

            # Tính cost
            cost = compute_cost(a3, minibatch_Y)

            # Lan truyền ngược
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Cập nhật tham số
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
        
        # In ra cost sau mỗi 1000 epoch
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
                
    # vẽ biểu đồ của cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
```

