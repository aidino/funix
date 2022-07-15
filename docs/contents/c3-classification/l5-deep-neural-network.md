# Deep neural network 

### Notations

<img src="images/deep-neural-network.png" style="zoom:33%;" />

- $L = 4$ : #number of layers
- $n^{[l]}$: # number of units in layer $l$
- $a^{[l]}$ : activation in layer $l$
- $a^{[l]} = g^{[l]}(z^{[l]})$: activation function lin layer $l$
- $W^{[l]} = $ weights for $z^{[l]}$
- $b^{[l]} = $ bias for $z^{[l]}$
- $x = a^{[0]}$
- $\hat{y} = a^{[L]}$



### Shapes

- shape of $W$ : $(n^{[l]}, n^{[l-1]})$
- shape of $b$ : $(n^{[l]}, 1)$
- shape of $dw$ = shape of $W$
- shape of $db$ = shape of $b$
- shape of $Z^{[l]}, A^{[l]}, dZ^{[l]}, dA^{[l]}: $ $(n^{[l]}, m)$



### Tại sao cần deep learning

Mạng nơron sâu tạo liên hệ với dữ liệu từ đơn giản tới phức tạp. Ở từng lớp, nó cố tạo quan hệ với lớp trước đó, ví dụ:

- Ứng dụng nhận diện khuôn mặt (Face recognition): Hình ảnh => Các cạnh => Các bộ phận trên khuôn mặt => Các khuôn mặt => Khuôn mặt mong muốn.
- Ứng dụng nhận diện âm thanh (Audio recognition): Âm thanh => Các đặc trưng âm thanh ở mức độ thấp (sss,bb) => Phonemes $m$ vị => Từ => Câu.

Các nhà nghiên cứu nơron cho rằng mạng nơron sâu “tư duy” như não bộ (đơn giản ⇒ phức tạp). 

Khi bắt đầu ứng dụng, chúng ta chưa cần bắt đầu trực tiếp bằng nhiều lớp ẩn. Hãy thử giải pháp đơn giản nhất (chẳng hạn: Hồi quy logistic) rồi thử mạng nơron nông,...

### Xây dựng các blocks cho Deep Neural Network

<img src="images/neural-network-blocks.jpg" alt="neural-network-blocks" style="zoom:100%;" />
$$
\large \color{purple} x = a^{[0]} \longrightarrow  \color{Red} z^{[1]} = w^{[1]}a^{[0]} + b^{[1]} \longrightarrow a^{[1]} = g^{[1]}(z^{[1]}) \color{Black} \to \cdots \to \color{green} z^{[l]} = w^{[l]}a^{[l-1]} + b^{[l]} \longrightarrow a^{[l]} = g^{[l]}(z^{[l]}) \color{black} \longrightarrow \mathcal{L}(y, a^{[l]})
$$
Tại layer $\large l: w^{[l]}, b^{[l]}$

- **Forward**

  - Input: $\large a^{[l-1]}$
  - Output: $\large a^{[l]}$
  - $\longrightarrow$ **cache:** $\large \color{red} a^{[l]}, z^{[l]}$ 

  $$
  \large\begin{array}{rcl}
  z^{[l]} & = & W^{[l]}a^{[l-1]} + b^{[l]} \\
  a^{[l]} & = & g^{[l]}(z^{[l]}) \\
  \end{array}
  $$

  

- **Backward**

  - Input: $\large da^{[l]}$
  - Output: $\large da^{[l-1]}, dw^{[l]}, db^{[l]}$

  Ta có: $\large da^{[l]} \text{ is short of } \frac{d\mathcal{L}(y, a^{[l]})}{da^{[l]}} $
  $$
  \Large \begin{array}{rcl}
  dz^{[l]} & = & \frac{d\mathcal{L}}{da^{[l]}} * \frac{da^{[l]}}{dz^{[l]}} \\
   & = & \color{red} da^{[l]}*g'^{[l]}(z^{[l]}) \\
  dw^{[l]} & = & \frac{d\mathcal{L}}{da^{[l]}} * \frac{da^{[l]}}{dz^{[l]}} * \frac{dz^{[l]}}{dw^{[l]}}\\
   & = & \color{red} dz^{[l]} * a^{[l-1]}\\
  db^{[l]} & = & \frac{d\mathcal{L}}{da^{[l]}} * \frac{da^{[l]}}{dz^{[l]}} * \frac{dz^{[l]}}{db^{[l]}}\\
   & = & \color{red} dz^{[l]}\\
  
  da^{[l-1]} & = & \frac{d\mathcal{L}}{da^{[l]}} * \frac{da^{[l]}}{dz^{[l]}} * \frac{dz^{[l]}}{da^{[l-1]}}\\
   & = & \color{red} dz^{[l]} * w^{[l]}\\
  \end{array}
  $$
  





### Code

#### Main flow

$ L $-lớp với cấu trúc sau: *[LINEAR -> RELU]$\times$(L-1) -> LINEAR -> SIGMOID*. Các hàm bạn cần và đầu vào là:

```python
def initialize_parameters_deep(layer_dims):
    ...
    return parameters 
def L_model_forward(X, parameters):
    ...
    return AL, caches
def compute_cost(AL, Y):
    ...
    return cost
def L_model_backward(AL, Y, caches):
    ...
    return grads
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
```

- **Detail**

```python
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
```

```python
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters
```



```python
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches
```



```python
def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

```



```python
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
```



```python
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters
```

- **Put it all together**

```python
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Triển khai mạng nơ-ron L lớp: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Đối số:
    X -- dữ liệu, mảng numpy có shape (số ví dụ, num_px * num_px * 3)
    Y -- vectơ true "label" (0 nếu là mèo, 1 nếu không phải mèo), có shape (1, số ví dụ)
    layers_dims -- danh sách chứa kích thước đầu vào và từng kích thước lớp có length (số lớp + 1).
    learning_rate -- tốc độ học của quy tắc cập nhật gradient descent
    num_iterations -- số lần lặp của vòng lặp tối ưu hóa
    print_cost -- nếu True, in ra cost sau mỗi 100 lần lặp
    
    Trả về:
    parameters -- các tham số mà mô hình đã tìm hiểu. Chúng sẽ được sử dụng sau để dự đoán.
    """

    np.random.seed(1)
    costs = []                         # theo dõi cost
    
    # Khởi tạo tham số.
    ### BẮT ĐẦU CODE Ở ĐÂY ###
    parameters = initialize_parameters_deep(layers_dims)
    ### KẾT THÚC CODE Ở ĐÂY ###
    
    # Vòng lặp (gradient descent)
    for i in range(0, num_iterations):

        # Lan truyền xuôi: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Tính cost.
        cost = compute_cost(AL, Y)
    
        # Lan truyền ngược.
        grads = L_model_backward(AL, Y, caches)
 
        # Cập nhật tham số.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # In ra cost sau mỗi 100 ví dụ
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # vẽ biểu đồ của cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
```



#### Helper functions

- Activation functions - Forward and backward propagation

```python
def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ
```

- Forward 

```python
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache
```



- Backward

```python
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db
```

- Predict

```python
def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p
```

- Print mistabled images

```python
def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))

```















