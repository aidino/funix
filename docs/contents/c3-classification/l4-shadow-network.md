# Shallow Neural Network

### Recap **Logistic Regression** 

![logistic-regession-chain](images/logistic-regession-chain.jpg)

- Chiều xuôi:  **Loss functions**

  - $z = w^{T}x + b$


  - $\hat{y} = a = \sigma(z)$


  - $\mathscr{L}(a, y) = -(y\log(a) + (1-y)\log(1-a))$

- Chiều ngược: **Gradient**
  - $d(a) = \frac{d \mathcal{L}}{da} = \frac{d }{da}(-(y\log(a) + (1-y)\log(1-a))) = -\frac{y}{a} + \frac{1-y}{1-a}$
  - $d(z) = \frac{d \mathcal{L}}{dz} = a -y$
  - $d(w_1) = x_1 * d(z)$
  - $d(w_2) = x_2 * d(z)$



### **Khái niệm**

> Neural network  (Artificial neural network - ANN hay neural network) là một **mô hình toán học** hay **mô hình tính toán** được xây dựng dựa trên các mạng neural sinh học. 
>
> Nó gồm có một nhóm các **node neural** nối với nhau, và xử lý thông tin bằng cách truyền theo các kết nối và tính giá trị mới tại các node (cách tiếp cận connectionism đối với tính toán). Trong nhiều trường hợp, Neural network là một **hệ thống thích ứng (adaptive system)** tự thay đổi cấu trúc của mình dựa trên các thông tin bên ngoài hay bên trong chảy qua mạng trong quá trình học.
>
> Trong thực tế sử dụng, nhiều mạng neural là các công cụ mô hình hóa dữ liệu thống kê **phi tuyến**. Chúng có thể được dùng để mô hình hóa các mối quan hệ phức tạp giữa dữ liệu vào và kết quả hoặc để tìm kiếm các dạng/mẫu trong dữ liệu.

![artificial_neural_network.svg](images/artificial_neural_network.svg.png)

Nguồn: [Wiki](https://vi.wikipedia.org/wiki/M%E1%BA%A1ng_th%E1%BA%A7n_kinh_nh%C3%A2n_t%E1%BA%A1o)



### Neuron

Neuron là đơn vị nguyên tử của neural network. Cho một đầu vào, nó sẽ cung cấp đầu ra, chuyển đổi cho lớp tiếp theo.

Một neuron có thể được coi là sự kết hợp của 2 thành phần.

![node-of-neural-network](images/node-of-neural-network.png)

- Phần 1: Tính toán $z$ dựa trên các đầu vào và trọng số
- Phần 2: Hàm kích hoạt (chi tiết ở phần sau)



### Hidden layer

Hidden layer là các lớp neuron nằm giữa input layer và output layer.

Shallow network thì thường chỉ có từ 1 đến 2 hidden layer

**Notation**

- Chỉ số trên $[i]$ biểu thị lớp ẩn số mấy, ví dụ: $z^{[1]}$ thuộc lớp ẩn thứ nhất

- $\textbf X$ là vector đầu vào

- $\textbf W^{[i]}_{j}$ là **weight** liên quan đến neuron $j$ thuộc lớp thứ $i$

- $\textbf b^{[i]}_{j}$ là **bias** liên quan đến neuron $j$ thuộc lớp thứ $i$

  

### Neural network

![basic-neural-network-with-weight](images/basic-neural-network-with-weight.jpg)



**Forward propagation**

Cho input **x**: 

- $z^{[1]} = \textbf W^{[1]}x + \textbf b^{[1]}$
- $a^{[1]} = \sigma(z^{[1]})$
- $z^{[2]} = \textbf W^{[2]}a^{[1]} + \textbf b^{[2]}$
- $a^{[2]} = \sigma(z^{[2]})$



**Vectorizing across multiple examples**

Ta có: 

$\textbf{X} = \begin{bmatrix}
\vdots  & \vdots  & \vdots & \vdots\\
 x^{(1)} & x^{(2)} & \cdots   & x^{(m)} \\
 \vdots & \vdots & \vdots & \vdots
\end{bmatrix}$

$\textbf{A}^{[1]} = \begin{bmatrix}
\vdots  & \vdots  & \vdots & \vdots\\
 a^{[1](1)} & a^{[1](2)} & \cdots   & a^{[1](m)} \\
 \vdots & \vdots & \vdots & \vdots
\end{bmatrix}$



$Z^{[1]} = W^{[1]}X + b^{[1]}$

$A^{[1]} = \sigma(Z^{[1]})$

$Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$

$A^{[2]} = \sigma(Z^{[2]})$



`for i = 1 to m:`

​		$z^{[1](i)} = W^{[1]}x^{i} + b^{[1]}$

​		$a^{[1](i)} = \sigma(z^{[1](i)})$

​		$z^{[2](i)} = W^{[2]}a^{[1](i)} + b^{[2]}$

​		$a^{[2](i)} = \sigma(z^{[2](i)})$



### Activation Functions







### Back propagation

![chain-neural-network](images/chain-neural-network.jpg)

- Lan truyền xuôi dùng để tính giá trị dự đoán, suy ra mất mát dựa vào Loss function: $\mathscr{L}(a, y) = -(y\log(a) + (1-y)\log(1-a))$
- Lan truyền ngược dùng để tính đạo hàm từng phần của Loss Function cho các trọng số ở mỗi Layer





