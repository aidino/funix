# Gradient Decent với Logistic regression

## Binary Classification



- $(x, y)  \left \{
    \begin{aligned}
      &x \in \mathrm{R}^{n} \\
      &y \in \{0,1\} \\
    \end{aligned} \right.$
- $m$ training example : $\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ..., (x^{(m)}, y^{(m)})\}$
- $\textbf{X} = \begin{bmatrix}
  \vdots  & \vdots  & \vdots & \vdots\\
   x^{(1)} & x^{(2)} & \cdots   & x^{(m)} \\
   \vdots & \vdots & \vdots & \vdots
  \end{bmatrix} \in \mathbb{R}^{(n\ast m)}$
- $\textbf{X}.shape = (n, m)$
- $\textbf{y} = [y^{(1)}, y^{(2)} ... , y^{(m)}]$
- $\textbf{y}.shape = (1,m)$

**Note**: trong ký hiệu này thì, mỗi cột là một observation. Trong một số tài liệu, hoặc theo những gì thường được ký hiệu trong các phần trước, thường sẽ biểu diễn trong một ma trận $\mathbb{R}^{(m\ast n)}$ với $m$ hàng tương ứng với $m$ observations.

Nhưng trong tài liệu này sẽ biểu diễn ngược lại, điều đó sẽ thuận lợi hơn, dễ triển khai hơn trong neural network

## Logistic Regression

Given $\textbf{x}$ , want $\hat{y} = P(y=1 | x)$ - Xác suất để y = 1khi $x$ xảy ra.

-  $\textbf{x} \in\mathbb{R}^{n}$

- parameters: $\textbf{w} \in \mathbb{R}^{n}$, $b \in \mathbb{R}$

- $\hat{y} = \textbf{w}^{T}x + b$ 

  đây là một hàm tuyến tính của đầu vào $\textbf{x}$, trên thực tế thì đây là những gì bạn thực hiện trên hồi quy tuyến tính nhưng đây không phải là một thuật toán hiệu quả để phân loại nhị phân.

  Hơn nữa, $\hat{y}$ nhận giá trị từ $0 -> 1$, do đó, với Logistic regression, hàm tính $\hat{y}$ như sau:

$$
\hat{y} =\sigma(\textbf{w}^{T}x + b)
$$

Đây gọi là hàm **sigmoid**

![sigmoid-graph](images/sigmoid-graph.png)

<center>Đồ thị hàm <b>sigmoid</b></center>

Công thức hàm **sigmoid**
$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$
**Lưu ý:**

- Nếu $z$ cực lớn thì $e^{-z}$ tiến đến 0 do đó $\sigma(z)$ tiến đến 1.
- Nếu $z$ cực nhỏ thì $\sigma(z) = \frac{1}{1+\textbf{bignum}}\approx 0$



## Logistic Regression Cost function

Hàm số:
$$
\hat{y} = \sigma(w^{T}x + b)
$$
trong đó:  $\sigma(z) = \frac{1}{1+e^{-z}}$
$$
\Leftrightarrow  \hat{y} = \frac{1}{1 + e^{-(w^{T}x + b)}}
$$
**Given** $\{(x^{(1)}, y^{(1)}), ... , (x^{(m)}, y^{(m)}) \}$ **want** $\hat{y}^{(i)} \approx y^{(i)}$

Thông thường, Loss (error) function sẽ là: $\mathscr{L}(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^{2}$

Nhưng trong Logistic regression thì hàm này là hàm nồi (convex function) do đó tồn tại các cực tiểu cục bộ => không thể giải quyết được bài toán tối ưu.

Vậy trong Logistic regression, Loss function sẽ là:
$$
\mathscr{L}(\hat{y}, y) = -(y\log(\hat{y}) + (1-y)\log(1-\hat{y}))
$$
**Giải thích:**

* Nếu $y=1$
* 