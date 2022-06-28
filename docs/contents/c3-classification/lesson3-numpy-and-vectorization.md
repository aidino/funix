# Numpy and Vectorization

## Vectorization

> - Đây là phương pháp nhằm loại bỏ các `for` loop trong code. Giúp chương trình chạy nhanh hơn.
> - Hàm `dot` thư viện NumPy sử dụng vectơ hóa mặc định. Vectơ hóa có thể thực hiện trong CPU hoặc GPU qua phép toán SIMD, nhưng nhanh hơn trong GPU.
> - Hãy tránh vòng lặp for khi có thể, và phần lớn các phương thức thư viện NumPy là phiên bản vectơ hóa.

Ví dụ: Tính **Logistic regression derivatives**



Từ công thức để tính đạo hàm:

$d(a) = \frac{d \mathcal{L}}{da} = \frac{d }{da}(-(y\log(a) + (1-y)\log(1-a))) = -\frac{y}{a} + \frac{1-y}{1-a}$

$d(z) = \frac{d \mathcal{L}}{dz} = a -y$

$d(w_1) = x_1 * d(z)$

$d(w_2) = x_2 * d(z)$

$d(b) = d(z)$

- <mark><strong>Non-Vectorization</strong></mark>

```python
# Đạo hàm
J = 0; dw1 = 0; dw2 = 0; db = 0;
for i = 1 to m
    # Forward
    z(i) = w1*x1(i) + w2*x2(i) + b
    a(i) = Sigmoid(z(i))
    J += -(y(i)*log(a(i)) + (1-y(i))*log(1-a(i)))

    # Backward pass
    dz(i) = a(i) - y(i)
    dw1 += dz(i) * x1(i)
    dw2 += dz(i) * x2(i)
    db += dz(i)

J = J/m
dw1 = dw1/m
dw2 = dw2/m
db = db/m
```

- <mark><strong>Vectorization</strong></mark>

Chúng ta  có đầu vào là ma trận 

- $\textbf{X} \in \mathrm{R}^{m\times n}$
- $ \textbf{y} \in \mathrm{R}^{m\times 1}$ 
- $\textbf{z} = \textbf{w}^{T}\times \textbf{X} + \textbf{b}$

```python
z = np.dot(w.T, X)
A = 1/(1 + np.exp(-z))

```

