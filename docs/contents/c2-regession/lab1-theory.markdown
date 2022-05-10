# Kỹ thuật hồi quy - Hồi quy một đầu vào

$$ y_{i} = w_{0} + w_{1}x_{i} + \varepsilon_{i} $$

- $y_{i}$ : Giá trị thực tế
- $w_{0}$ : intercept
- $w_{1}$ : slope
- $\varepsilon_{i}$ : Error

$$ RSS(w_{0}, w_{1}) = \sum_{i=1}^{N}(y_{i} - [w_{0} + w_{1}x_{i}])^{2} $$

- $RSS(w_{0}, w_{1})$ : Residual sum of square - Tổng bình phương phần dư

## Giải bài toán tối ưu $RSS$

- **Đạo hàm từng phần $RSS$**
  
$$
\nabla RSS(w_{0}, w_{1}) = \begin{bmatrix}
\frac{\partial Rss(w_{0}, w_{1})}{\partial w_{0}} \\
\frac{\partial Rss(w_{0}, w_{1})}{\partial w_{1}}
\end{bmatrix}

= \begin{bmatrix}
-2\sum_{i=1}^{N}(y_{i} - [w_{0} + w_{1}x_{i}]) \\
-2\sum_{i=1}^{N}(y_{i} - [w_{0} + w_{1}x_{i}]x_{i})
\end{bmatrix}
\begin{matrix}
(1) \\
(2)
\end{matrix}
$$

$\nabla RSS(w_{0}, w_{1})$ : Gradient

### Approach 1: Set $\nabla RSS(w_{0}, w_{1})=0$, Giải tìm $w_{0}$ và $w_{1}$

$$
(1) \Leftrightarrow  \sum_{i=1}^{N} y_{i} - \sum_{i=1}^{N}w_{0} - \sum_{i=1}^{N}w_{1}x_{i} = 0
$$

$$
\Leftrightarrow Nw_{0} = \sum_{i=1}^{N}y_{i} - w_{1}\sum_{i=1}^{N}x_{i}
$$

$$
\Leftrightarrow w_{0} = \frac{\sum_{i=1}^{N}y_{i}}{N} - \frac{w_{1}\sum_{i=1}^{N}x_{i}}{N} 
$$

Từ phương trình $(2)$ 

$$
(2)\Leftrightarrow  \sum_{i=1}^{N}x_{i}y_{i} - w_{0}\sum_{i=1}^{N}x_{i} - w_{1}\sum_{i=1}^{N}x_{i}^{2} = 0
$$

Thay $w_{0}$ 

$$
w_{1} = \frac{\sum_{i=1}^{N}x_{i}y_{i} - \frac{\sum_{i=1}^{N}y_{i}\sum_{i=1}^{N}x_{i} }{N}}{\sum_{i=1}^{N}x_{i}^{2} - \frac{\sum_{i=1}^{N}x_{i}\sum_{i=1}^{N}x_{i}}{N}}
$$

Note - Cần phải tính các đại lượng

- $\sum_{i=1}^{N}x_{i}$
- $\sum_{i=1}^{N}y_{i}$
- $\sum_{i=1}^{N}x_{i}y_{i}$
- $\sum_{i=1}^{N}x_{i}^{2}$

### Approach 2: Gradient decent