# SVM - Support Vector Machine

## Thuật toán SVM

Ý tưởng của SVM là tìm một siêu phẳng (hyper lane) để phân tách các điểm dữ liệu. Siêu phẳng này sẽ chia không gian thành các miền khác nhau và mỗi miền sẽ chứa một loại dữ liệu.

Vấn đề là có rất nhiều siêu phẳng, chúng ta phải chọn cái nào để tối ưu nhất? Siêu phẳng tối ưu mà chúng ta cần chọn là siêu phẳng phân tách có lề lớn nhất. Lý thuyết học máy đã chỉ ra rằng một siêu phẳng như vậy sẽ cực tiểu hóa giới hạn lỗi mắc phải.

## SVM Kernel

Chúng ta có thể tạo ranh giới quyết định phi tuyến rất phức tạp theo 2 cách: 

- Ánh xạ dữ liệu đến các chiều không gian nhiều chiều hơn
- Sử dụng các Kernel.

 ## Ánh xạ đến không gian nhiều chiều hơn

Ánh xạ dữ liệu đến không gian nhiều chiều hơn sẽ giúp bạn đưa đường phi tuyến phức tạp trở thành 1 siêu phẳng tuyến tính trong chiều không gian mới. Tuy nhiên, cách làm này không hiệu quả lắm về mặt tài nguyên tính toán.

## The Kernel tricks

Sử dụng các hàm Kernel mô tả **quan hệ giữa hai điểm dữ liệu bất kỳ** trong không gian mới, thay vì đi tính toán trực tiếp **từng điểm dữ liệu trong không gian nhiều chiều mới** sẽ giúp bạn có thể tạo ranh giới quyết định phi tuyến rất phức tạp.

## Các dạng hàm Kernel

- Gaussian RBF Kernel: 
  $$
  \Large K(\overrightarrow{x}, \overrightarrow{l^i}) = e^{-\frac{\parallel \overrightarrow{x} - \overrightarrow{l^i} \parallel ^2 }{2\sigma^2}}
  $$

- Sigmoid Kernel
  $$
  K(X,Y) = tanh(\gamma . Z^TY + r)
  $$
  
- Polynomial Kernel

$$
K(X,Y) = (\gamma.X^TY + r)^d, \gamma > 0
$$

## Code

**Xây dựng phân loại với scikit-learn**

```python
from sklearn.svm import LinearSVC
sentiment_clf = LinearSVC(C=1, random_state=0, max_iter=2000)
sentiment_clf.fit(sentiment_X_train, sentiment_y_train)

print ("***Sentiment result***")
print("Train accuracy: {}".format(sentiment_clf.score(sentiment_X_train, sentiment_y_train)))
print("Validation accuracy: {}".format(sentiment_clf.score(sentiment_X_valid, sentiment_y_valid)))
```



**Hyperparameter turning**

```python
from sklearn.model_selection import GridSearchCV

tuned_parameters = {
    "C": [1, 2, 5, 10, 20, 100]
}

tuned_sentiment_cls = GridSearchCV(
                        LinearSVC(C=1),
                        param_grid=tuned_parameters,
                        n_jobs=2,
                        verbose=1,
)

tuned_sentiment_cls.fit(sentiment_X_train, sentiment_y_train)

print ("***Sentiment result***")
print("Train accuracy: {}".format(tuned_sentiment_cls.score(sentiment_X_train, sentiment_y_train)))
print("Validation accuracy: {}".format(tuned_sentiment_cls.score(sentiment_X_valid, sentiment_y_valid)))
```

