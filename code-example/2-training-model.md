# Training

## 1. Regression

```python
# import libs
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
```

**cross validatioon**

```python
n_folds = 10
def rmse_cv(model, train, y):
    # Tạo danh sách các fold
    kf = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(train.values)
    # Tiến hành kiểm chứng chéo với metric là MSE
    mse= np.sqrt(-cross_val_score(model, train.values, np.log(y), scoring="neg_mean_squared_error", cv = kf))
    # Trả về mảng giá trị MSE của từng fold
    return(mse)
```



### 1.1 Ridge Regression

```python
model = Ridge(alpha = 1e+3, tol = 0.0001, random_state=0)
score = rmse_cv(model, train_lasso_selected, y)
print("Lasso Selection: Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```

### 1.2 Elastic Net Regression

```python
model = ElasticNet(alpha = 8, l1_ratio = 0.01, tol = 0.0001, random_state=0)
# Kiểm chứng chéo trên các tập dữ liệu đã được lựa chọn thuộc tính\
score = rmse_cv(model, train_lasso_selected, y)
print("Lasso Selection: Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```

### 1.3 Support Vector Regression

```python
model = SVR(kernel = 'sigmoid')
# Kiểm chứng chéo trên các tập dữ liệu đã được lựa chọn thuộc tính
score = rmse_cv(model, train_lasso_selected, y)
print("Lasso Selection: Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```

### 1.4 Decision Tree Regression

```python
model = DecisionTreeRegressor(random_state = 0)
# Kiểm chứng chéo trên các tập dữ liệu đã được lựa chọn thuộc tính
score = rmse_cv(model, train_lasso_selected, y)
print("Lasso Selection: Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```

### 1.5 Random Forest Regression

```python
model = RandomForestRegressor(n_estimators = 10, random_state = 0)
# Kiểm chứng chéo trên các tập dữ liệu đã được lựa chọn thuộc tính
score = rmse_cv(model, train_lasso_selected, y)
print("Lasso Selection: Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```



## 2. Classification

```python
#Import libs
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
```

### 2.1 Logistic Regression

```python
model = LogisticRegression()
```

### 2.2 K-Nearest Neighbors vote

```python
model = KNeighborsClassifier(n_neighbors=2)
```

### 2.3 Decision Tree

```python
model = DecisionTreeClassifier()
```

### 2.4 Random Forest

```python
model = RandomForestClassifier(n_estimators=100, random_state=10)
```

### 2.5 AdaboostClassifier

```python
model = AdaBoostClassifier()
```

### 2.6 Neural Network

```python
model = MLPClassifier(hidden_layer_sizes=(10, 5, ), activation='logistic', max_iter=300)
```



## 3. Save and Load model

```python
from joblib import dump, load

clf = MLPClassifier(hidden_layer_sizes=(100, 50, ), max_iter=300)
clf.fit(train, y_train)

#Save model
dump(clf, 'nn_model.joblib')

# Load model
clf_nn = load('nn_model.joblib')
```



## 4. Evalution matrix

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
```

### 4.1 Độ chính xác (Accuracy)

Tỷ lệ phần trăm các dự đoán đúng.

```python
print('Accuracy Random Forest test:', accuracy_score(y_test, rf.predict(X_test)))
print('Accuracy Logistic Regression test:', accuracy_score(y_test, logit.predict(X_test)))
```

### 4.2 Phần trăm nhóm thiểu số được phân loại đúng

```python
def return_minority_perc(y_true, y_pred):
    minority_total = np.sum(y_true)
    minority_correct = np.sum(np.where((y_true==1)&(y_pred==1),1,0))
    return minority_correct / minority_total *100
  
print('% minority correctly classified, Random Forest test:', return_minority_perc(y_test, rf.predict(X_test)))
```

### 4.3 Precision, Recall, F-measure, Support

- **Precision** = $\large \frac{TP}{TP + FP}$

- **Recall** = $\large \frac{TP}{TP+FN}$

- **F1** = $\Large 2 \times \frac{precision \times recall}{precision + recall}$ 

- **Support** = Số trường hợp ở mỗi lớp

**Precision**, **Recall** và **F-measure** phụ thuộc vào ngưỡng xác suất được sử dụng để xác định kết quả phân lớp.

```python
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
)
```

#### 4.3.1 Precision

```python
print('Precision Random Forest test:', precision_score(y_test, rf.predict(X_test)))
```

#### 4.3.2 Recall

```python
print('Recall Random Forest test:', recall_score(y_test, rf.predict(X_test), pos_label=1))
```

#### 4.3.3 F-measure/chỉ số F1 

```python
print('F-measure Random Forest test:', f1_score(y_test, rf.predict(X_test), pos_label=1))
```

#### 4.3.4 Tất cả phép đo - 1 hàm

```python
precision, recall, fscore, support = precision_recall_fscore_support(
    y_test, rf.predict(X_test), pos_label=1,)
```



### 4.4 Report

```python
from yellowbrick.classifier import (
    ClassificationReport,
    DiscriminationThreshold,)

visualizer = ClassificationReport(rf)
visualizer.fit(X_train, y_train)        # Khớp visualizer và mô hình
visualizer.score(X_test, y_test)        # Đánh giá mô hình trên dữ liệu kiểm tra
visualizer.show()              
```

#### 4.4.1 Precision và Recall với ngưỡng xác suất

```python
visualizer = DiscriminationThreshold(logit,
                                     n_trials=1,
                                     cv=0.5,
                                     argmax='fscore',
                                     random_state=0,
                                     is_fitted='auto',
                                     exclude = "queue_rate")

visualizer.fit(X_train, y_train)        # Khớp visualizer và mô hình
visualizer.score(X_test, y_test)        # Đánh giá mô hình trên dữ liệu kiểm tra
visualizer.show()
```

#### 4.4.2 Ma trận nhầm lẫn, false positive rate (FPR) và false negative rate (FNR)

- **FPR** = $\large \frac{FP}{TN + FP}$

- **FNR**= $\large \frac{FN}{FP + FN}$ 

Ma trận nhầm lẫn, **FPR** và **FNR** phụ thuộc vào ngưỡng xác suất được sử dụng để xác đinh kết quả phân lớp.

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, rf.predict(X_test), labels=[0,1])

tn, fp, fn, tp = confusion_matrix(y_test, rf.predict(X_test), labels=[0,1]).ravel()
FPR = fp / (fp + tn)
FNR = fn / (fn + tp)
print('False Positive Rate, Random Forests: ', FPR)
print('False Negative Rate, Random Forests: ', FNR)
```

#### 4.4.3 FPR và FNR với ngưỡng xác suất

```python
thresholds = np.linspace(0, 1, 100)
fpr_ls = []
fnr_ls = []

# lấy xác suất
probs = logit.predict_proba(X_test)[:,1]

for threshold in thresholds:   
    # lấy dự đoán lớp dựa trên ngưỡng
    preds = np.where(probs>= threshold, 1, 0)
    # lấy ma trận nhầm lẫn
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0,1]).ravel()
    #  lấy FRP và FNR
    FPR = fp / (fp + tn)
    FNR = fn / (fn + tp)
    fpr_ls.append(FPR)
    fnr_ls.append(FNR)
    
metrics = pd.concat([
    pd.Series(fpr_ls),
    pd.Series(fnr_ls)], axis=1)
metrics.columns = ['fpr', 'fnr']
metrics.index = thresholds

metrics.plot()
plt.xlabel('Probability Threshold')
plt.ylabel('FPR / FNR')
plt.title('FPR and FNR vs Discriminant Threshold')
```

#### 4.4.4 Precision-Recall Curve

```python
from sklearn.metrics import plot_precision_recall_curve
from yellowbrick.classifier import PrecisionRecallCurve

# Sklearn
rf_disp = plot_precision_recall_curve(rf, X_test, y_test)
logit_disp = plot_precision_recall_curve(logit, X_test, y_test)

ax = plt.gca()
rf_disp.plot(ax=ax, alpha=0.8)
logit_disp.plot(ax=ax, alpha=0.8)

# Yellowbrick
visualizer = PrecisionRecallCurve(rf, classes=[0, 1])
visualizer.fit(X_train, y_train)        # Khớp dữ liệu huấn luyện với visualizer
visualizer.score(X_test, y_test)        # Đánh giá mô hình trên dữ liệu kiểm tra
visualizer.show()   
```

#### 4.4.5 Độ chính xác cân bằng

```python
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
)

# Độ chính xác
print('Accuracy Baseline test: ', accuracy_score(y_test, y_test_base))
print('Accuracy Random Forest test:', accuracy_score(y_test, rf.predict(X_test)))
print('Accuracy Logistic Regression test:', accuracy_score(y_test, logit.predict(X_test)))

# Độ chính xác cân bằng
print('Balanced accuracy, Baseline test: ', balanced_accuracy_score(y_test, y_test_base))
print('Balanced accuracy, Random Forest test:', balanced_accuracy_score(y_test, rf.predict(X_test)))
print('Balanced accuracy, Regression test:',  balanced_accuracy_score(y_test, logit.predict(X_test)))

# Recall mỗi phân lớp
print('Recall, class 0 and 1: ', recall_score(y_test, y_test_base, labels=[0,1], average=None))
print('Recall, class 0 and 1:', recall_score(y_test, rf.predict(X_test), labels=[0,1], average=None))
print('Recall, class 0 and 1:',  recall_score(y_test, logit.predict(X_test), labels=[0,1], average=None))
```

#### 4.5.6 Recall ở mỗi phân lớp

```python
print('Recall, class 0 and 1: ', recall_score(y_test, y_test_base, labels=[0,1], average=None))
print('Recall, class 0 and 1:', recall_score(y_test, rf.predict(X_test), labels=[0,1], average=None))
print('Recall, class 0 and 1:',  recall_score(y_test, logit.predict(X_test), labels=[0,1], average=None))
```

```bash
Recall, class 0 and 1:  [1. 0.]
Recall, class 0 and 1: [0.99997692 0.60246914]
Recall, class 0 and 1: [0.99967683 0.71111111]
```

