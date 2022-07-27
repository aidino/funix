# Decision Tree

- [Phần 1](decision-trees-annotated.pdf)
  - Dự đoán khả năng chi trả nợ với DT
  - Ý tưởng đằng sau DT
  - Học tập từ dữ liệu với thuật toán DT
  - Thuật toán đệ quy tham lam
  - Huấn luyện ranh giới quyết định
  - Chọn đặc trưng tốt nhất để phân chia
  - Khi nào thì dừng đệ quy
  - Dự đoán kết quả với DT
  - Phân loại đa lớp với DT
  - Hướng dẫn phân chia cho các đầu vào liên tục
  - (THAM KHẢO) Chọn ngưỡng tốt nhất để phân chia
  - Trực quan hoá danh giới quyết định

- [Phần 2](decision-trees-overfitting-annotated.pdf)

  - Nhắc lại về quá khớp

  - Quá khớp trong DT

  - Lý thuyết dao cạo OCCAM: Huấn luyện các mô hình đơn giản hơn

  - Dừng sớm (early stopping) trong DT

  - Tóm tắt về Quá khớp và điều chuẩn trong DT

    

### Code

- Scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier(max_depth=6)
decision_tree_model.fit(train_X, train_y)

# Predict
decision_tree_model.predict(sample_validation_data)
decision_tree_model.predict_proba(sample_validation_data)

# Score
print (decision_tree_model.score(train_X, train_y))
print ('----------')
print (decision_tree_model.score(val_X, val_y))


```









