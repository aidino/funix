# 1. Preprocessing

### 1.1 Missing Data

#### 1.1.1 Median/Mean imputation

```python
from feature_engine.imputation import MeanMedianImputer

#  Feature-Engine captures the numerical variables automatically
imputer = MeanMedianImputer(imputation_method='median')
imputer.fit(X_train)

# feature-engine returns a dataframe
tmp = imputer.transform(X_train)
tmp.head()

# Feature-Engine allows you to specify variable groups easily
imputer = MeanMedianImputer(imputation_method='mean',
                            variables=['LotFrontage', 'MasVnrArea'])
imputer.fit(X_train)
```



#### 1.1.2 Gán giá trị ngẫu nhiên

```python
# Feature-Engine captures the numerical variables automatically
imputer = ArbitraryNumberImputer(arbitrary_number = -999)

# Feature-engine allows you to specify variable groups easily
imputer = ArbitraryNumberImputer(arbitrary_number=-999,
                                 variables=['LotFrontage', 'MasVnrArea'])

# We can impute different variables with different numbers
imputer = ArbitraryNumberImputer( imputer_dict=
               {'LotFrontage': -999,'MasVnrArea': -999,'GarageYrBlt': -1})
```



#### 1.1.3 Gán giá trị cuối phân phối

```python
# Feature-Engine captures the numerical variables automatically
imputer = EndTailImputer(imputation_method='gaussian', tail='right')

# Feature-engine allows you to specify variable groups easily
imputer = EndTailImputer(imputation_method='iqr', tail='left',
                         variables=['LotFrontage', 'MasVnrArea'])

# Feature-engine can be used with the Scikit-learn pipeline
pipe = Pipeline([
('imputer_skewed',EndTailImputer(imputation_method='iqr',tail='right',variables=['GarageYrBlt','MasVnrArea'])),
('imputer_gaussian',EndTailImputer(imputation_method='gaussian',tail='right',variables=['LotFrontage'])),
])
pipe.fit(X_train)
tmp = pipe.transform(X_train)
```



#### 1.1.4 Gán hạng mục thường xuất hiện

```python
# Feature-Engine captures the numerical variables automatically
imputer = CategoricalImputer(imputation_method='frequent')

# Feature-engine allows you to specify variable groups easily
imputer =CategoricalImputer(
    imputation_method='frequent', variables=['BsmtQual'])
```



#### 1.1.5 Gán giá trị bị thiếu là một biến hạng mục mới

```python
# Feature-Engine captures the numerical variables automatically
imputer = CategoricalImputer()

# Feature-engine allows you to specify variable groups easily
imputer = CategoricalImputer(variables=['BsmtQual'])

# Feature-engine can be used with the Scikit-learn pipeline
pipe = Pipeline([
    ('imputer_mode', CategoricalImputer(imputation_method='frequent', variables=['BsmtQual'])),
    ('imputer_missing', CategoricalImputer(variables=['FireplaceQu'])),
])
```



#### 1.1.6 Chỉ số khuyết dữ liệu  (Missing Indicator)

```python
# Feature-Engine's missing indicator selects all variables by default
imputer = AddMissingIndicator(missing_only=True)

# Feature-engine allows you to specify variable groups easily
imputer = AddMissingIndicator(variables=['BsmtQual', 'FireplaceQu', 'LotFrontage'])

# Feature-engine can be used with the Scikit-learn pipeline
pipe = Pipeline([
    ('missing_ind', AddMissingIndicator()),
    ('imputer_mode', CategoricalImputer(imputation_method='frequent', variables=['FireplaceQu', 'BsmtQual'])),
    ('imputer_median', MeanMedianImputer(imputation_method='median', variables=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'])),
])
```



#### 1.1.7 Gán mẫu ngẫu nhiên

```python
# Feature-Engine Random Sampler captures all the variables by default
imputer = RandomSampleImputer(random_state = 29)
```



#### 1.1.8 Gán KNN

```python
# gán đa biến
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5,  # số lượng neighbour K
                     weights='distance', # hệ số trọng số
                     metric='nan_euclidean', # phép đo tìm neighbour
                     add_indicator=False) # thêm chỉ số dự báo 

imputer.fit(X_train)
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)

# sklearn trả về một mảng Numpy
# tạo một dataframe
train_t = pd.DataFrame(train_t, columns=X_train.columns)
test_t = pd.DataFrame(test_t, columns=X_test.columns)

# ============== Tự động tìm các tham số gán tốt nhất. ===============
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline(steps=[
    ('imputer', KNNImputer(
        n_neighbors=5,
        weights='distance',
        add_indicator=False)),
    
    ('scaler', StandardScaler()),
    ('regressor', Lasso(max_iter=2000)),
])

param_grid = {
    'imputer__n_neighbors': [3,5,10],
    'imputer__weights': ['uniform', 'distance'],
    'imputer__add_indicator': [True, False],
    'regressor__alpha': [10, 100, 200],
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_search.best_params_


```



#### 1.1.9 Gán đa biến chuỗi các phương trình (MICE)

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(
    estimator=BayesianRidge(), # bộ ước tính dự đoán NA 
    initial_strategy='mean', # cách NA được gán trong bước 1
    max_iter=10, # số chu kỳ
    imputation_order='ascending', # để gán các biến
    n_nearest_features=None, # liệu có giới hạn số yếu tố dự báo
    skip_complete=True, # liệu có bỏ qua các biến không có NA
    random_state=0,
)

imputer.fit(X_train)
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)

# =============== So sánh phép gán với các mô hình khác. ================
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

imputer_bayes = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=0)

imputer_knn = IterativeImputer(
    estimator=KNeighborsRegressor(n_neighbors=5),
    max_iter=10,
    random_state=0)

imputer_nonLin = IterativeImputer(
    estimator=DecisionTreeRegressor(max_features='sqrt', random_state=0),
    max_iter=10,
    random_state=0)

imputer_missForest = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=10, random_state=0),
    max_iter=10,
    random_state=0)

imputer_bayes.fit(X_train)
imputer_knn.fit(X_train)
imputer_nonLin.fit(X_train)
imputer_missForest.fit(X_train)

X_train_bayes = imputer_bayes.transform(X_train)
X_train_knn = imputer_knn.transform(X_train)
X_train_nonLin = imputer_nonLin.transform(X_train)
X_train_missForest = imputer_missForest.transform(X_train)

# biến đổi mảng numpy thành dataframe
X_train_bayes = pd.DataFrame(X_train_bayes, columns = predictors)
X_train_knn = pd.DataFrame(X_train_knn, columns = predictors)
X_train_nonLin = pd.DataFrame(X_train_nonLin, columns = predictors)
X_train_missForest = pd.DataFrame(X_train_missForest, columns = predictors)
```





#### 1.1.10 Automatic selection of best imputation technique with Sklearn

```python
import pandas as pd
import numpy as np

# import classes for imputation
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# import extra classes for modelling
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(0)

# load dataset with all the variables
data = pd.read_csv('../../lab6-13_dataset/house-price/houseprice.csv',)

# find categorical variables
# those of type 'Object' in the dataset
features_categorical = [c for c in data.columns if data[c].dtypes=='O']

# find numerical variables
# those different from object and also excluding the target SalePrice
features_numerical = [c for c in data.columns if data[c].dtypes!='O' and c !='SalePrice']

# separate intro train and test set
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('SalePrice', axis=1),  # just the features
    data['SalePrice'],  # the target
    test_size=0.3,  # the percentage of obs in the test set
    random_state=0)  # for reproducibility

# We create the preprocessing pipelines for both
# numerical and categorical data

# adapted from Scikit-learn code available here under BSD3 license:
# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numeric_transformer, features_numerical),
        ('categorical', categorical_transformer, features_categorical)])

# Note that to initialise the pipeline I pass any argument to the transformers.
# Those will be changed during the gridsearch below.

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', Lasso(max_iter=2000))])

# now we create the grid with all the parameters that we would like to test
param_grid = {
    'preprocessor__numerical__imputer__strategy': ['mean', 'median'],
    'preprocessor__categorical__imputer__strategy': ['most_frequent', 'constant'],
    'regressor__alpha': [10, 100, 200],
}

grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, scoring='r2')
# cv=3 is the cross-validation
# no_jobs =-1 indicates to use all available cpus
# scoring='r2' indicates to evaluate using the r squared

# for more details in the grid parameters visit:
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# and now we train over all the possible combinations of the parameters above
grid_search.fit(X_train, y_train)

# and we print the best score over the train set
print(("best linear regression from grid search: %.3f"
       % grid_search.score(X_train, y_train)))

# we can print the best estimator parameters like this
grid_search.best_estimator_

# and find the best fit parameters like this
grid_search.best_params_
```



### 1.2 Outliers

[read more](https://ngohongthai.github.io/funix/contents/c2-regession/lab-7-4-outliers.html)

#### 1.2.1 Trimming/Truncation

```python
from feature_engine.outliers import OutlierTrimmer

strimmer = OutlierTrimmer(capping_method='iqr', fold=1.5)
strimmer.fit(boston)
boston_trimmed = strimmer.transform(boston)
boston.shape, boston_trimmed.shape

# capping_method:'gaussian', 'iqr', 'quantiles'

# Gaussian limits:
# right tail: mean + 3* std
# left tail: mean - 3* std

# IQR limits:
# right tail: 75th quantile + 3* IQR
# left tail: 25th quantile - 3* IQR
# where IQR is the inter-quartile range: 75th quantile - 25th quantile.

# percentiles:
# right tail: 95th percentile
# left tail: 5th percentile
```

#### 1.2.2 Censoring/Capping.

> Censoring (Kiểm duyệt) hoặc Capping (Giới hạn) là giới hạn max/min của phân phối tại một giá trị bất kỳ. Nói cách khác, những giá trị lớn hơn hoặc nhỏ hơn các giá trị được xác định tùy ý đều được kiểm duyệt.
> Capping có thể thực hiện ở cả 2 đầu hoặc 1 đầu phân phối còn tùy thuộc vào biến và người dùng.



```python
from feature_engine.outliers import Winsorizer
from feature_engine.outliers import ArbitraryOutlierCapper

# Quy tắc tiệm cận IQR
windsoriser = Winsorizer(capping_method='iqr', # chọn iqr cho các ranh giới quy tắc IQR hoặc gaussian cho mean và std
                          tail='both', # giới hạn đuôi trái, phải hoặc cả 2
                          fold=1.5,
                          variables=['RM', 'LSTAT', 'CRIM'])
windsoriser.fit(boston)

# Phép xấp xỉ Gauss
windsoriser = Winsorizer(capping_method='gaussian', # chọn iqr cho các ranh giới quy tắc IQR hoặc gaussian cho mean và std
                          tail='both', # giới hạn đuôi trái, phải hoặc cả 2
                          fold=3,
                          variables=['RM', 'LSTAT', 'CRIM'])
windsoriser.fit(boston)

# Quantile
windsoriser = Winsorizer(capping_method='quantiles', # chọn từ iqr, gaussian hoặc quantiles
                          tail='both', # cap left, right or both tails giới hạn đuôi trái, phải hoặc cả 2
                          fold=0.05,
                          variables=['RM', 'LSTAT', 'CRIM'])
windsoriser.fit(boston)


# Giới hạn tùy ý
capper = ArbitraryOutlierCapper(max_capping_dict={'age': 50, 'fare': 200},
                                min_capping_dict=None)
capper.fit(data.fillna(0))


capper = ArbitraryOutlierCapper(max_capping_dict=None,
                                min_capping_dict={
                                    'age': 10,
                                    'fare': 100
                                })
capper.fit(data.fillna(0))


capper = ArbitraryOutlierCapper(max_capping_dict={
    'age': 50, 'fare': 200},
    min_capping_dict={
    'age': 10, 'fare': 100})
capper.fit(data.fillna(0))
```



### 1.3 Categories Encoding

#### 1.3.1 One Hot Encoding

```python
# Scikit-learn
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories='auto',
                       drop='first', # trả về k-1, sử dụng drop=false để trả về k biến giả
                       sparse=False,
                       handle_unknown='error') # giúp xử lý nhãn hiếm

encoder.fit(X_train.fillna('Missing'))
# quan sát các hạng mục đã tìm hiểu
encoder.categories_ # MỚI: trong bản phát hành mới nhất của Scikit-learn
# chúng ta có thể truy xuất tên các đặc trưng như sau:
encoder.get_feature_names()

# Feature-engine
from feature_engine.encoding import OneHotEncoder

ohe_enc = OneHotEncoder(
    top_categories=None,
    variables=['sex', 'embarked'],  # có thể chọn biến để mã hóa
    drop_last=True)  # trả về k-1, false trả về k

ohe_enc.fit(X_train.fillna('Missing'))
tmp = ohe_enc.transform(X_train.fillna('Missing'))

```



#### 1.3.2 One Hot Encoding of Frequent/Top Categories

```python
	ohe_enc = OneHotEncoder(
    top_categories=10,  # có thể thay đổi giá trị này để chọn nhiều hoặc ít biến hơn
    variables=['Neighborhood', 'Exterior1st', 'Exterior2nd'], # có thể lựa chọn biến để mã hóa
    drop_last=False)

ohe_enc.fit(X_train)
```



#### 1.3.3 Integer Encoding

>  Mã hóa số nguyên gồm việc thay thế các hạng mục bằng các chữ số từ 1 đến n (hoặc 0 đến n-1, tùy thuộc vào cách triển khai), trong đó n là số hạng mục riêng biệt của biến.

```python
# scikit-learn
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(X_train['Neighborhood'])
# Không may, LabelEncoder hoạt động với một biến tại thời điểm đó. Tuy nhiên, có một cách tự động điều này cho tất cả các biến hạng mục
d = defaultdict(LabelEncoder) # mã hóa biến
train_transformed = X_train.apply(lambda x: d[x.name].fit_transform(x))

# Sử dụng dictionary để mã hóa dữ liệu tương lai
test_transformed = X_test.apply(lambda x: d[x.name].transform(x))


# feature-engine
ordinal_enc = OrdinalEncoder(
    encoding_method='arbitrary',
    variables=['Neighborhood', 'Exterior1st', 'Exterior2nd'])
ordinal_enc.fit(X_train)
```



#### 1.3.4 Count or frequency encoding

> Trong mã hóa đếm, chúng ta thay thế các hạng mục bằng số lượng quan sát hiển thị hạng mục đó trong tập dữ liệu. Tương tự, chúng ta có thể thay thế hạng mục bằng tần số - hoặc tỷ lệ phần trăm - của các quan sát trong tập dữ liệu. Nghĩa là, nếu có 10 trong số 100 quan sát hiển thị blue thì chúng ta sẽ thay thế blue bằng 10 nếu thực hiện mã hóa đếm hoặc 0.1 nếu thay thế bằng tần số.

```python
	from feature_engine.encoding import CountFrequencyEncoder
count_enc = CountFrequencyEncoder(
    encoding_method='count', # để thực hiện tần số ==> encoding_method='frequency'
    variables=['Neighborhood', 'Exterior1st', 'Exterior2nd'])

count_enc.fit(X_train)
```



#### 1.3.5 Target guided encodings - Ordered ordinal encoding

>Sắp xếp các hạng mục theo mục tiêu là gán một số cho hạng mục từ 1 đến k, trong đó k là số hạng mục riêng biệt trong biến, nhưng việc đánh số này được báo bằng giá trị mean của mục tiêu cho từng hạng mục.
> Ví dụ, chúng ta có biến city với các giá trị London, Manchester và Bristol; nếu tỷ lệ mặc định là 30% ở London, 20% ở Bristol và 10% ở Manchester thì chúng ta thay London bằng 1, Bristol bằng 2 và Manchester bằng 3.

```python
from feature_engine.encoding import OrdinalEncoder
ordinal_enc = OrdinalEncoder(
    # LƯU Ý rằng chúng ta chỉ ra ordered trong encoding_method, nếu không thì nó sẽ gán số bất kỳ
    encoding_method='ordered',
    variables=['Neighborhood', 'Exterior1st', 'Exterior2nd'])
ordinal_enc.fit(X_train, y_train)
X_train = ordinal_enc.transform(X_train)
X_test = ordinal_enc.transform(X_test)
```



#### 1.3.6 Target guided encodings - Mean encoding

> Mã hóa trung bình là thay thế hạng mục theo gia trị mục tiêu trung bình cho hạng mục đó. Ví dụ: chúng ta có biến city với các hạng mục London, Manchester và Bristol; nếu chúng ta muốn dự đoán tỷ lệ mặc định, nếu tỷ lệ mặc định cho London là 30% thì thay London bằng 0.3, nếu tỷ lệ mặc định cho Manchester là 20% thì thay Manchester bằng 0.2,....

```python
from feature_engine.encoding import MeanEncoder
mean_enc = MeanEncoder(
    variables=['cabin', 'sex', 'embarked'])
mean_enc.fit(X_train, y_train)
```



#### 1.3.7 Target guided encodings - Probability Ratio Encoding

> Mã hóa này chỉ thích hợp với các bài toán phân loại có biến nhị phân.
>
> Với từng hạng mục, chúng ta tính mean của target=1 là xác suất của mục tiêu là 1 ( P(1) ), và xác suất của target=0 ( P(0) ). Sau đó, chúng ta tính tỷ lệ P(1)/P(0) và thay thế các hạng mục bằng tỷ lệ đó.

```python
from feature_engine.encoding import PRatioEncoder

ratio_enc = PRatioEncoder(
    encoding_method = 'ratio',
    variables=['cabin', 'sex', 'embarked'])
ratio_enc.fit(X_train, y_train)
ratio_enc.encoder_dict_
```



#### 1.3.8 Target guided encodings - Weight of evidence

> WoE = ln ( Distribution of Goods / Distribution of bads )
> WoE = ln ( p(1) / p(0) )
> Note: WoE is well suited for Logistic Regression

```python
from feature_engine.encoding import WoEEncoder as fe_WoEEncoder
from category_encoders.woe import WOEEncoder

woe_enc = fe_WoEEncoder(variables=['cabin', 'sex', 'embarked']) woe_enc.fit(X_train, y_train)

woe_enc = WOEEncoder(cols=['cabin', 'sex', 'embarked'])
woe_enc.fit(X_train, y_train)
```



#### 1.3.9 Rare Label Encoding

>Giá trị hiếm là các hạng mục trong một biến hạng mục chỉ xuất hiện trong tỷ lệ nhỏ các quan sát. Không có quy tắc chung nào để xác định thế nào là tỷ lệ phần trăm nhỏ, nhưng thông thường, bất kỳ giá trị nào dưới 5% đều có thể coi là hiếm.
>
>Scenario for re-grouping:
>\- Một hạng mục nổi bật
>\- Ít hạng mục
>\- Cardinality cao

```python
from feature_engine.encoding import RareLabelEncoder

rare_encoder = RareLabelEncoder(
    tol=0.05,  # % tối thiểu được coi là không hiếm
    n_categories=4, # số lượng hạng mục tối thiểu mà biến có để nhóm lại thành các hạng mục chiếm
    variables=['Neighborhood', 'Exterior1st', 'Exterior2nd',
               'MasVnrType', 'ExterQual', 'BsmtCond'] # các biến để nhóm lại
) 
rare_encoder.fit(X_train.fillna('Missing'))
X_train = rare_encoder.transform(X_train.fillna('Missing'))
X_test = rare_encoder.transform(X_test.fillna('Missing'))
```



### 1.4 Rời rạc hoá dữ liệu số

#### 1.4.1 Equal width discretisation (Rời rạc hóa sử dụng khoảng cách bằng nhau)

> Rời rạc hóa sử dụng khoảng cách bằng nhau chia phạm vi các giá trị thành N bin có cùng khoảng cách. Khoảng cách (width) được xác định bởi phạm vi giá trị trong biến và số bin mà chúng ta muốn sử dụng để chia biến:
>
> width = (max value - min value) / N
>
> trong đó N là số bin/khoảng.

```python
#scikit-learn
from sklearn.preprocessing import KBinsDiscretizer
disc = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
disc.fit(X_train[['age', 'fare']])

train_t = disc.transform(X_train[['age', 'fare']])
train_t = pd.DataFrame(train_t, columns = ['age', 'fare'])

test_t = disc.transform(X_test[['age', 'fare']])
test_t = pd.DataFrame(test_t, columns = ['age', 'fare'])

#feature-engine
from feature_engine.discretisation import EqualWidthDiscretiser

disc = EqualWidthDiscretiser(bins=8, variables = ['age', 'fare'])
disc.fit(X_train)
train_t = disc.transform(X_train)
test_t = disc.transform(X_test)
```



#### 1.4.2 Equal frequency discretisation (Rời rạc hóa sử dụng tần số bằng nhau)

>Rời rạc hóa sử dụng tần số bằng nhau chia phạm vi các giá trị có thể có của biến thành N bin, trong đó mỗi bin chứa cùng một lượng quan sát. Điều này đặc biệt hữu ích với các biến bị lệch vì nó trải đều các quan sát trên các bin khác nhau. Chúng ta tìm các ranh giới của khoảng bằng cách xác định quantile.
>
>Rời rạc hóa sử dụng tần số bằng nhau dùng quantile gồm việc chia biến liên tục thành N quantile; trong đó, N được người dùng xác định.

```python
#scikit-learn
from sklearn.preprocessing import KBinsDiscretizer
disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
disc.fit(X_train[['age', 'fare']]
         
train_t = disc.transform(X_train[['age', 'fare']])
train_t = pd.DataFrame(train_t, columns = ['age', 'fare'])
test_t = disc.transform(X_test[['age', 'fare']])
test_t = pd.DataFrame(test_t, columns = ['age', 'fare'])

#feature-engine
from feature_engine.discretisation import EqualFrequencyDiscretiser
disc = EqualFrequencyDiscretiser(q=10, variables = ['age', 'fare'])
disc.fit(X_train)
train_t = disc.transform(X_train)
test_t = disc.transform(X_test)

```



#### 1.4.3 Discretisation plus Encoding (Rời rạc hóa sử dụng mã hóa)

Chúng ta sẽ làm gì với biến sau khi rời rạc hóa? Có nên dùng bucket làm biến số không? Hay chúng ta nên sử dụng khoảng làm biến hạng mục?

Câu trả lời là, chúng ta có thể thực hiện một trong hai.

Nếu chúng ta đang xây dựng các thuật toán dựa trên **decision tree (DT)** và đầu ra của rời rạc hóa là **số nguyên** (mỗi số nguyên tham chiếu đến một bin), thì chúng ta có thể sử dụng chúng trực tiếp vì DT sẽ chọn ra các mối quan hệ phi tuyến tính giữa biến rời rạc hóa và mục tiêu.

Thay vào đó, nếu chúng ta đang xây dựng mô hình tuyến tính thì các bin không nhất thiết phải giữ mối quan hệ tuyến tính với mục tiêu. Trong trường hợp này, nó giúp cải thiện chất lượng mô hình, coi bin là hạng mục và mã hóa one-hot hoặc mã hóa có hướng dẫn mục tiêu như mã hóa trung bình, trọng số bằng chứng hoặc mã hóa thứ tự có hướng dẫn mục tiêu.

Chúng ta có thể dễ dàng thực hiện bằng cách kết hợp các bộ mã hóa và rời rạc hóa của feature-engine.

```python
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.encoding import OrdinalEncoder

disc = EqualFrequencyDiscretiser(
    q=10, variables=['age', 'fare'], return_object=True)
# tìm các khoảng
disc.fit(X_train)
# biến đổi train và test
train_t = disc.transform(X_train)
test_t = disc.transform(X_test)

enc = OrdinalEncoder(encoding_method = 'ordered')
enc.fit(train_t, y_train)
train_t = enc.transform(train_t)
test_t = enc.transform(test_t)
```





#### 1.4.4 Rời rạc hóa với Decision Tree 

Rời rạc hóa với Decision Tree (DT) sử dụng DT để xác định các bin tối ưu. Khi DT đưa ra quyết định, nó sẽ chỉ định một quan sát cho một trong n lá cuối. Do đó, bất kỳ DT nào cũng sẽ tạo một đầu ra rời rạc, mà các giá trị là các dự đoán tại mỗi lá trong số n lá của nó.

Làm thế nào để thực hiện rời rạc hóa với với cây?
- 1) Huấn luyện DT có độ sâu giới hạn (2, 3 hoặc 4) bằng cách sử dụng biến mà chúng ta muốn rời rạc hóa và mục tiêu.
- 2) Thay thế các giá trị bằng kết quả trả về của cây.

**Ưu điểm**

- Kết quả trả về của DT có quan hệ đơn điệu với mục tiêu.
- Các nút cuối của cây hoặc các bin trong biến rời rạc hóa cho thấy entropy giảm, nghĩa là các quan sát trong từng bin giống nhau hơn so với các quan sát của các bin khác.

**Hạn chế**

- Dễ bị over-fitting
- Quan trọng hơn là một số điều chỉnh tham số của cây cần thu được số lượng phân chia tối ưu (ví dụ: độ sâu của cây, số mẫu tối thiểu trong một phân vùng, số phân vùng tối đa và mức tăng thông tin tối thiểu). Điều này có thể tốn thời gian.

```python
from sklearn.tree import DecisionTreeClassifier

treeDisc = DecisionTreeDiscretiser(cv=10, scoring='accuracy',
                                   variables=['age', 'fare'],
                                   regression=False,
                                   param_grid={'max_depth': [1, 2, 3],
                                              'min_samples_leaf':[10,4]})

treeDisc.fit(X_train, y_train)
treeDisc.binner_dict_['age'].best_params_

train_t = treeDisc.transform(X_train)
test_t = treeDisc.transform(X_test)
```



### 1.5 Feature scaling

#### 1.5.1 Chuẩn tắc hóa (Standardisation) 

Chuẩn tắc hóa gồm căn giữa biến ở 0 và chuẩn tắc hóa phương sai thành 1. Quy trình là trừ đi mean của mỗi quan sát rồi chia cho độ lệch chuẩn:

$$
\Large z = \frac{x - x_{mean}}{std}
$$


Kết quả của phép biến đổi trên là **z**, được gọi là z-score thể hiện độ lệch chuẩn mà một quan sát nhất định lệch khỏi mean. z-score xác định vị trí của quan sát trong một phân phối (theo số lượng độ lệch chuẩn với giá trị trung bình của phân phối). Dấu của z-score (+ hoặc -) cho biết quan sát nằm trên (+) hay dưới (-) mean.


Hình dạng của phân phối chuẩn tắc hóa (hoặc chuẩn hóa z-score) sẽ giống với phân phối ban đầu của biến. Nếu phân phối ban đầu là chuẩn thì phân phối chuẩn tắc hóa sẽ là chuẩn. Nhưng nếu phân phối ban đầu bị lệch thì phân phối chuẩn tắc hóa của biến cũng sẽ bị lệch. Nói cách khác, **chuẩn tắc hóa một biến không chuẩn hóa phân phối của dữ liệu** và nếu đây là kết quả mong muốn, chúng ta nên thực hiện bất kỳ kỹ thuật nào được thảo luận trong phần 7 của khóa học.


Tóm lại, chuẩn tắc hóa:

- căn giữa mean ở 0
- co giãn phương sai ở 1
- duy trì hình dạng của phân phối ban đầu
- giá trị min/max của các biến khác nhau thay đổi
- duy trì outlier

Chuẩn tắc hóa tốt với các thuật toán yêu cầu đặc trung tập trung ở 0.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```



#### 1.5.2 Chuẩn hóa trung bình (Mean normalisation)

Chuẩn hóa trung bình gồm việc căn giữa các biến ở 0 và điều chỉnh lại phạm vi giá trị. Quy trình gồm việc trừ đi giá trị trung bình của mỗi quan sát rồi chia cho hiệu giữa $max$ và $min$:

$$
\Large x_{scaled} = \frac{x - x_{mean}}{x_{max}-x_{min}}
$$



Kết quả của phép biến đổi trên là một phân phối căn giữa 0 và min/max nằm trong phạm vi từ -1 đến 1. Hình dạng của phân phối chuẩn hóa trung bình sẽ tương tự như của phân phối ban đầu, nhưng phương sai có thể thay đổi nên sẽ không giống hệt nhau.


Một lần nữa, kỹ thuật này sẽ không **chuẩn hóa phân phối của dữ liệu**, do đó nếu đây là kết quả mong muốn, chúng ta nên thực hiện thêm bất kỳ kỹ thuật nào đã thảo luận trong phần 7 của khóa học.

Tóm lại, chuẩn hóa trung bình:

- tập trung mean ở 0
- variance will be different phương sai sẽ khác
- may alter the shape of the original distribution có thể thay đổi hình dạng của phân phối ban đầu
- min và max nằm trong khoảng từ -1 đến 1
- duy trì các outlier

Chuẩn hóa trung bình khá tốt với các thuật toán yêu cầu đặc trưng căn giữa ở 0.

```python
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler_mean = StandardScaler(with_mean=True, with_std=False)

# thiết lập robustscaler để nó KHÔNG loại median
# nhưng chuẩn hóa bằng max()-min(), quan trọng với
# phạm vi quantile từ 0-100, thể hiện min và max
scaler_minmax = RobustScaler(with_centering=False,
                             with_scaling=True,
                             quantile_range=(0, 100))

scaler_mean.fit(X_train)
scaler_minmax.fit(X_train)

# biến đổi tập huấn luyện và tập kiểm tra
X_train_scaled = scaler_minmax.transform(scaler_mean.transform(X_train))
X_test_scaled = scaler_minmax.transform(scaler_mean.transform(X_test))

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```



#### 1.5.3 Co giãn về min/max - MinMaxScaling

Co giãn về min-max các giá trị từ 0 đến 1. Nó trừ min từ tất cả các quan sát rồi chia cho phạm vi giá trị:

$$
\Large x_{scaled} = \frac{x - x_{mean}}{x_{max} - x_{min}}
$$



Kết quả của phép biến đổi trên là phân phối có các giá trị thay đổi trong phạm vi từ 0 đến 1. Nhưng giá trị trung bình không tập trung ở 0 và độ lệch chuẩn cũng thay đổi trong các biến. Hình dạng của phân phối khi co giãn max/min sẽ tương tự như phân phối ban đầu, nhưng phương sai có thể thay đổi nên chúng sẽ không giống nhau. Kỹ thuật co giãn này cũng nhạy với các outlier.

Kỹ thuật này sẽ không **chuẩn hóa phân phối của dữ liệu** do đó nếu đây là kết quả mong muốn, chúng ta nên thực hiện bất kỳ kỹ thuật nào đã thảo luận trong phần 7 của khóa học.

Tóm lại, MinMaxScaling:

- không tập trung mean ở 0
- phương sai thay đổi trên các biến
- có thể không duy trì hình dạng của phân phối ban đầu
- các giá trị min và max là 0  và 1 
- nhạy với outlier
  
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# khớp scaler với tập huấn luyện, nó sẽ học các tham số
scaler.fit(X_train)

# biến đổi tập huấn luyện và tập kiểm tra
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```



#### 1.5.4 Co giãn về giá trị lớn nhất tuyệt đối - MaxAbsScaling

MaxAbsScaling co giãn dữ liệu thành giá trị tuyệt đối lớn nhất.

$$
\Large x_{scaled} = \frac{x}{abs(x_{max})}
$$


Kết quả của phép biến đổi trên là phân phối có các giá trị thay đổi trong khoảng từ -1 đến 1, nhưng giá trị trung bình không căn giữa ở 0 và độ lệch chuẩn thay đổi trên các biến.

Scikit-learn gợi ý sử dụng transformer có ý nghĩa với dữ liệu, với các dữ liệu thưa thớt và căn giữa ở 0.


Tóm lại, MaxAbsScaling:

- Không căn giữa ở 0 (nhưng có thể với một phương pháp khác)
- phương sai thay đổi trên các biến
- không giữ hình dạng của phân phối ban đầu
- nhạy với outlier
  
```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```



#### 1.5.5 Căn giữa + MaxAbsScaling

Chúng ta có thể căn giữa các phân phối ở 0 rồi co lại thành giá trị tuyệt đối lơn nhất, như Scikit-learn gợi ý bằng cách kết hợp sử dụng 2 transformer.

```python
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

# thiết lập StandardScaler để nó loại bỏ mean
# nhưng không chia cho độ lệch chuẩn
scaler_mean = StandardScaler(with_mean=True, with_std=False)

# thiết lập MaxAbsScaler chuẩn hóa
scaler_maxabs = MaxAbsScaler()

# khớp scaler với tập huấn luyện để nó học các tham số
scaler_mean.fit(X_train)
scaler_maxabs.fit(X_train)

# biến đổi tập huấn luyện và tập kiểm tra
X_train_scaled = scaler_maxabs.transform(scaler_mean.transform(X_train))
X_test_scaled = scaler_maxabs.transform(scaler_mean.transform(X_test))

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```



#### 1.5.6 Co giãn về trung vị và phân vị - RobustScaling

Trong quy trình này, median bị loại khỏi các quan sát rồi bị co lại lệ thành IQR. IQR là phạm vi giữa quartile thứ nhất (quantile thứ 25) và quartile thứ ba (quantile thứ 75).

$$
\Large x_{scaled} = \frac{x - x_{median}}{x_{quantile(0.75)} - x_{quantile(0.25)}}
$$


Phương pháp RobustScaling này tạo ra các ước tính mạnh mẽ hơn cho trung tâm và phạm vi của biến, và được khuyến nghị dùng nếu dữ liệu hiển thị outlier.

Tóm lại, RobustScaling:

- căn giữa ở 0
- phương sai thay đổi trên các biến
- không giữ hình dạng của phân phối ban đầu.
- các giá trị min, max thay đổi
- outlier mạnh mẽ

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

# khớp scaler với tập huấn luyện, nó sẽ học các tham số
scaler.fit(X_train)

# biến đổi tập huấn luyện và tập kiểm tra
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```



#### 1.5.7 Chuẩn hóa độ dài vectơ đơn vị

Trong quy trình này, chúng ta co giãn các thành phần của vectơ đặc trưng sao cho vectơ hoàn chỉnh có độ dài là 1, hoặc nói cách khác, có chuẩn là 1. **Lưu ý** rằng quy trình chuẩn hóa này sẽ chuẩn hóa vectơ **đặc trưng** chứ không phải vectơ **quan sát**. Vì vậy, chúng ta chia chuẩn của vectơ đặc trưng cho từng quan sát trên các biến khác nhau mà không phải chia chuẩn của vectơ **quan sát** cho các quan sát có cùng đối tượng.

Trước tiên, chúng ta hãy xem công thức rồi minh họa với một ví dụ.



**Công thức Co giãn về vectơ đơn vị**

Co giãn về vectơ đơn vị được tính bằng cách chia từng vectơ đặc trưng cho khoảng cách Manhattan (chuẩn l1) hoặc khoảng cách Euclid của vectơ (chuẩn l2):

$\Large X_{scaled-l_1} = \frac{X}{l_1(X)}$

$\Large X_{scaled-l_2} = \frac{X}{l_2(X)}$


**Khoảng cách Manhattan** là tổng các thành phần tuyệt đối của vectơ:  

$l_1(X) = |x1| + |x2| + ... + |xn|$


Còn **khoảng cách Euclid** được tính bằng căn bậc hai của tổng các thành phần của vectơ:

$l_2(X) = \sqrt{ x_1^2 + x_2^2 + ... + x_n^2 }$


Trong ví dụ trên, $x_1$ là biến 1, $x_2$ là biến 2 và $x_n$ là biến n; $X$ là dữ liệu cho 1 quan sát trên các biến (hay 1 hàng).

Cũng **lưu ý** rằng khi khoảng cách Euclid bình phương các giá trị của các thành phần vectơ đặc trưng, outlier sẽ có trọng số lớn hơn. Chúng ta thường ưu tiên sử dụng chuẩn hóa l1 với outlier.



**Ví dụ cho Co giãn về vectơ đơn vị** 

Giả sử dữ liệu có 1 quan sát (1 hàng) và 3 biến:

- number of pets (số thú nuôi)
- number of children (số trẻ em )
- age (tuổi)

Giá trị của từng biến cho quan sát riêng lẻ đó là 10, 15 và 20. Vectơ X = [10, 15, 20]. Sau đó:

$l_1(X) = 10 + 15 + 20 = 45$

$l_2(X) = \sqrt{ 10^2 + 15^2 + 20^2} = \sqrt{ 100 + 225 + 400} = 26.9$

Khoản cách Euclid luôn nhỏ hơn khoảng cách Manhattan.


Do đó các giá trị vectơ được chuẩn hóa là:

$\large X_{scaled-l_1} = [ \frac{10}{45}, \frac{15}{45}, \frac{20}{45} ]      =  [0.22, 0.33, 0.44]$

$\large X_{scaled-l_2} = [\frac{10}{26.9}, \frac{15}{26.9}, \frac{20}{26.9} ] =  [0.37, 0.55, 0.74]$




Scikit-learn đề xuất quy trình co giãn này cho phân loại hoặc phân cụm văn bản. Ví dụ: tích vô hướng của hai vectơ TF-IDF được chuẩn hóa l2 là độ tương tự cosin của các vectơ và là thước đo độ tương tự cơ sở cho Mô hình không gian vectơ thường được cộng đồng Truy xuất thông tin sử dụng.

```python
from sklearn.preprocessing import Normalizer
# thiết lập Normalizer
scaler = Normalizer(norm='l1')

# khớp scaler, quy trình này sẽ KHÔNG THỰC HIỆN BẤT CỨ ĐIỀU GÌ
scaler.fit(X_train)

# biến đổi tập huấn luyện và tập kiểm tra
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# ====== Co giãn thành L2 =======
scaler = Normalizer(norm='l2')

# khớp scaler, quy trình này sẽ KHÔNG THỰC HIỆN BẤT CỨ ĐIỀU GÌ
scaler.fit(X_train)

# biến đổi tập huấn luyện và tập kiểm tra
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```



### 1.6 Date Time

#### 1.6.1 Date

```python
import datetime

# Trích xuất tuần trong năm
data['issue_dt_week'] = data['issue_dt'].dt.week

# Trích xuất tháng
data['issue_dt_month'] = data['issue_dt'].dt.month

# Trích xuất quý
data['issue_dt_quarter'] = data['issue_dt'].dt.quarter

# Trích xuất kỳ
data['issue_dt_semester'] = np.where(data['issue_dt_quarter'].isin([1,2]), 1, 2)

#  Trích xuất năm
data['issue_dt_year'] = data['issue_dt'].dt.year

# Trích xuất ngày ở nhiều định dạng khác nhau
data['issue_dt_day'] = data['issue_dt'].dt.day
data['issue_dt_dayofweek'] = data['issue_dt'].dt.dayofweek
data['issue_dt_dayofweek'] = data['issue_dt'].dt.day_name()
data['issue_dt_is_weekend'] = np.where(data['issue_dt_dayofweek'].isin(['Sunday', 'Saturday']), 1,0)

# Trích xuất thời gian trôi qua giữa các ngày
data['issue_dt'] - data['last_pymnt_dt']

# tính số tháng trôi qua giữa 2 ngày
data['months_passed'] = (data['last_pymnt_dt'] - data['issue_dt']) / np.timedelta64(1, 'M')
data['months_passed'] = np.round(data['months_passed'],0)

# chênh lệch về thời gian với hôm nay
(datetime.datetime.today() - data['issue_dt']).head()

```

#### 1.6.2 Time

```python
import datetime

# Trích xuất giờ, phút, giây
df['hour'] = df['date'].dt.hour
df['min'] = df['date'].dt.minute
df['sec'] = df['date'].dt.second

# Trích xuất phần thời gian
df['time'] = df['date'].dt.time

# Trích xuất giờ, phút, giây cùng lúc
df[['h','m','s']] = pd.DataFrame([(x.hour, x.minute, x.second) for x in df['time']])
```

#### 1.6.3 Tính chênh lệch thời gian
```python
import datetime

# tính thời gian trôi qua theo giây
df['diff_seconds'] = df['End_date'] - df['Start_date']
df['diff_seconds'] = df['diff_seconds']/np.timedelta64(1,'s')

# tính thời gian trôi qua theo phút
df['diff_seconds'] = df['End_date'] - df['Start_date']
df['diff_seconds'] = df['diff_seconds']/np.timedelta64(1,'m')

```
[See more](https://www.datasciencemadesimple.com/difference-two-timestamps-seconds-minutes-hours-pandas-python-2/)

#### 1.6.4 Làm việc với các múi giờ khác nhau

```python
import datetime

# để làm việc với các múi giờ khác nhau, trước tiên chúng ta thống nhất múi giờ trung tâm
# đặt utc = True
df['time_utc'] = pd.to_datetime(df['time'], utc=True)

# tiếp theo, thay đổi tất cả các dấu thời gian theo múi giờ mong muốn, chẳng hạn như Europe/London
# trong ví dụ này
df['time_london'] = df['time_utc'].dt.tz_convert('Europe/London')
```



### 1.7 Features Selection

#### 1.7.1 Lựa chọn đặc trưng theo các phương pháp xuôi

Lựa chọn đặc trưng theo các phương pháp xuôi bắt đầu bằng cách huấn luyện mô hình học máy cho từng đặc trưng trong tập dữ liệu và lựa chọn đặc trưng mở đầu khiến mô hình hoạt động tốt nhất theo tiêu chí đánh giá nhất định.

Ở bước thứ hai, nó tạo ra các mô hình học máy cho tất cả các tổ hợp đặc trưng đã chọn ở bước trước và đặc trưng thứ hai. Nó chọn cặp tạo ra thuật toán hoạt động tốt nhất.

Phương pháp này tiếp tục bằng cách thêm mỗi lần 1 đặc trưng vào các đặc trưng đã chọn ở các bước trước cho đến khi xác định trước tiêu chí dừng.

Về lý thuyết, các mô hình có nhiều đặc trưng hơn sẽ hoạt động tốt hơn. Thuật toán sẽ tiếp tục thêm các đặc trưng mới cho đến khi đáp ứng tiêu chí, chẳng hạn: cho đến khi chất lượng của mô hình không tăng vượt quá một ngưỡng nhất định hoặc cho đến khi lựa chọn được một số đặc trưng nhất định như được triển khai trong thư viện mà chúng ta sẽ thảo luận trong notebook này.

Ví dụ, phép đo chất lượng mô hình có thể là roc_auc cho phân loại và r^2 cho hồi quy và nó do người dùng xác định.

Lựa chọn đặc trưng theo các phương pháp xuôi được gọi là thủ tục tham lam vì nó đánh giá nhiều tổ hợp đối tượng có thể: đơn, đôi, ba,... Do đó, nó rất khó tính toán và thậm chí là không khả thi nếu không gian đặc trưng lớn.


mlxtend là một gói đặc biệt trong Python thực hiện kiểu lựa chọn đặc trưng này: http://rasbt.github.io/mlxtend/


Trong triển khai mlxtend của Lựa chọn đặc trưng theo các phương pháp xuôi, tiêu chí dừng là số lượng đặc trưng được đặt tùy ý. Do đó, việc tìm kiếm sẽ kết thúc khi chúng ta đạt được số lượng đặc trưng được chọn mong muốn.


Điều này hơi tùy ý, chúng ta có thể đang chọn một số đặc trưng gần tối ưu hoặc tương tự như vậy, một số lượng lớn các đặc trưng. Tuy nhiên, bằng cách xem xét phép đo chất lượng mà thuật toán trả về khi lựa chọn đặc trưng, chúng ta có thể có biết liệu nhiều đặc trưng hơn có thêm giá trị không.

**Lưu ý**
Nếu muốn dừng tìm kiếm bằng cách sử dụng tiêu chí khác, chúng ta sẽ phải tự viết code thuật toán :(

Chúng ta sẽ sử dụng thuật toán lựa chọn đặc trưng theo các phương pháp xuôi từ mlxtend trong tập dữ liệu phân loại và hồi quy. 

```bash
conda install mlxtend -y
```

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, r2_score

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# ===== REGRESSION ========
# lựa chọn đặc trưng theo các phương pháp xuôi
sfs = SFS(RandomForestRegressor(n_estimators=10, n_jobs=4, random_state=10), 
           k_features=20, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='r2',
           cv=2)

sfs = sfs.fit(np.array(X_train), np.array(y_train))    

# chỉ số của các cột đã chọn 
sfs.k_feature_idx_

# các cột đã chọn
X_train.columns[list(sfs.k_feature_idx_)]

```



#### 1.7.2 Lựa chọn đặc trưng theo các phương pháp ngược

Lựa chọn đặc trưng theo các phương pháp ngược bắt đầu bằng cách khớp mô hình học máy sử dụng tất cả các đặc trưng trong tập dữ liệu và xác đinh chất lượng mô hình.

Sau đó, nó huấn luyện mô hình trên tất cả các tổ hợp có thể có của tất cả các đặc trưng - 1, loại bỏ đặc trưng trả về mô hình có chất lượng thấp cao nhất khi bỏ đặc trưng đó đi.

Ở bước thứ ba, huấn luyện các mô hình trong tất cả các tổ hợp có thể của các đặc trưng còn lại từ bước hai bớt đi 1 đặc trưng và loại bỏ đặc trưng khiến mô hình hoạt động tốt nhất.

Thuật toán dừng theo một tiêu chí do người dùng xác định. Tiêu chí này có thể là chất lượng mô hình không giảm vượt quá một ngưỡng nhất định hoặc đạt tới số lượng đặc trưng đã chọn nhất định như trong triển khai mlxtend.


Ví dụ, phép đo chất lượng mô hình có thể là roc_auc cho phân loại và r^2 cho hồi quy và nó do người dùng xác định.

Lựa chọn đặc trưng theo các phương pháp ngược được gọi là thủ tục tham lam vì nó đánh giá tất cả các tổ hợp đặc trưng n, rồi n-1, n-2,... Do đó, nó rất khó tính toán và thậm chí là không khả thi nếu không gian đặc trưng lớn.

```python

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, r2_score

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs = SFS(RandomForestRegressor(n_estimators=10, n_jobs=4, random_state=10), 
           k_features=20, 
           forward=False, 
           floating=False, 
           verbose=2,
           scoring='r2',
           cv=2)

sfs = sfs.fit(np.array(X_train), np.array(y_train))

selected_feat = X_train.columns[list(sfs.k_feature_idx_)]
```



#### 1.7.3 Tìm kiếm đầy đủ

Tìm kiếm đầy đủ tìm tập hợp con các đặc trưng tốt nhất trong số tất cả các tập hợp con đặc trưng có thể theo một phép đo đặc trưng xác định cho một thuật toán học máy nhất định.

 Ví dụ: nếu chúng ta huấn luyện hồi quy logistic và tập dữ liệu gồm 4 đặc trưng, thuật toán sẽ đánh giá tất cả **15** tổ hợp đặc trưng như sau:

- tất cả các tổ hợp có thể của 1 đặc trưng
- tất cả các tổ hợp có thể của 2 đặc trưng
- tất cả các tổ hợp có thể của 3 đặc trưng
- tất cả 4 đặc trưng

và chọn tổ hợp dẫn đến chất lượng tốt nhất (ví dụ: độ chính xác của phân loại) của hồi quy logistic.

Tìm kiếm đầy đủ là một thuật toán tham lam vì nó đánh giá tất cả các kết hợp đặc trưng có thể có. Nó rất khó tính toán và thậm chí là không khả thi nếu không gian đặc trưng lớn.

```python
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

efs = EFS(RandomForestRegressor(n_estimators=5,
                                n_jobs=4,
                                random_state=0,
                                max_depth=2),
                                min_features=1,
                                max_features=2,
                                scoring='r2',
                                print_progress=True,
                                cv=2)
efs = efs.fit(np.array(X_train), y_train)

X_train.columns[list(efs.best_idx_)]
```

Tìm kiếm đầy đủ rất khó tính toán, chúng ta không thường sử dụng quy trình này cho những lý do tương tự, nhưng nếu truy cập tới các siêu máy tính thì có thể thử.



#### 1.7.4 Lọc dựa trên các kiểm định thống kê

```python
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest

# Loại bỏ các đặc trưng không đổi
constant_features = [
    feat for feat in X_train.columns if X_train[feat].std() == 0
]
X_train.drop(labels=constant_features, axis=1, inplace=True)
X_test.drop(labels=constant_features, axis=1, inplace=True)

# Loại bỏ các đặc trưng gần như không đổi
sel = VarianceThreshold(threshold=0.01) # đặt ngưỡng = 0.01
sel.fit(X_train) # tìm các đặc trưng với phương sai thấp

features_to_keep = X_train.columns[sel.get_support()]

X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

X_train= pd.DataFrame(X_train)
X_train.columns = features_to_keep

X_test= pd.DataFrame(X_test)
X_test.columns = features_to_keep

# Loại bỏ các đặc trưng trùng lặp
duplicated_feat = []
for i in range(0, len(X_train.columns)):
    if i % 10 == 0:  # điều này giúp chúng ta hiểu vòng lặp diễn ra thế nào
        print(i)

    col_1 = X_train.columns[i]

    for col_2 in X_train.columns[i + 1:]:
        if X_train[col_1].equals(X_train[col_2]):
            duplicated_feat.append(col_2)

X_train.drop(labels=duplicated_feat, axis=1, inplace=True)
X_test.drop(labels=duplicated_feat, axis=1, inplace=True)

# Loại bỏ các đặc trưng tương quan
def correlation(dataset, threshold):
    
    col_corr = set()  # tập hợp tất cả tên của các cột tương quan
    corr_matrix = dataset.corr()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # chúng ta cần tìm giá trị hệ số tuyệt đối
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]  # lấy tên của cột
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.8)

X_train.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)

# Lựa chọn các đặc trưng dựa trên anova
sel_ = SelectKBest(f_classif, k=20).fit(X_train, y_train) # set k=20

# thu thập tên các đặc trưng đã chọn
features_to_keep = X_train.columns[sel_.get_support()]

# lựa chọn đặc trưng
X_train_anova = sel_.transform(X_train)
X_test_anova = sel_.transform(X_test)

# mảng numpy thành dataframe
X_train_anova = pd.DataFrame(X_train_anova)
X_train_anova.columns = features_to_keep

X_test_anova = pd.DataFrame(X_test_anova)
X_test_anova.columns = features_to_keep

```



#### 1.7.5 Lọc theo RMSE

Quy trình này hoạt động như sau:

- Xây dựng mô hình trên mỗi đặc trưng để dự đoán mục tiêu.
- Đưa ra dự đoán sử dụng mô hình được tạo ra từ đặc trưng đã đề cập.
- Đo lường chất lượng của dự đoán đó, có thể là roc-auc (bài toàn phân loại), msse (bài toán hồi quy).
- Xếp hạng các đặc trưng theo phép đo (roc-auc hoặc mse).
- Chọn ra các đặc trưng có xếp hạng cao nhất.

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error

# xác đinh mse cho từng đặc trưng
mse_values = []

# lặp qua từng biến
for feature in X_train.columns:
    
    # huấn luyện cây quyết định
    clf = DecisionTreeRegressor()
    clf.fit(X_train[feature].fillna(0).to_frame(), y_train)
    
    # đưa ra dự đoán
    y_scored = clf.predict(X_test[feature].fillna(0).to_frame())
    
    # xác định mse và lưu trữ nó
    mse_values.append(mean_squared_error(y_test, y_scored))

mse_values = pd.Series(mse_values, index=X_train.columns)
mse_values.index = X_train.columns
mse_values.sort_values(ascending=False).plot.bar(figsize=(20,8))

selected_features = mse_values[mse_values < np.mean(mse_values)].index
X_train = X_train[selected_features]
X_test = X_test[selected_features]

```

#### 1.7.6 Hệ số hồi quy tuyến tính

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel

sel_ = SelectFromModel(LinearRegression())
sel_.fit(X_train, y_train)
selected_feat = X_train.columns[(sel_.get_support())]

```