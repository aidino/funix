import numpy as np

# Vectorization

# MSE
def mse_non_vectorized(data, function):
    """ Tính mse không thông qua vectorization

    Args:
        data (tuple list): Dữ liệu đầu vào [(x1, y1), (x2, y2), ...]
        function (function): hàm dự đoán  y_pred = function(x)

    Returns:
        scalar: mse
    """
    sosr = 0 # sum of squared residuals
    for (x, y) in data:
        residuals = y - function(x)
        sosr += residuals**2
    mse = sosr / len(data)
    return mse

def mse_vectorized(y, y_pred):
    """Tính mse thông qua vectorization

    Args:
        y (array): Giá trị thực tế
        y_pred (array): Giá trị dự đoán

    Returns:
        scalar: mse
    """
    error = y - y_pred
    loss = 1/(y.size) * np.dot(error.T, error)
    return loss


# Linear Regression

def add_intercept_ones(x):
    """ Thêm hệ số bằng 1 vào đầu dữ liệu x """
    intercept_ones = np.ones((len(x),1)) # results in array( [ [1],..,[1] ] )
    x_b = np.c_[intercept_ones,x] # we now add the additional ones as a new column to our X
    return x_b

# Normal Equation
def normal_equation_linear_regression(x,y):
    """ Tính theta cho hàm linear regression bằng hàm normal equation"""
    x_b = add_intercept_ones(x)
    theta_optimal = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y) # the normal equation
    return theta_optimal

