import numpy as np
from LinearModel import LinearRegressionModel
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def compare_models(X_train, y_train, X_test, y_test):
    """
    Compares the performance of the custom LinearRegressionModel and the SGDRegressor from sklearn.
    
    Args:
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Testing data and labels.
    """
    # Training the custom model
    my_model = LinearRegressionModel(inputs=X_train, labels=y_train, validation_split=0.1, loss_fn=mean_squared_error, scaler=StandardScaler())
    my_model.train()

    # Training the SGDRegressor from sklearn
    sgd_reg = SGDRegressor(max_iter=10000, tol=1e-5, penalty=None, eta0=0.01, n_iter_no_change=5, random_state=42)
    sgd_reg.fit(X_train, y_train.ravel())  # y.ravel() because fit() expects 1D targets

    # Predictions
    y_pred_my_model = my_model.predict(X_test, is_scaled=True)  # Assuming the data should be scaled
    y_pred_sgd_reg = sgd_reg.predict(X_test).reshape(-1, 1)  # Reshaping to match the shape

    # Calculate MSE
    mse_my_model = mean_squared_error(y_test, y_pred_my_model)
    mse_sgd_reg = mean_squared_error(y_test, y_pred_sgd_reg)

    # Print results
    print(f"\nMSE for your model: {mse_my_model} {my_model}")
    print(f"MSE for SGDRegressor: {mse_sgd_reg} Params: {sgd_reg.intercept_} {sgd_reg.coef_}")
    
    # Determine the better model
    if mse_my_model < mse_sgd_reg:
        print("Your model performed better!")
    elif mse_my_model > mse_sgd_reg:
        print("SGDRegressor performed better!")
    else:
        print("Both models have the same performance!")

if __name__ == "__main__":

    # Linear dataset
    print("Comparing on Linear Dataset:")
    X_linear = 2 * np.random.rand(500, 1)
    y_linear = 4 + 3 * X_linear + np.random.randn(500, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)
    compare_models(X_train, y_train, X_test, y_test)

    # Quadratic dataset
    print("\nComparing on Quadratic Dataset:")
    X_quad = 6 * np.random.rand(200, 1) - 3
    y_quad = 0.5 * X_quad ** 2 + X_quad + 2 + np.random.randn(200, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_quad, y_quad, test_size=0.2, random_state=42)
    compare_models(X_train, y_train, X_test, y_test)

    # Synthetic dataset
    print("\nComparing on Synthetic Dataset:")
    X_syn, y_syn = make_regression(n_samples=5000, n_features=2, noise=0.1, random_state=42)
    y_syn = y_syn.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)
    compare_models(X_train, y_train, X_test, y_test)