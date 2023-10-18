import numpy as np
from sklearn.preprocessing import add_dummy_feature, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange


class LinearRegressionModel:
    def __init__(self, inputs: np.array, labels: np.array, scaler=None, loss_fn=None,validation_split=0.1) -> None:
        """
        Initialize the Linear Regression Model.
        
        Args:
        - inputs (np.array): Input data of shape (number of instances, number of features).
        - labels (np.array): Corresponding labels of shape (number of instances, 1).
        - validation_split (float, optional): Fraction of data to be used as validation set. Defaults to 0.1.
        
        Raises:
        - AssertionError: If the number of instances in inputs and labels don't match.
        """
        if inputs.shape[0] != labels.shape[0]:
            raise AssertionError("The number of instances in inputs and labels must be the same.")
        
        # Splitting the data into training and validation sets
        self.train_inputs, self.val_inputs, self.train_labels, self.val_labels = train_test_split(
            inputs, labels, test_size=validation_split, random_state=42
        )

        self.trainable_params = np.random.rand(self.train_inputs.shape[1] + 1, 1) * 0.01
        
        self.scaler = scaler
        self.loss_fn = loss_fn or self._mse_loss
        
    
    """Closed-form solutions""" 
    # 1) Normal Equation 2) Pseudoinverse (Singular Value Decomposition, SVD)
    # These 2 approach get very slow when the number of features grows large
    # The symbol "@" in numpy, tensorflow and pytorch is equivalent to matrix multiplication
    def _normal_eqn(self, X: np.array, y: np.array) -> np.array:
        """
        Compute the optimal parameters using the Normal Equation.
        
        Args:
        - X (np.array): Input data of shape (number of instances, number of features + 1).
        - y (np.array): Corresponding labels of shape (number of instances, 1).
        
        Returns:
        - np.array: Optimal parameters using the Normal Equation.
        """
        return np.linalg.inv(X.T @ X) @ X.T @ y

    def _pseudoinverse(self, X: np.array, y: np.array) -> np.array:
        """
        Compute the optimal parameters using the Pseudoinverse.
        
        Args:
        - X (np.array): Input data of shape (number of instances, number of features + 1).
        - y (np.array): Corresponding labels of shape (number of instances, 1).
        
        Returns:
        - np.array: Optimal parameters using the Pseudoinverse.
        """
        return np.linalg.pinv(X) @ y
    

    """Gradient Descent"""
    # Batch Gradient Descent (BGD)
    # 1) ensure that all features have a similar scale (i.e. feature scaling) for faster convergence
    # 2) If a cost function is a convex & continuous function it is gurunteed that it will find the global minima
    
    # Stochastic Gradient Descent (SGD)
    # 1) Unlike BGD that uses entire training set, SGD uses only 1 random data point for training
    # 2) Due to its stochastic nature, much less regular than BGD, the cost function will bounce up and down, 
    # decreasing only on average. You can gradually decrease the learning rate to resolve this issue (learning schedule)
    # 3) However, also this stochastic nature can help to jump out of local minima for irregular cost function
    
    # Mini-Batch Gradient Descent (mBGD)
    # 1) More stable than SGD by using a batch size of data points 
    def _bgd(self, X: np.array, y: np.array, lr: float, epochs: int, val_X=None, val_y=None) -> np.array:
        """
        Batch Gradient Descent strategy.
        
        Args:
        - X (np.array): Input data of shape (number of instances, number of features + 1).
        - y (np.array): Corresponding labels of shape (number of instances, 1).
        - lr (float): Learning rate.
        - epochs (int): Number of training epochs.
        - val_X (np.array, optional): Validation input data. If provided, used to compute validation loss.
        - val_y (np.array, optional): Validation labels. If provided, used to compute validation loss.

        Returns:
        - np.array: Updated parameters after training.
        """
        m = len(X)
        epoch_iterator = tqdm(range(epochs), desc="Training", unit="epoch") if epochs > 1 else range(epochs)
        for epoch in epoch_iterator:
            gradients = 2/m * X.T @ (X @ self.trainable_params - y)
            self.trainable_params -= lr * gradients

            # Compute training & validation loss and display
            self._validate_and_update_tqdm(X,y,val_X,val_y,epoch_iterator,epoch,epochs)

        return self.trainable_params

    def _sgd(self, X: np.array, y: np.array, lr: float, epochs: int, val_X=None, val_y=None) -> np.array:
        """
        Stochastic Gradient Descent strategy.
        
        Args:
        - X (np.array): Input data of shape (number of instances, number of features + 1).
        - y (np.array): Corresponding labels of shape (number of instances, 1).
        - lr (float): Learning rate.
        - epochs (int): Number of training epochs.
        
        Returns:
        - np.array: Updated parameters after training.
        """
        m = len(X)
        epoch_iterator = tqdm(range(epochs), desc="Training", unit="epoch") if epochs > 1 else range(epochs)
        for epoch in epoch_iterator:
            for i in range(m):
                random_index = np.random.randint(m)
                xi = X[random_index:random_index+1]
                yi = y[random_index:random_index+1]
                gradients = 2 * xi.T @ (xi @ self.trainable_params - yi)
                self.trainable_params -= lr * gradients

            # Compute training & validation loss and display
            self._validate_and_update_tqdm(X,y,val_X,val_y,epoch_iterator,epoch,epochs)

        return self.trainable_params

    def _mbgd(self, X: np.array, y: np.array, lr: float, epochs: int, batch_size: int, val_X=None, val_y=None) -> np.array:
        """
        Mini-Batch Gradient Descent strategy.
        
        Args:
        - X (np.array): Input data of shape (number of instances, number of features + 1).
        - y (np.array): Corresponding labels of shape (number of instances, 1).
        - lr (float): Learning rate.
        - epochs (int): Number of training epochs.
        - batch_size (int): Size of mini-batches.
        
        Returns:
        - np.array: Updated parameters after training.
        """
        m = len(X)
        n_batches = int(np.ceil(m / batch_size))
        epoch_iterator = tqdm(range(epochs), desc="Training", unit="epoch") if epochs > 1 else range(epochs)
        for epoch in epoch_iterator:
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                xi = X_shuffled[start:end]
                yi = y_shuffled[start:end]
                gradients = 2/batch_size * xi.T @ (xi @ self.trainable_params - yi)
                self.trainable_params -= lr * gradients

            # Compute training & validation loss and display
            self._validate_and_update_tqdm(X,y,val_X,val_y,epoch_iterator,epoch,epochs)

        return self.trainable_params

    def _validate_and_update_tqdm(self,X,y,val_X,val_y,epoch_iterator,epoch,epochs):
        epoch_iterator.set_description(f"Epoch {epoch+1}/{epochs}")
        
        # If validation data is provided, compute and display validation loss
        train_loss = self.loss_fn(self.predict(X,is_training=True), y)
        if val_X is not None and val_y is not None:
            val_loss = self.loss_fn(self.predict(val_X,is_training=True), val_y)
            epoch_iterator.set_postfix({"Training Loss": train_loss, "Validation Loss": val_loss})
        else:
            epoch_iterator.set_postfix({"Training Loss": train_loss})
        
    def train(self, strategy: str = "pseudoinverse", lr: float = 0.01, epochs: int = 100, batch_size: int = 32) -> np.array:
        # reset the trainable_params 
        self.trainable_params = np.random.rand(self.train_inputs.shape[1] + 1, 1) * 0.01
        
        # Preprocess the training data
        X = add_dummy_feature(self.scaler.fit_transform(self.train_inputs))
        y = self.train_labels
        
        # Preprocess the validation data
        val_X = add_dummy_feature(self.scaler.transform(self.val_inputs))
        val_y = self.val_labels
        
        strategies = {
            "normal_eqn": self._normal_eqn,
            "pseudoinverse": self._pseudoinverse,
            "BGD": lambda X, y: self._bgd(X, y, lr, epochs, val_X, val_y),
            "SGD": lambda X, y: self._sgd(X, y, lr, epochs, val_X, val_y),
            "mBGD": lambda X, y: self._mbgd(X, y, lr, epochs, batch_size, val_X, val_y),
        }
        
        if strategy not in strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Choose one of {list(strategies.keys())}.")
        
        self.trainable_params = strategies[strategy](X, y)
        return self.trainable_params

    def predict(self, new_inputs: np.array, is_scaled: bool = True, is_training: bool = False) -> np.array:
        """
        Predict the output for new input data.
        
        Args:
        - new_inputs (np.array): New input data of shape (number of instances, number of features).
        - scaled (bool, optional): Whether to scale the input data before prediction. Defaults to True.
        - is_training (bool, optional): Whether to transform the input data before prediction. Defaults to False.
        
        Returns:
        - np.array: Predicted output.
        """
        if is_training:
            transformed_inputs = new_inputs
        elif is_scaled:
            transformed_inputs = add_dummy_feature(self.scaler.transform(new_inputs))
        else:
            transformed_inputs = add_dummy_feature(new_inputs)
        return transformed_inputs @ self.trainable_params

    def demonstrate(self):
        """
        Demonstrate the various training methods and their results.
        """
        print("\nTesting the Linear Model:")
        
        print("\nTraining using Normal Equation:")
        self.train(strategy="normal_eqn")
        print(self)
        
        print("\nTraining using Pseudoinverse:")
        self.train(strategy="pseudoinverse")
        print(self)
        
        print("\nTraining using Batch Gradient Descent:")
        self.train(strategy="BGD")
        print(self)
        
        print("\nTraining using Stochastic Gradient Descent:")
        self.train(strategy="SGD")
        print(self)
        
        print("\nTraining using Mini-Batch Gradient Descent:")
        self.train(strategy="mBGD")
        print(self)

    def __repr__(self) -> str:
        params = [f"{x:.5}" for x in self.trainable_params.ravel().tolist()]
        bias = params[0]
        other_params = ", ".join(params[1:])
        return f"Parameters: Bias: {bias}, Params: [{other_params}]"
    
if __name__ == "__main__":
    np.random.seed(42)  # to make this code example reproducible
    m = 100  # number of instances
    X = 2 * np.random.rand(m, 1)  # column vector
    y = 4 + 3 * X + np.random.randn(m, 1)  # column vector
    
    model = LinearRegressionModel(X, y, scaler=StandardScaler(), loss_fn=mean_squared_error)
    model.demonstrate()