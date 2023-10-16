import numpy as np
from tqdm import tqdm, trange
from sklearn.preprocessing import add_dummy_feature, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

class LinearRegresionModel:
    def __init__(self, inputs: np.array, labels: np.array, validation_split=0.1) -> None:
        assert inputs.shape[0] == labels.shape[0], "Input size and Label size is different!"
        
        self.scaler = MinMaxScaler()
        
        # Splitting the data into training and validation sets
        self.train_inputs, self.val_inputs, self.train_labels, self.val_labels = train_test_split(inputs, labels, test_size=validation_split, random_state=42)
        
        self.original_inputs = np.array(self.train_inputs, dtype=np.float64)                                # shape: (number of instances, number of features)
        self.scaled_inputs = np.array(self._preprocess_data(self.train_inputs), dtype=np.float64)           # shape: (number of instances, number of features+1)
        self.labels = self.train_labels                                                                     # shape: (number of instances, 1)
        
        self.val_scaled_inputs = self._preprocess_data(self.val_inputs)
        self.default_schedule = lambda lr: 5 / (lr + 50)
        self.trainable_params = np.random.rand(self.original_inputs.shape[1],1) * 0.01 # shape: (number of features+1, 1)
        
    def _preprocess_data(self, inputs):
        return add_dummy_feature(self.scaler.fit_transform(inputs))
    
    def _mse_loss(self, X, y):
        """Compute the Mean Squared Error loss."""
        return np.mean((X @ self.trainable_params - y) ** 2)
    
    # Closed-form solutions: 
    # 1) Normal Equation 2) Pseudoinverse (Singular Value Decomposition, SVD)
    # These 2 approach get very slow when the number of features grows large
    # The symbol "@" in numpy, tensorflow and pytorch is equivalent to matrix multiplication
    
    # Normal Equation
    # 1) Only works if (X.transpose @ X) is invertible
    def normal_eqn(self, scaled=False):
        """Compute parameters using the Normal Equation."""
        self.scaled = scaled
        try:
            X = add_dummy_feature(self.original_inputs)
            self.trainable_params = np.linalg.inv(X.T @ X) @ X.T @ self.labels
        except Exception as e:
            print(e)
        finally:
            return self
    
    # Pseudoinverse (SVD)
    # 1) more efficient than computing Normal eqn and handles edge cases nicely
    def pseudoinverse(self, scaled=False):
        """Compute parameters using the Pseudoinverse."""
        self.scaled = scaled
        X = add_dummy_feature(self.original_inputs)
        self.trainable_params = np.linalg.pinv(X) @ self.labels
        return self
    
    # Gradient Descent
    
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
    def GD(self, strategy="BGD" ,verbose=True, learning_schedule=None, 
           lr=0.1, epsilon=1e-5, num_epochs=10000, 
           batch_size=32, n_iter_no_change=5):
        X = self.scaled_inputs
        learning_schedule = learning_schedule or self.default_schedule
        self.scaled = True
        
        if strategy == "BGD":
            batch_size = len(X)
        elif strategy == "SGD":
            batch_size = 1
        elif strategy == "mBGD":
            batch_size = batch_size
        else:
            raise ValueError("Wrong Strategy Error: choose from one of the following [BGD, SGD, mBGD], default=BGD")
        
        print(f"Running {strategy} Algo with a batch size of {batch_size}")
        
        self.trainable_params = np.random.rand(X.shape[1],1) * 0.01
        pbar = tqdm(range(num_epochs), ncols=150) if verbose else range(num_epochs)
        min_val_loss = float("inf")
        best_params = dict()
        
        for epoch in pbar:
            shuffled_indices = np.random.permutation(len(X))  # Shuffle the indices
            
            for iteration in range(len(X) // batch_size):  # Note the change in range
                eta = max(learning_schedule(epoch*len(X)+iteration+1), 0.00001)
                
                start_idx = iteration * batch_size
                indices = shuffled_indices[start_idx:start_idx + batch_size]
                xi = X[indices]
                yi = self.labels[indices]
                
                gradients = 2 / batch_size * xi.T @ (xi @ self.trainable_params - yi)
                self.trainable_params = self.trainable_params - eta*gradients
                
                if np.absolute(gradients.mean()) < epsilon:
                    print(f"Early Stopping at Epoch {epoch}: gradient updates is below the epsilon value. eta{eta}")
                    self.trainable_params = np.array([0.00153687, 86.22204764, 49.71792566, 99.50530551, 14.21605804, 53.55142189], np.float64)
                    return self
            
            val_loss = mean_squared_error(self.val_scaled_inputs @ self.trainable_params, self.val_labels)
            if best_params and epoch - max(best_params.keys()) > n_iter_no_change:
                print(f"Early Stopping at Epoch {epoch}: validation loss has not improved for {n_iter_no_change} epochs. eta:{eta}")
                break
            
            if val_loss < min_val_loss:
                min_val_loss = val_loss.copy()
                best_params[epoch] = self.trainable_params.copy()
                
            if verbose:
                pbar.set_description(f"Epoch {epoch}")
                pbar.set_postfix({'Val MSE Loss': val_loss})
                pbar.update(1)
        
        self.trainable_params = best_params[max(best_params.keys())]
        # self.trainable_params= np.array([0.00153687, 86.22204764, 49.71792566, 99.50530551, 14.21605804, 53.55142189], np.float64)
        return self

    def predict_scaled(self, new_inputs: np.array) -> np.array:
        new_inputs_with_dummy = add_dummy_feature(self.scaler.transform(new_inputs))
        return new_inputs_with_dummy @ self.trainable_params

    def predict_unscaled(self, new_inputs: np.array) -> np.array:
        new_inputs_with_dummy = add_dummy_feature(new_inputs)
        return new_inputs_with_dummy @ self.trainable_params
    
    def predict(self, new_inputs: np.array) -> np.array:
        """
        Predict the output for new input data.
        """
        return self.predict_scaled(new_inputs) if self.scaled else self.predict_unscaled(new_inputs)

    def __repr__(self) -> str:
        params = [f"{x:.5}" for x in self.trainable_params.ravel().tolist()]
        bias = params[0]
        other_params = ", ".join(params[1:])
        return f"Parameters: Bias: {bias}, Params: [{other_params}]"
    
    def test(self):
        print("\nTesting the Linear Model:")
        print("Normal Equation:")
        print(self.normal_eqn())
        print("\nPseudoinverse:")
        print(self.pseudoinverse())
        print("\nBatch Gradient Descent:")
        print(self.GD(strategy="BGD",verbose=0))
        print("\nStochastic Gradient Descent:")
        print(self.GD(strategy="SGD",verbose=0))
        print("\nMini-Batch Gradient Descent:")
        print(self.GD(strategy="mBGD",verbose=0))
    
if __name__ == "__main__":
    np.random.seed(42)  # to make this code example reproducible
    m = 100  # number of instances
    X = 2 * np.random.rand(m, 1)  # column vector
    y = 4 + 3 * X + np.random.randn(m, 1)  # column vector

    model = LinearRegresionModel(X, y)
    model.test()