import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

class MLPTrainer:
    def __init__(self, input_dim, model, device="cpu", lr=0.005):
        self.device = torch.device(device)
        self.model = model(input_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.history = []  # To track loss over epochs

    def prepare_data(self, X_train, y_train, X_test, y_test):
        """Convert data to PyTorch tensors and move to device."""
        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(self.device)

    def train(self, n_epochs=50, log_interval=10):
        """Train the MLP model."""
        for epoch in range(n_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train)  # Forward pass
            loss = self.criterion(outputs, self.y_train)  # Loss computation
            loss.backward()  # Backward pass
            self.optimizer.step()  # Update weights
            self.history.append(loss.item())
            if epoch % log_interval == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def evaluate(self):
        """Evaluate the model and compute the MSE on the test set."""
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_test).cpu().numpy().flatten()  # Move predictions to CPU
        mse = mean_squared_error(self.y_test.cpu().numpy(), y_pred)
        print(f"MLP MSE: {mse}")
        return mse, y_pred

    def predict(self, X_new):
        """Make predictions on new data."""
        self.model.eval()
        # Ensure X_new is a PyTorch tensor and move to the correct device
        if not isinstance(X_new, torch.Tensor):
            X_new_torch = torch.as_tensor(X_new, dtype=torch.float32, device=self.device)
        else:
            X_new_torch = X_new.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_new_torch).cpu().numpy().flatten()
        return predictions


    def predict_deployment(self, X_data):
        """Predict outcomes for deployment data."""
        self.model.eval()
        X_data_torch = torch.tensor(X_data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_data_torch).cpu().numpy().flatten()
        return predictions