import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

class LigthCurve:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.model = MLPRegressor(hidden_layer_sizes=(100, 50))

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        return self.model.predict(self.X_test)

    def plot_results(self, idx=0):
        plt.plot(self.y_test[idx], label='Real')
        plt.plot(self.predict()[idx], label='Predict')
        plt.legend()
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Ligth Curve')
        plt.show()
