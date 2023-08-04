import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

class Build:
    def __init__(self, X, y, text):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.model = self.build_model()
        self.text = text

    def build_model(self):
        model = Sequential()
        model.add(Dense(400, input_dim=self.X_train.shape[1], activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.y_train.shape[1]))
        model.compile(loss='mse', optimizer='adam')
        return model

    def train(self):
        self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=32, verbose=0)

    def predict(self):
        return self.model.predict(self.X_test)

    def plot_results(self, idx=0):
        plt.plot(self.y_test[idx], label='Real')
        plt.plot(self.predict()[idx], label='Predict')
        plt.legend()
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(self.text)
        plt.show()