import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

def build_model(units, activation, kernel_initializer, loss, optimizer, input_shape):
    model = Sequential()
    model.add(Dense(units=units, input_shape=input_shape, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dropout(0.2))

    model.add(Dense(200, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dropout(0.2))

    model.add(Dense(100, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dropout(0.2))

    model.add(Dense(50, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dropout(0.2))

    model.add(Dense(units=input_shape[1], activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['binary_accuracy'])
    return model

X = np.random.rand(100, 10)
y = np.random.rand(100, 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifier = KerasClassifier(build_fn=build_model, input_shape=(X_train.shape[1],))
print(classifier)
print("-------------------------------------------------------------------------")

parametros = {
    'batch_size': [10, 30, 50],
    'epochs': [50, 100, 200],
    'optimizer': ['adam', 'rmsprop', 'adamax'],
    'loss': ['binary_crossentropy', 'mean_squared_error', 'mean_absolute_error'],
    'kernel_initializer': ['random_uniform', 'normal', 'zeros'],
    'activation': ['relu', 'sigmoid', 'softmax'],
    'units': [32, 16, 8]
}

grid_search = GridSearchCV(estimator=classifier, param_grid=parametros, scoring='accuracy', cv=5)
print(grid_search)
print("-------------------------------------------------------------------------")

grid_search = grid_search.fit(X_test, y_test)
print(grid_search)
print("-------------------------------------------------------------------------")

print(grid_search)
print("-------------------------------------------------------------------------")

best_params = grid_search.best_params_
print(best_params)
print("-------------------------------------------------------------------------")

best_precision = grid_search.best_score_
print(best_precision)
print("-------------------------------------------------------------------------")