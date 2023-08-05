import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

def build_model(units, activation, kernel_initializer, loss, optimizer):
    model = Sequential()
    model.add(Dense(units=units, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    model.add(Dropout(0.2))

    model.add(Dense(units=units, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dropout(0.2))

    model.add(Dense(units=units, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dropout(0.2))

    model.add(Dense(units=1, activation='sigmoid'))
    print(model.summary())
    print("-------------------------------------------------------------------------")

    model.compile(loss=loss, optimizer=optimizer, metrics=['binary_accuracy'])
    print(model)
    print("-------------------------------------------------------------------------")

    return model

X = np.random.rand(100, 30)  # Corrigido para 30 dimensões.
y = np.random.rand(100, 1)   # Corrigido para 1 dimensão.
print("Imprimindo X e Y")
print(X)
print(y)
print("-------------------------------------------------------------------------")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Imprimindo dados de treino e teste X")
print(X_train)
print(X_test)
print("-------------------------------------------------------------------------")
print("Imprimindo dados de treino e teste Y")
print(y_train)
print(y_test)
print("-------------------------------------------------------------------------")

classifier = KerasClassifier(build_fn=build_model)
print("Classifier")
print(classifier)
print("-------------------------------------------------------------------------")

parametros = {
    'batch_size' : [10, 30, 50],
    'epochs': [50, 100, 150],
    'optimizer': ['adam', 'rmsprop', 'adamax'],
    'loss': ['binary_crossentropy', 'mean_squared_error', 'mean_absolute_error'],
    'kernel_initializer': ['random_uniform', 'normal', 'zeros'],
    'activation': ['relu', 'sigmoid', 'softmax'],
    'units': [32, 16, 8]
}
print("Parametros")
print(parametros)
print("-------------------------------------------------------------------------")

grid_search = GridSearchCV(estimator=classifier, param_grid=parametros, scoring='accuracy', cv=5)
print("Grid Search 1")
print(grid_search)
print("-------------------------------------------------------------------------")

grid_search.fit(X_train, y_train)  # Corrigido para usar X_train e y_train.
print("Grid Search 2")
print(grid_search)
print("-------------------------------------------------------------------------")

print("Grid Search 3")
print(grid_search)
print("-------------------------------------------------------------------------")

best_params = grid_search.best_params_
print("Best params")
print(best_params)
print("-------------------------------------------------------------------------")

best_precision = grid_search.best_score_
print("Best precision")
print(best_precision)
print("-------------------------------------------------------------------------")
