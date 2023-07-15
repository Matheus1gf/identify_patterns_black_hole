import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Dados simulados (exemplo)
X_espectro = np.random.rand(100, 10)  # Matriz de características do espectro eletromagnético
X_curva_luz = np.random.rand(100, 20)  # Matriz de características da curva de luz
X_ondas_gravitacionais = np.random.rand(100, 30)  # Matriz de características das ondas gravitacionais

# Dados projetados (exemplo)
y_espectro = np.random.rand(100, 5)  # Matriz de dados projetados do espectro eletromagnético
y_curva_luz = np.random.rand(100, 10)  # Matriz de dados projetados da curva de luz
y_ondas_gravitacionais = np.random.rand(100, 15)  # Matriz de dados projetados das ondas gravitacionais

# Divisão em conjuntos de treinamento e teste
X_espectro_train, X_espectro_test, y_espectro_train, y_espectro_test = train_test_split(X_espectro, y_espectro, test_size=0.2)
X_curva_luz_train, X_curva_luz_test, y_curva_luz_train, y_curva_luz_test = train_test_split(X_curva_luz, y_curva_luz, test_size=0.2)
X_ondas_gravitacionais_train, X_ondas_gravitacionais_test, y_ondas_gravitacionais_train, y_ondas_gravitacionais_test = train_test_split(X_ondas_gravitacionais, y_ondas_gravitacionais, test_size=0.2)

# Pré-processamento dos dados
scaler_espectro = StandardScaler()
X_espectro_train = scaler_espectro.fit_transform(X_espectro_train)
X_espectro_test = scaler_espectro.transform(X_espectro_test)

scaler_curva_luz = StandardScaler()
X_curva_luz_train = scaler_curva_luz.fit_transform(X_curva_luz_train)
X_curva_luz_test = scaler_curva_luz.transform(X_curva_luz_test)

scaler_ondas_gravitacionais = StandardScaler()
X_ondas_gravitacionais_train = scaler_ondas_gravitacionais.fit_transform(X_ondas_gravitacionais_train)
X_ondas_gravitacionais_test = scaler_ondas_gravitacionais.transform(X_ondas_gravitacionais_test)

# Treinamento do modelo para cada tipo de dado
model_espectro = MLPRegressor(hidden_layer_sizes=(100, 50))
model_espectro.fit(X_espectro_train, y_espectro_train)

model_curva_luz = MLPRegressor(hidden_layer_sizes=(100, 50))
model_curva_luz.fit(X_curva_luz_train, y_curva_luz_train)

model_ondas_gravitacionais = MLPRegressor(hidden_layer_sizes=(100, 50))
model_ondas_gravitacionais.fit(X_ondas_gravitacionais_train, y_ondas_gravitacionais_train)

# Previsão de dados projetados para cada tipo de dado
y_espectro_pred = model_espectro.predict(X_espectro_test)
y_curva_luz_pred = model_curva_luz.predict(X_curva_luz_test)
y_ondas_gravitacionais_pred = model_ondas_gravitacionais.predict(X_ondas_gravitacionais_test)

plt.plot(y_espectro_test[0], label='Real')
plt.plot(y_espectro_pred[0], label='Predito')
plt.legend()
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.title('Espectro Eletromagnético')
plt.show()

plt.plot(y_curva_luz_test[0], label='Real')
plt.plot(y_curva_luz_pred[0], label='Predito')
plt.legend()
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.title('Curva de luz')
plt.show()

plt.plot(y_ondas_gravitacionais_train[0], label='Real')
plt.plot(y_ondas_gravitacionais_pred[0], label='Predito')
plt.legend()
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.title('Ondas gravitacionais')
plt.show()