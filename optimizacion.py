import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Definir la función de Convolución Separable en Profundidad
class ConvolucionSeparableProfundidad(nn.Module):
    def __init__(self, canales_entrada, canales_salida, tamaño_kernel, stride=1, padding=0):
        super(ConvolucionSeparableProfundidad, self).__init__()
        self.profundidad = nn.Conv1d(canales_entrada, canales_entrada, kernel_size=tamaño_kernel,
                                     stride=stride, padding=padding, groups=canales_entrada)
        self.puntual = nn.Conv1d(canales_entrada, canales_salida, kernel_size=1)

    def forward(self, x):
        x = self.profundidad(x)
        x = self.puntual(x)
        return x

# Definir el modelo TextCNN
class TextCNN(nn.Module):
    def __init__(self, tamaño_vocabulario, dim_embedding):
        super(TextCNN, self).__init__()
        self.emb = nn.Embedding(tamaño_vocabulario, dim_embedding)
        self.conv1 = nn.Conv1d(dim_embedding, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(64, 10)  # Ajustar el tamaño de salida según la tarea de clasificación

    def forward(self, x):
        x = self.emb(x).transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = x.mean(dim=2)
        return self.fc(x)

# Interfaz de Streamlit
st.title("Demo de Convolución Separable en Profundidad y Text CNN")

# Selección del modelo
eleccion_modelo = st.selectbox("Elige el modelo", ["Convolución Separable en Profundidad", "Text CNN"])

if eleccion_modelo == "Convolución Separable en Profundidad":
    st.write("La Convolución Separable en Profundidad reduce el número de parámetros.")
    datos_entrada = st.text_input("Introduce un array 1D de valores (separados por comas):", "1,2,3,4,5")
    array_entrada = np.array([float(x) for x in datos_entrada.split(",")])
    tensor_entrada = torch.tensor(array_entrada).float().unsqueeze(0).unsqueeze(0)  # Forma (1, 1, N)
    
    modelo = ConvolucionSeparableProfundidad(canales_entrada=1, canales_salida=1, tamaño_kernel=3, padding=1)
    salida = modelo(tensor_entrada)
    resultado = salida.squeeze().detach().numpy()
    st.write("Salida:", resultado)
    
    # Gráfico del resultado de la convolución
    fig, ax = plt.subplots()
    ax.plot(array_entrada, label="Entrada")
    ax.plot(resultado, label="Salida", linestyle='--')
    ax.set_title("Resultado de la Convolución Separable en Profundidad")
    ax.legend()
    st.pyplot(fig)

elif eleccion_modelo == "Text CNN":
    st.write("Modelo Text CNN para clasificación de secuencias.")
    tamaño_vocabulario = st.slider("Tamaño del Vocabulario", min_value=10, max_value=1000, value=50)
    dim_embedding = st.slider("Dimensión del Embedding", min_value=2, max_value=300, value=100)
    
    modelo = TextCNN(tamaño_vocabulario, dim_embedding)
    datos_entrada = st.text_input("Introduce una secuencia de enteros (por ejemplo, '1,2,3') para el embedding:", "1,2,3,4")
    secuencia_entrada = torch.tensor([int(x) for x in datos_entrada.split(",")])
    
    with torch.no_grad():
        salida = modelo(secuencia_entrada.unsqueeze(0))
    resultado = salida.squeeze().numpy()
    st.write("Salida:", resultado)
    
    # Gráfico de las salidas de la capa final
    fig, ax = plt.subplots()
    ax.bar(range(len(resultado)), resultado)
    ax.set_title("Distribución de la salida de Text CNN")
    ax.set_xlabel("Clases")
    ax.set_ylabel("Valor")
    st.pyplot(fig)
