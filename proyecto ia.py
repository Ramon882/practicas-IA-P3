# Importar las bibliotecas necesarias
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Cargar el dataset MNIST
# Transformación para convertir imágenes a tensores
transform = transforms.Compose([transforms.ToTensor()])

# Cargar el dataset de entrenamiento y prueba
train = datasets.MNIST('', train=True, download=True, transform=transform)
test = datasets.MNIST('', train=False, download=True, transform=transform)

# Cargadores de datos para procesar los datos en lotes
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

# 2. Definir la clase de la red neuronal
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Definir las capas completamente conectadas
        self.fc1 = nn.Linear(28 * 28, 64)  # Entrada: 28x28=784, salida: 64
        self.fc2 = nn.Linear(64, 64)       # Capa oculta 1
        self.fc3 = nn.Linear(64, 64)       # Capa oculta 2
        self.fc4 = nn.Linear(64, 10)       # Capa de salida: 10 clases (dígitos 0-9)

    def forward(self, x):
        # Propagación hacia adelante con activaciones ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # Aplicar log_softmax a la salida
        return F.log_softmax(x, dim=1)

# 3. Inicializar la red neuronal
net = Net()

# 4. Definir la función de pérdida y el optimizador
loss_function = nn.CrossEntropyLoss()  # Para clasificación multiclase
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Optimización con Adam

# 5. Entrenar la red neuronal
for epoch in range(3):  # Número de épocas de entrenamiento
    for data in trainset:  # Iterar sobre lotes de datos de entrenamiento
        X, y = data  # Obtener características (X) y etiquetas (y)
        net.zero_grad()  # Limpiar gradientes acumulados
        output = net(X.view(-1, 28 * 28))  # Aplanar la imagen y pasarla por la red
        loss = loss_function(output, y)  # Calcular la pérdida
        loss.backward()  # Retropropagar el error
        optimizer.step()  # Actualizar los pesos de la red
    print(f"Época {epoch + 1}, Pérdida: {loss.item()}")

# 6. Evaluar el modelo en el conjunto de prueba
correct = 0
total = 0

with torch.no_grad():  # Deshabilitar cálculo de gradientes para eficiencia
    for data in testset:  # Iterar sobre el conjunto de prueba
        X, y = data
        output = net(X.view(-1, 28 * 28))  # Aplanar y pasar por la red
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:  # Comparar la predicción con la etiqueta real
                correct += 1
            total += 1

print(f"Precisión en el conjunto de prueba: {round(correct / total, 3)}")

# 7. Visualizar una muestra de la imagen y la predicción
sample = next(iter(testset))  # Obtener una muestra del conjunto de prueba
image, label = sample[0][0], sample[1][0]  # Extraer la imagen y su etiqueta

# Mostrar la imagen
plt.imshow(image.view(28, 28), cmap="gray")
plt.show()

# Mostrar la predicción
predicted_label = torch.argmax(net(image.view(-1, 28 * 28))[0])
print(f"Etiqueta real: {label}, Etiqueta predicha: {predicted_label}")

