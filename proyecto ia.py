# Importar las bibliotecas necesarias
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# 1. Cargar y preprocesar los datos
# Normalización para escalar los valores de píxeles entre -1 y 1
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Dataset de entrenamiento (dígitos escritos a mano)
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# Cargador de datos para procesar los datos en lotes
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Dataset de prueba para evaluar el modelo
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# Cargador de datos para pruebas
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 2. Definir la red neuronal
class Net(nn.Module):  # Heredamos de nn.Module para construir el modelo
    def __init__(self):
        super(Net, self).__init__()
        # Primera capa: de entrada (784 características) a 128 neuronas
        self.fc1 = nn.Linear(28 * 28, 128)
        # Segunda capa: de 128 neuronas a 64
        self.fc2 = nn.Linear(128, 64)
        # Capa de salida: de 64 a 10 neuronas (10 clases)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Aplanar la imagen (28x28) en un vector de 784 elementos
        x = x.view(-1, 28 * 28)
        # Aplicar la función de activación ReLU después de cada capa
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Salida de la última capa sin función de activación (logits)
        x = self.fc3(x)
        return x

# Crear una instancia del modelo
net = Net()

# 3. Configurar el entrenamiento
# Definir la función de pérdida (entropía cruzada para clasificación)
criterion = nn.CrossEntropyLoss()
# Optimizador para actualizar los pesos (descenso de gradiente estocástico)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 4. Entrenar el modelo
for epoch in range(5):  # Número de épocas (iteraciones completas sobre el conjunto de datos)
    running_loss = 0.0
    for inputs, labels in trainloader:  # Procesar cada lote
        optimizer.zero_grad()  # Limpiar los gradientes previos
        outputs = net(inputs)  # Hacer predicciones
        loss = criterion(outputs, labels)  # Calcular la pérdida
        loss.backward()  # Retropropagar el error
        optimizer.step()  # Actualizar los pesos
        running_loss += loss.item()
    # Imprimir la pérdida promedio por época
    print(f"Época {epoch + 1}, Pérdida: {running_loss / len(trainloader)}")

# 5. Evaluar el modelo
correct = 0
total = 0
with torch.no_grad():  # No calcular gradientes durante la evaluación
    for inputs, labels in testloader:
        outputs = net(inputs)  # Hacer predicciones
        _, predicted = torch.max(outputs.data, 1)  # Obtener la clase con mayor probabilidad
        total += labels.size(0)  # Contar el total de etiquetas
        correct += (predicted == labels).sum().item()  # Contar las predicciones correctas

# Imprimir la precisión final del modelo
print(f"Precisión en el conjunto de prueba: {100 * correct / total}%")
