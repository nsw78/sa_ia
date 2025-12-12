"""
ETAPA 3 — Deep Learning + PyTorch
Módulo para ensinar redes neurais e PyTorch
"""
import streamlit as st


def render_etapa3():
    """Renderiza o conteúdo da Etapa 3"""
    
    st.title("🧠 ETAPA 3 — Deep Learning + PyTorch")
    st.markdown("**Duração:** 10 dias")
    
    st.markdown("""
    Mergulhe no mundo das redes neurais e aprenda PyTorch, o framework mais usado 
    em pesquisa e cada vez mais em produção.
    """)
    
    # Tópicos
    st.header("📚 O que você vai dominar:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 🔢 **Tensores e Operações**
        - 🧠 **Redes Neurais Artificiais**
        - 👁️ **CNN (Visão Computacional)**
        - 🔄 **RNN/LSTM (Sequências)**
        """)
    
    with col2:
        st.markdown("""
        - 🤖 **Transformers**
        - ⚙️ **Otimizadores (Adam, SGD)**
        - 📉 **Funções de Perda**
        - 🔁 **Training Loops**
        """)
    
    st.success("🎯 **Resultado:** Capaz de treinar modelos neurais reais do zero.")
    
    st.markdown("---")
    
    tabs = st.tabs([
        "Tensores & Básico",
        "Redes Neurais",
        "CNN",
        "RNN/LSTM",
        "Transformers",
        "Exercícios"
    ])
    
    with tabs[0]:
        render_tensores()
    
    with tabs[1]:
        render_redes_neurais()
    
    with tabs[2]:
        render_cnn()
    
    with tabs[3]:
        render_rnn()
    
    with tabs[4]:
        render_transformers()
    
    with tabs[5]:
        render_exercicios_etapa3()


def render_tensores():
    """Seção de Tensores"""
    st.subheader("🔢 Tensores e PyTorch Básico")
    
    st.markdown("""
    ### O que são Tensores?
    
    Tensores são arrays multidimensionais, a estrutura básica do PyTorch.
    """)
    
    code = """
import torch
import numpy as np

# Criar tensores
tensor_zeros = torch.zeros(3, 4)
tensor_ones = torch.ones(2, 3)
tensor_random = torch.randn(3, 3)  # distribuição normal

# De numpy para tensor
arr = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(arr)

# Operações básicas
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Soma
c = a + b
print("Soma:\\n", c)

# Multiplicação elemento a elemento
d = a * b
print("\\nMultiplicação elemento a elemento:\\n", d)

# Multiplicação de matrizes
e = torch.matmul(a, b)
print("\\nMultiplicação de matrizes:\\n", e)

# Reshape
f = a.view(1, 4)  # ou a.reshape(1, 4)
print("\\nReshape:\\n", f)

# GPU (se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_gpu = a.to(device)
print(f"\\nDispositivo: {tensor_gpu.device}")
"""
    
    st.code(code, language="python")
    
    st.markdown("---")
    
    st.markdown("""
    ### Autograd - Diferenciação Automática
    
    O poder do PyTorch está no cálculo automático de gradientes.
    """)
    
    code_autograd = """
import torch

# Tensor com gradiente
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# Operação
z = x**2 + y**3

# Calcular gradientes
z.backward()

# Ver gradientes
print(f"dz/dx = {x.grad}")  # 2x = 4
print(f"dz/dy = {y.grad}")  # 3y² = 27

# Exemplo mais complexo
x = torch.randn(5, requires_grad=True)
y = x * 2

while y.data.norm() < 1000:
    y = y * 2

# Gradiente de função não-escalar
gradients = torch.tensor([0.1, 1.0, 0.0001, 0.1, 0.01])
y.backward(gradients)

print(f"\\nGradiente de x: {x.grad}")
"""
    
    st.code(code_autograd, language="python")
    
    st.info("💡 **Autograd é fundamental:** Permite treinar redes neurais via backpropagation!")


def render_redes_neurais():
    """Seção de Redes Neurais"""
    st.subheader("🧠 Redes Neurais Artificiais")
    
    st.markdown("""
    ### Criar uma Rede Neural Simples
    
    Vamos criar um classificador MNIST do zero.
    """)
    
    code = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Definir a arquitetura
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Hiperparâmetros
input_size = 784  # 28x28
hidden_size = 128
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Modelo, loss e otimizador
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Preparar dados
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    transform=transform, 
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True
)

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')

# Salvar modelo
torch.save(model.state_dict(), 'model.pth')
"""
    
    st.code(code, language="python")
    
    st.markdown("### Componentes Principais:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Arquitetura:**
        - `nn.Module`: classe base
        - `nn.Linear`: camada densa
        - `nn.ReLU`: ativação
        - `forward()`: propagação
        """)
    
    with col2:
        st.markdown("""
        **Treinamento:**
        - Loss function (criterion)
        - Optimizer (Adam, SGD)
        - zero_grad() + backward()
        - optimizer.step()
        """)


def render_cnn():
    """Seção de CNN"""
    st.subheader("👁️ CNN - Redes Convolucionais")
    
    st.markdown("""
    ### Arquitetura CNN para Imagens
    
    CNNs são o padrão para visão computacional.
    """)
    
    code = """
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Camadas convolucionais
        self.conv1 = nn.Conv2d(
            in_channels=1,    # imagens grayscale
            out_channels=32,  # 32 filtros
            kernel_size=3,    # filtro 3x3
            padding=1
        )
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout para regularização
        self.dropout = nn.Dropout(0.25)
        
        # Camadas fully connected
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(x)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Criar modelo
model = CNN()
print(model)

# Ver número de parâmetros
total_params = sum(p.numel() for p in model.parameters())
print(f"\\nTotal de parâmetros: {total_params:,}")
"""
    
    st.code(code, language="python")
    
    st.markdown("---")
    
    st.markdown("""
    ### Transfer Learning com Modelos Pré-treinados
    """)
    
    code_transfer = """
import torchvision.models as models
import torch.nn as nn

# Carregar ResNet pré-treinado
model = models.resnet18(pretrained=True)

# Congelar camadas
for param in model.parameters():
    param.requires_grad = False

# Substituir última camada
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10 classes

# Apenas a última camada será treinada
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Para fine-tuning completo depois:
# for param in model.parameters():
#     param.requires_grad = True
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
"""
    
    st.code(code_transfer, language="python")
    
    st.success("✅ Transfer learning acelera treinamento e melhora resultados!")


def render_rnn():
    """Seção de RNN/LSTM"""
    st.subheader("🔄 RNN/LSTM - Dados Sequenciais")
    
    st.markdown("""
    ### LSTM para Séries Temporais
    
    LSTMs capturam dependências de longo prazo em sequências.
    """)
    
    code = """
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully connected
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Inicializar hidden e cell states
        h0 = torch.zeros(self.num_layers, x.size(0), 
                        self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), 
                        self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Pegar output do último time step
        out = self.fc(out[:, -1, :])
        
        return out

# Exemplo de uso
sequence_length = 50
input_size = 1
hidden_size = 128
num_layers = 2
num_classes = 1

model = LSTM(input_size, hidden_size, num_layers, num_classes)

# Input shape: (batch_size, sequence_length, input_size)
x = torch.randn(32, sequence_length, input_size)
output = model(x)
print(f"Output shape: {output.shape}")
"""
    
    st.code(code, language="python")
    
    st.markdown("### Aplicações de RNN/LSTM:")
    
    st.markdown("""
    - 📈 **Previsão de séries temporais**
    - 📝 **Geração de texto**
    - 🗣️ **Processamento de linguagem**
    - 🎵 **Geração de música**
    - 💰 **Previsão de preços**
    """)


def render_transformers():
    """Seção de Transformers"""
    st.subheader("🤖 Transformers - Atenção é Tudo")
    
    st.markdown("""
    ### Arquitetura Transformer
    
    Transformers revolucionaram IA e são base de LLMs modernos.
    """)
    
    code = """
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
    
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Exemplo de uso
d_model = 512
num_heads = 8
batch_size = 2
seq_length = 10

mha = MultiHeadAttention(d_model, num_heads)

x = torch.randn(batch_size, seq_length, d_model)
output = mha(x, x, x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
"""
    
    st.code(code, language="python")
    
    st.markdown("---")
    
    st.markdown("""
    ### Usando Transformers do Hugging Face
    """)
    
    code_hf = """
from transformers import AutoModel, AutoTokenizer
import torch

# Carregar modelo pré-treinado
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Preparar input
text = "Transformers mudaram o mundo da IA!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Obter embeddings
embeddings = outputs.last_hidden_state
print(f"Embeddings shape: {embeddings.shape}")

# Para classificação
from transformers import AutoModelForSequenceClassification

classifier = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2
)
"""
    
    st.code(code_hf, language="python")
    
    st.info("💡 **Transformers são a base de:** GPT, BERT, T5, Claude, etc!")


def render_exercicios_etapa3():
    """Exercícios da Etapa 3"""
    st.subheader("💪 Exercícios Práticos")
    
    exercicios = [
        {
            "titulo": "1. Classificador de Imagens",
            "descricao": "Crie uma CNN para classificar imagens do CIFAR-10.",
            "requisitos": [
                "Arquitetura CNN customizada",
                "Data augmentation",
                "Training loop completo",
                "Validação e early stopping",
                "Visualizar predições",
                "Acurácia > 75%"
            ]
        },
        {
            "titulo": "2. Previsão de Séries Temporais",
            "descricao": "Use LSTM para prever preços de ações ou temperatura.",
            "requisitos": [
                "Preparar dados sequenciais",
                "Normalização adequada",
                "LSTM com múltiplas camadas",
                "Avaliar com RMSE",
                "Visualizar predições vs real",
                "Experimentar diferentes janelas"
            ]
        },
        {
            "titulo": "3. Transfer Learning",
            "descricao": "Use ResNet pré-treinado para classificação customizada.",
            "requisitos": [
                "Dataset customizado (Kaggle)",
                "Fine-tuning estratégico",
                "Comparar com treino from scratch",
                "Data augmentation",
                "Alcançar > 90% acurácia",
                "Deploy com FastAPI"
            ]
        },
        {
            "titulo": "4. Sentiment Analysis",
            "descricao": "Classifique sentimento de reviews usando Transformers.",
            "requisitos": [
                "Usar BERT ou DistilBERT",
                "Fine-tune no dataset",
                "Tokenização adequada",
                "Métricas: Accuracy, F1",
                "Inference com novos textos",
                "API de classificação"
            ]
        }
    ]
    
    for ex in exercicios:
        with st.expander(f"{ex['titulo']}"):
            st.markdown(f"**Descrição:** {ex['descricao']}")
            st.markdown("**Requisitos:**")
            for req in ex['requisitos']:
                st.markdown(f"- {req}")
    
    st.markdown("---")
    st.markdown("### 📋 Checklist de Domínio:")
    
    checklist = [
        "Domino operações com tensores",
        "Entendo autograd e backpropagation",
        "Sei criar redes neurais do zero",
        "Domino CNNs para visão computacional",
        "Entendo RNN/LSTM para sequências",
        "Conheço arquitetura Transformer",
        "Sei fazer transfer learning",
        "Consigo treinar modelos end-to-end"
    ]
    
    for i, item in enumerate(checklist):
        st.checkbox(item, key=f"check_etapa3_{i}")

