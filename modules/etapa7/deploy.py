"""
ETAPA 7 — Deploy e Infraestrutura de IA
Módulo para ensinar deploy e MLOps
"""
import streamlit as st


def render_etapa7():
    """Renderiza o conteúdo da Etapa 7"""
    
    st.title("🏗️ ETAPA 7 — Deploy e Infraestrutura de IA")
    st.markdown("**Duração:** Flexível")
    
    st.markdown("""
    Aprenda a colocar modelos de IA em produção de forma profissional, 
    escalável e confiável.
    """)
    
    # Tópicos
    st.header("📚 O que você vai aprender:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - ☁️ **GPUs no GCP/AWS/Azure**
        - 🐳 **Kubernetes para IA**
        - 🔄 **CI/CD para Modelos**
        - 📊 **MLflow + wandb**
        """)
    
    with col2:
        st.markdown("""
        - ⚙️ **MLOps End-to-End**
        - 📈 **Monitoring e Alertas**
        - 🔧 **Model Serving**
        - 🚀 **Escalabilidade**
        """)
    
    st.success("🎯 **Resultado:** Você se transforma em AI Platform Engineer — raríssimo no mercado.")
    
    st.markdown("---")
    
    tabs = st.tabs([
        "Cloud & GPUs",
        "Kubernetes",
        "MLOps",
        "Monitoring",
        "Exercícios"
    ])
    
    with tabs[0]:
        render_cloud()
    
    with tabs[1]:
        render_kubernetes()
    
    with tabs[2]:
        render_mlops()
    
    with tabs[3]:
        render_monitoring()
    
    with tabs[4]:
        render_exercicios_etapa7()


def render_cloud():
    """Cloud e GPUs"""
    st.subheader("☁️ Cloud Providers e GPUs")
    
    st.markdown("### AWS SageMaker")
    
    code_aws = """
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

# Configurar
session = sagemaker.Session()
role = "arn:aws:iam::ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole"

# Deploy modelo PyTorch
pytorch_model = PyTorchModel(
    model_data='s3://bucket/model.tar.gz',
    role=role,
    framework_version='2.0',
    py_version='py310',
    entry_point='inference.py'
)

# Deploy endpoint
predictor = pytorch_model.deploy(
    instance_type='ml.g4dn.xlarge',  # GPU instance
    initial_instance_count=1,
    endpoint_name='my-ai-endpoint'
)

# Fazer predição
result = predictor.predict(data)

# Autoscaling
client = boto3.client('application-autoscaling')

client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10
)

# Política de scaling
client.put_scaling_policy(
    PolicyName='target-tracking-policy',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,  # 70% utilização
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
"""
    
    st.code(code_aws, language="python")
    
    st.markdown("---")
    
    st.markdown("### Google Cloud Vertex AI")
    
    code_gcp = """
from google.cloud import aiplatform

# Inicializar
aiplatform.init(project='my-project', location='us-central1')

# Upload modelo
model = aiplatform.Model.upload(
    display_name='my-model',
    artifact_uri='gs://bucket/model/',
    serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/pytorch-gpu.1-11:latest'
)

# Deploy endpoint
endpoint = model.deploy(
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=5
)

# Predição
prediction = endpoint.predict(instances=[...])

# Batch prediction
batch_prediction_job = aiplatform.BatchPredictionJob.create(
    job_display_name='batch-prediction',
    model_name=model.resource_name,
    instances_format='jsonl',
    predictions_format='jsonl',
    gcs_source='gs://bucket/input/',
    gcs_destination_prefix='gs://bucket/output/'
)
"""
    
    st.code(code_gcp, language="python")
    
    st.markdown("### Comparação de Custos:")
    
    import pandas as pd
    custos = {
        "Provider": ["AWS", "GCP", "Azure"],
        "GPU (T4)": ["$0.526/h", "$0.35/h", "$0.526/h"],
        "GPU (A100)": ["$4.10/h", "$3.67/h", "$4.11/h"],
        "Inference": ["SageMaker", "Vertex AI", "ML Studio"],
        "Facilidade": ["⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐"]
    }
    
    df = pd.DataFrame(custos)
    st.table(df)


def render_kubernetes():
    """Kubernetes para IA"""
    st.subheader("🐳 Kubernetes para IA")
    
    st.markdown("""
    ### Deploy de Modelo com Kubernetes
    """)
    
    yaml = """
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-model-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-model
  template:
    metadata:
      labels:
        app: ai-model
    spec:
      containers:
      - name: model-server
        image: myregistry/ai-model:v1
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: MODEL_PATH
          value: "/models/model.pt"
        - name: WORKERS
          value: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ai-model-service
spec:
  selector:
    app: ai-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-model-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
    
    st.code(yaml, language="yaml")
    
    st.markdown("---")
    
    st.markdown("### KServe - Model Serving no Kubernetes")
    
    kserve_yaml = """
# kserve-model.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: pytorch-model
spec:
  predictor:
    pytorch:
      storageUri: gs://bucket/models/pytorch
      resources:
        requests:
          cpu: "1"
          memory: 2Gi
          nvidia.com/gpu: "1"
        limits:
          cpu: "2"
          memory: 4Gi
          nvidia.com/gpu: "1"
      # Autoscaling
      minReplicas: 1
      maxReplicas: 5
      scaleTarget: 10  # requests per second
      scaleMetric: concurrency
"""
    
    st.code(kserve_yaml, language="yaml")
    
    commands = """
# Deploy
kubectl apply -f kserve-model.yaml

# Ver status
kubectl get inferenceservice pytorch-model

# Fazer predição
curl -X POST \\
  http://pytorch-model.default.example.com/v1/models/pytorch-model:predict \\
  -H 'Content-Type: application/json' \\
  -d '{"instances": [[1.0, 2.0, 3.0]]}'
"""
    
    st.code(commands, language="bash")


def render_mlops():
    """MLOps"""
    st.subheader("⚙️ MLOps End-to-End")
    
    st.markdown("""
    ### Pipeline CI/CD para ML
    """)
    
    github_actions = """
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ --cov=src/
      
      - name: Lint
        run: |
          pip install black flake8
          black --check src/
          flake8 src/

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Train model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
        run: |
          pip install -r requirements.txt
          python train.py
      
      - name: Evaluate model
        run: python evaluate.py
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: model
          path: models/

  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v2
        with:
          name: model
      
      - name: Deploy to SageMaker
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_KEY }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET }}
        run: python deploy.py
"""
    
    st.code(github_actions, language="yaml")
    
    st.markdown("---")
    
    st.markdown("### MLflow + DVC para Versionamento")
    
    code_mlflow = """
# train.py com MLflow
import mlflow
import mlflow.pytorch
from dvclive import Live

# Configurar
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("production-model")

with mlflow.start_run() as run:
    # Iniciar DVCLive para métricas
    with Live(save_dvc_exp=True) as live:
        
        # Logar parâmetros
        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
        mlflow.log_params(params)
        
        # Treinar modelo
        for epoch in range(params["epochs"]):
            train_loss = train_epoch(model, train_loader)
            val_loss = validate(model, val_loader)
            
            # Logar métricas
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            live.log_metric("train_loss", train_loss)
            live.log_metric("val_loss", val_loss)
            live.next_step()
        
        # Avaliar
        metrics = evaluate_model(model, test_loader)
        mlflow.log_metrics(metrics)
        
        # Logar modelo
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name="ProductionModel"
        )
        
        # Logar artifacts
        mlflow.log_artifact("confusion_matrix.png")
        
        print(f"Run ID: {run.info.run_id}")

# Carregar melhor modelo
best_model = mlflow.pytorch.load_model("models:/ProductionModel/Production")
"""
    
    st.code(code_mlflow, language="python")


def render_monitoring():
    """Monitoring"""
    st.subheader("📈 Monitoring e Observabilidade")
    
    st.markdown("""
    ### Prometheus + Grafana
    """)
    
    code_monitoring = """
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Métricas
prediction_counter = Counter(
    'model_predictions_total',
    'Total de predições',
    ['model_name', 'version']
)

prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Latência das predições',
    ['model_name']
)

model_accuracy = Gauge(
    'model_accuracy',
    'Acurácia atual do modelo',
    ['model_name']
)

# Instrumentar API
@app.post("/predict")
async def predict(request: PredictionRequest):
    start = time.time()
    
    try:
        # Fazer predição
        result = model.predict(request.data)
        
        # Incrementar contador
        prediction_counter.labels(
            model_name='my-model',
            version='v1'
        ).inc()
        
        # Registrar latência
        latency = time.time() - start
        prediction_latency.labels(
            model_name='my-model'
        ).observe(latency)
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

# Iniciar servidor de métricas
start_http_server(8001)
"""
    
    st.code(code_monitoring, language="python")
    
    st.markdown("---")
    
    st.markdown("### Data Drift Detection")
    
    code_drift = """
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Dados de referência (treino)
reference_data = pd.read_csv("training_data.csv")

# Dados de produção
production_data = pd.read_csv("production_data.csv")

# Criar relatório
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset()
])

report.run(
    reference_data=reference_data,
    current_data=production_data,
    column_mapping=ColumnMapping()
)

# Salvar
report.save_html("drift_report.html")

# Verificar drift
drift_score = report.as_dict()["metrics"][0]["result"]["drift_score"]

if drift_score > 0.3:
    # Alertar
    send_alert(f"Data drift detectado! Score: {drift_score}")
    # Retreinar modelo
    trigger_retraining()
"""
    
    st.code(code_drift, language="python")
    
    st.info("""
    💡 **Monitorar:**
    - Latência e throughput
    - Acurácia em produção
    - Data drift
    - Concept drift
    - Resource usage
    - Erros e exceções
    """)


def render_exercicios_etapa7():
    """Exercícios da Etapa 7"""
    st.subheader("💪 Exercícios Práticos")
    
    exercicios = [
        {
            "titulo": "1. Deploy Completo na Cloud",
            "descricao": "Deploy um modelo em produção com autoscaling.",
            "requisitos": [
                "Escolher AWS/GCP/Azure",
                "Containerizar modelo",
                "Deploy com GPU",
                "Configurar autoscaling",
                "Load balancer",
                "Monitoramento básico"
            ]
        },
        {
            "titulo": "2. Pipeline MLOps",
            "descricao": "Criar pipeline completo de CI/CD para ML.",
            "requisitos": [
                "GitHub Actions ou GitLab CI",
                "Testes automatizados",
                "Treino automatizado",
                "Registro no MLflow",
                "Deploy automático",
                "Rollback strategy"
            ]
        },
        {
            "titulo": "3. Kubernetes Deployment",
            "descricao": "Deploy modelo usando Kubernetes.",
            "requisitos": [
                "Cluster K8s (local ou cloud)",
                "Deploy com Helm",
                "HPA configurado",
                "Ingress/Load balancer",
                "Monitoring com Prometheus",
                "Logging centralizado"
            ]
        },
        {
            "titulo": "4. Sistema de Monitoring",
            "descricao": "Implementar monitoring completo.",
            "requisitos": [
                "Prometheus + Grafana",
                "Métricas de negócio",
                "Data drift detection",
                "Alertas automáticos",
                "Dashboard executivo",
                "Retraining automático"
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
        "Sei fazer deploy em cloud (AWS/GCP/Azure)",
        "Domino Kubernetes básico",
        "Consigo criar pipelines CI/CD",
        "Uso MLflow para tracking",
        "Implementei monitoring com Prometheus",
        "Sei detectar data drift",
        "Entendo autoscaling",
        "Domino troubleshooting em produção"
    ]
    
    for i, item in enumerate(checklist):
        st.checkbox(item, key=f"check_etapa7_{i}")

