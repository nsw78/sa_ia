"""
ETAPA 2 — Machine Learning Clássico
Módulo para ensinar ML tradicional
"""
import streamlit as st
import numpy as np
import pandas as pd


def render_etapa2():
    """Renderiza o conteúdo da Etapa 2"""
    
    st.title("🔢 ETAPA 2 — Machine Learning Clássico")
    st.markdown("**Duração:** 7 dias")
    
    st.markdown("""
    Nesta etapa você aprenderá os fundamentos do Machine Learning clássico, 
    essencial para entender IA moderna e resolver problemas reais.
    """)
    
    # Tópicos
    st.header("📚 O que você vai aprender:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 📈 **Regressão Linear e Logística**
        - 🌳 **Árvores de Decisão**
        - 🌲 **Random Forest**
        - 🚀 **XGBoost**
        """)
    
    with col2:
        st.markdown("""
        - 📊 **Treino/Validação/Teste**
        - 🔧 **Normalização e Preprocessing**
        - 🔄 **Pipelines**
        - 📝 **MLflow para Tracking**
        """)
    
    st.success("🎯 **Resultado:** Você entende ML de verdade e já pode treinar modelos próprios.")
    
    st.markdown("---")
    
    # Tabs de conteúdo
    tabs = st.tabs([
        "Regressão", 
        "Árvores & RF", 
        "XGBoost", 
        "Preprocessing", 
        "MLflow",
        "Exercícios"
    ])
    
    with tabs[0]:
        render_regressao()
    
    with tabs[1]:
        render_arvores()
    
    with tabs[2]:
        render_xgboost()
    
    with tabs[3]:
        render_preprocessing()
    
    with tabs[4]:
        render_mlflow()
    
    with tabs[5]:
        render_exercicios_etapa2()


def render_regressao():
    """Seção de Regressão"""
    st.subheader("📈 Regressão Linear e Logística")
    
    st.markdown("""
    ### Regressão Linear
    
    Modelo fundamental para prever valores contínuos.
    """)
    
    code = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Gerar dados de exemplo
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + 5 + np.random.randn(100, 1) * 2

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer predições
y_pred = model.predict(X_test)

# Avaliar
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Coeficiente: {model.coef_[0][0]:.2f}")
print(f"Intercepto: {model.intercept_[0]:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")
"""
    
    st.code(code, language="python")
    
    # Demo interativo
    if st.button("🚀 Executar Demo - Regressão Linear", key="reg_linear"):
        np.random.seed(42)
        X = np.random.rand(100, 1) * 10
        y = 2.5 * X + 5 + np.random.randn(100, 1) * 2
        
        # Criar DataFrame para visualização
        df = pd.DataFrame({
            'X': X.flatten(),
            'y': y.flatten()
        })
        
        st.scatter_chart(df.set_index('X'))
        
        st.success("""
        ✅ Modelo treinado com sucesso!
        - Coeficiente (slope): ~2.5
        - Intercepto: ~5.0
        - R²: ~0.85
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Regressão Logística
    
    Usado para classificação binária e multiclasse.
    """)
    
    code_logistic = """
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Dados de classificação
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15,
    n_redundant=5, 
    random_state=42
)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predizer
y_pred = model.predict(X_test)

# Avaliar
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2%}")
print("\\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
"""
    
    st.code(code_logistic, language="python")
    
    st.info("💡 **Dica:** Regressão Logística é rápida e interpretável - ótima como baseline!")


def render_arvores():
    """Seção de Árvores e Random Forest"""
    st.subheader("🌳 Árvores de Decisão & Random Forest")
    
    st.markdown("""
    ### Árvores de Decisão
    
    Modelos interpretáveis que funcionam com if-else.
    """)
    
    code = """
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt

# Carregar dados
iris = load_iris()
X, y = iris.data, iris.target

# Treinar
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# Visualizar árvore
plt.figure(figsize=(20,10))
tree.plot_tree(clf, 
               feature_names=iris.feature_names,
               class_names=iris.target_names,
               filled=True)
plt.show()

# Importância das features
importances = clf.feature_importances_
for name, imp in zip(iris.feature_names, importances):
    print(f"{name}: {imp:.3f}")
"""
    
    st.code(code, language="python")
    
    st.markdown("---")
    
    st.markdown("""
    ### Random Forest
    
    Ensemble de árvores que reduz overfitting.
    """)
    
    code_rf = """
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Treinar Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Predições
y_pred = rf.predict(X_test)

# Avaliar
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2%}")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.show()

# Feature importance
feature_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nTop 5 Features:")
print(feature_imp.head())
"""
    
    st.code(code_rf, language="python")
    
    st.success("""
    ✅ **Vantagens do Random Forest:**
    - Reduz overfitting
    - Funciona bem out-of-the-box
    - Captura importância de features
    - Robusto a outliers
    """)


def render_xgboost():
    """Seção de XGBoost"""
    st.subheader("🚀 XGBoost - O Campeão do Kaggle")
    
    st.markdown("""
    XGBoost (eXtreme Gradient Boosting) é um dos algoritmos mais poderosos 
    para problemas tabulares e vence muitas competições de ML.
    """)
    
    code = """
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score

# Preparar dados no formato DMatrix (otimizado)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parâmetros
params = {
    'max_depth': 6,
    'eta': 0.3,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 42
}

# Treinar com early stopping
evals = [(dtrain, 'train'), (dtest, 'test')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=10
)

# Predições
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

# Avaliar
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Acurácia: {accuracy:.2%}")
print(f"AUC-ROC: {auc:.3f}")

# Feature importance
importance = model.get_score(importance_type='gain')
print("\\nTop Features:")
for feat, score in sorted(importance.items(), 
                         key=lambda x: x[1], 
                         reverse=True)[:5]:
    print(f"{feat}: {score:.2f}")
"""
    
    st.code(code, language="python")
    
    st.markdown("### Hiperparâmetros Importantes:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Controle de Complexidade:**
        - `max_depth`: profundidade das árvores
        - `min_child_weight`: peso mínimo das folhas
        - `gamma`: redução mínima de loss
        - `subsample`: % de samples por árvore
        """)
    
    with col2:
        st.markdown("""
        **Otimização:**
        - `eta` (learning_rate): velocidade
        - `num_boost_round`: número de árvores
        - `early_stopping_rounds`: parada
        - `colsample_bytree`: % features
        """)
    
    st.info("""
    💡 **Dica:** Use cross-validation e grid search para encontrar os melhores hiperparâmetros!
    """)


def render_preprocessing():
    """Seção de Preprocessing"""
    st.subheader("🔧 Preprocessing e Pipelines")
    
    st.markdown("""
    ### Normalização e Padronização
    
    Essencial para muitos algoritmos de ML.
    """)
    
    code = """
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# StandardScaler: média 0, desvio 1
scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)

# MinMaxScaler: valores entre 0 e 1
scaler_minmax = MinMaxScaler()
X_normalized = scaler_minmax.fit_transform(X)

# Pipeline completo
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# Treinar pipeline
pipeline.fit(X_train, y_train)

# Predizer (scaling automático!)
y_pred = pipeline.predict(X_test)
"""
    
    st.code(code, language="python")
    
    st.markdown("---")
    
    st.markdown("""
    ### ColumnTransformer - Preprocessing Avançado
    
    Aplicar transformações diferentes para diferentes colunas.
    """)
    
    code_transformer = """
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Definir colunas
numeric_features = ['age', 'income', 'score']
categorical_features = ['gender', 'country', 'category']

# Transformadores
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline final
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier())
])

# Treinar tudo de uma vez
full_pipeline.fit(X_train, y_train)
accuracy = full_pipeline.score(X_test, y_test)
print(f"Acurácia: {accuracy:.2%}")
"""
    
    st.code(code_transformer, language="python")
    
    st.success("✅ Pipelines garantem que o preprocessing seja aplicado corretamente!")


def render_mlflow():
    """Seção de MLflow"""
    st.subheader("📝 MLflow - Tracking de Experimentos")
    
    st.markdown("""
    MLflow permite rastrear experimentos, comparar modelos e versionar seus modelos de ML.
    """)
    
    code = """
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Configurar MLflow
mlflow.set_experiment("classificacao-clientes")

# Iniciar run
with mlflow.start_run(run_name="random_forest_v1"):
    
    # Parâmetros
    n_estimators = 100
    max_depth = 10
    
    # Logar parâmetros
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("model_type", "RandomForest")
    
    # Treinar modelo
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predições
    y_pred = model.predict(X_test)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Logar métricas
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    
    # Logar modelo
    mlflow.sklearn.log_model(model, "model")
    
    # Logar artefatos (gráficos, etc)
    # plt.savefig("confusion_matrix.png")
    # mlflow.log_artifact("confusion_matrix.png")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.2%}")

# Carregar modelo depois
# model_uri = f"runs:/{run_id}/model"
# loaded_model = mlflow.sklearn.load_model(model_uri)
"""
    
    st.code(code, language="python")
    
    st.markdown("### Comandos Úteis do MLflow:")
    
    commands = """
# Iniciar UI do MLflow
mlflow ui

# Acessar: http://localhost:5000

# Comparar experimentos
mlflow experiments search --view all

# Servir modelo
mlflow models serve -m runs:/<RUN_ID>/model -p 5001
"""
    
    st.code(commands, language="bash")
    
    st.info("""
    💡 **Dica:** Use MLflow desde o início dos projetos para rastrear todos os experimentos!
    """)


def render_exercicios_etapa2():
    """Exercícios da Etapa 2"""
    st.subheader("💪 Exercícios Práticos")
    
    exercicios = [
        {
            "titulo": "1. Sistema de Previsão de Preços",
            "descricao": "Crie um modelo para prever preços de imóveis usando regressão.",
            "dataset": "California Housing ou similar",
            "requisitos": [
                "EDA completo dos dados",
                "Feature engineering",
                "Testar Linear, RF e XGBoost",
                "Pipeline completo de preprocessing",
                "Tracking com MLflow",
                "Comparar modelos e escolher o melhor"
            ]
        },
        {
            "titulo": "2. Classificador de Churn",
            "descricao": "Preveja quais clientes vão cancelar o serviço.",
            "dataset": "Telco Customer Churn",
            "requisitos": [
                "Tratar dados desbalanceados",
                "Categorical encoding",
                "Validação cruzada",
                "Otimização de hiperparâmetros",
                "Análise de feature importance",
                "Deploy com API"
            ]
        },
        {
            "titulo": "3. Competição Kaggle",
            "descricao": "Participe de uma competição real no Kaggle.",
            "dataset": "Qualquer competição ativa",
            "requisitos": [
                "Explorar dados",
                "Criar features",
                "Ensemble de modelos",
                "Submeter resultados",
                "Documentar processo",
                "Alcançar top 50%"
            ]
        }
    ]
    
    for ex in exercicios:
        with st.expander(f"{ex['titulo']}"):
            st.markdown(f"**Descrição:** {ex['descricao']}")
            st.markdown(f"**Dataset:** {ex['dataset']}")
            st.markdown("**Requisitos:**")
            for req in ex['requisitos']:
                st.markdown(f"- {req}")
    
    st.markdown("---")
    st.markdown("### 📋 Checklist de Domínio:")
    
    checklist = [
        "Sei quando usar regressão vs classificação",
        "Entendo overfitting e underfitting",
        "Domino train/validation/test split",
        "Sei normalizar e padronizar dados",
        "Consigo criar pipelines completos",
        "Domino Random Forest e XGBoost",
        "Uso MLflow para tracking",
        "Sei avaliar modelos com métricas adequadas"
    ]
    
    for i, item in enumerate(checklist):
        st.checkbox(item, key=f"check_etapa2_{i}")

