# Churn Prediction API

API REST para previsão de cancelamento de clientes utilizando Machine Learning. Desenvolvida com FastAPI, scikit-learn, XGBoost e Docker.

---

## Visão Geral

Este projeto treina um modelo de classificação binária no dataset [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) e serve previsões via API REST. O objetivo é prever se um cliente vai cancelar o serviço com base no seu perfil e dados de uso.

**Arquitetura:**

```
train.py          → carrega, limpa, encoda os dados e treina os modelos
cleaning.py       → pipeline de limpeza e pré-processamento
model.py          → definição dos modelos (Regressão Logística, Random Forest, XGBoost)
constants.py      → definição compartilhada das colunas de features
app/
├── main.py       → aplicação FastAPI e definição das rotas
├── schema.py     → validação do request com Pydantic
└── predictor.py  → encoding + inferência do modelo
```

---

## Modelos Avaliados

Três modelos foram treinados e comparados utilizando `classification_report`:

| Modelo | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|
| Random Forest | 0.62 | 0.45 | 0.53 |
| XGBoost | 0.51 | 0.75 | 0.60 |
| Regressão Logística | 0.50 | 0.79 | 0.61 |

**Modelo selecionado: XGBoost** — melhor equilíbrio entre recall e precision na classe minoritária (churn), com `scale_pos_weight=2.77` para lidar com o desbalanceamento de classes.

---

## Estrutura do Projeto

```
churn-model-api/
├── app/
│   ├── main.py
│   ├── schema.py
│   └── predictor.py
├── data/
│   └── .gitkeep
├── model/
│   └── .gitkeep
├── cleaning.py
├── constants.py
├── model.py
├── train.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/churn-model-api.git
cd churn-model-api
```

### 2. Configure o ambiente

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 3. Baixe o dataset

Baixe o [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) e coloque o arquivo CSV dentro da pasta `data/`.

### 4. Treine o modelo

```bash
python train.py
```

Isso irá gerar `model/churn_model.pkl` e `model/expected_columns.pkl`.

### 5. Execute a API

```bash
uvicorn app.main:app --reload
```

Acesse a documentação interativa em: `http://localhost:8000/docs`

---

## Executando com Docker

### Build da imagem

```bash
docker build -t churn-api .
```

### Execução do container

```bash
docker run -p 8000:8000 churn-api
```

---

## Contrato da API

### `POST /predict`

**Request body:**

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

**Response:**

```json
{
  "Cancelamento": "O cliente vai cancelar",
  "Probabilidade de acerto": "82.75%"
}
```

---

## Tecnologias Utilizadas

- **Python 3.12**
- **FastAPI** — framework para APIs REST
- **Pydantic** — validação de requests
- **scikit-learn** — pré-processamento e modelos baseline
- **XGBoost** — modelo final de classificação
- **joblib** — serialização do modelo
- **pandas** — manipulação de dados
- **Docker** — conteinerização

---

## Próximos Passos

- [ ] Adicionar `StandardScaler` para features numéricas
- [ ] Integrar MLflow para rastreamento de experimentos
- [ ] Adicionar detecção de drift com Evidently AI
- [ ] Adicionar pipeline de CI/CD com GitHub Actions