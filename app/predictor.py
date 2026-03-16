from constants  import binary_columns,one_hot_columns
from app.schema import CustomerData
from pathlib    import Path
import joblib
import pandas as pd

path_model      = Path('model/churn_xgboost.pkl')
path_columns    = Path('model/expected_columns.pkl')

def predict(customer: CustomerData):
    
    dict = customer.model_dump()
    
    df = pd.DataFrame([dict])
    df['TotalCharges']  = df['TotalCharges'].astype(float)
    df.replace(['No internet service', 'No phone service'],'No', inplace=True)
    encoded = df.copy()
    encoded[binary_columns]    = df[binary_columns].apply(lambda col: col.map({'Yes':1, 'No': 0, 'Male': 1, 'Female': 0}))
    df_encoded          = pd.get_dummies(encoded,columns=one_hot_columns,drop_first=True)
    df_encoded          = df_encoded.apply(lambda col: col.astype(int) if col.dtype == bool else col)
    model = joblib.load(path_model)
    expected_columns    = joblib.load(path_columns)
    df_encoded          = df_encoded.reindex(columns=expected_columns,fill_value=0)
    
    churn               = (model.predict(df_encoded))
    churn_proba         = f"{float(model.predict_proba(df_encoded)[0][1]) * 100:.2f}"

    cancelamento        = 'O cliente vai cancelar' if churn==1 else 'Cliente fiel'
        
    return {'Cancelamento': cancelamento ,'Probabilidade de acerto':f'{churn_proba}%'}
    