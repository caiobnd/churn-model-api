import joblib
from cleanning  import split_data,load_data,encoding,clean_data
from model      import train_logistic_regression,train_xgboost,train_random_forest
from pathlib    import Path

path_df     = Path('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_raw      = load_data(path_df)
df_clean    = clean_data(df_raw)
df_encoded  = encoding(df_clean)
X_train,X_test,y_train,y_test = split_data(df_encoded)

model_forest    = train_random_forest(X_train,y_train)
model_xgboost   = train_xgboost(X_train,y_train)
model_regression= train_logistic_regression(X_train,y_train)

path_forest     = Path('model/churn_random_fores.pkl')
path_xgboost    = Path('model/churn_xgboost.pkl')
path_regression = Path('model/churn_lr.pkl')

paths =[path_forest,path_xgboost,path_regression]
models=[model_forest,model_xgboost,model_regression]

for model, path in zip(models, paths):
    print(f'Salving {model}')
    joblib.dump(model,path)