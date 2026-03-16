import joblib
from cleanning  import split_data,load_data,encoding,clean_data
from model      import train_logistic_regression,train_xgboost,train_random_forest
from sklearn.metrics import classification_report
from pathlib    import Path

path_df     = Path('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_raw      = load_data(path_df)
df_clean    = clean_data(df_raw)
df_encoded  = encoding(df_clean)
X_train,X_test,y_train,y_test = split_data(df_encoded)

model_forest    = train_random_forest(X_train,y_train)
model_xgboost   = train_xgboost(X_train,y_train)
model_regression= train_logistic_regression(X_train,y_train)

models=[model_forest,model_xgboost,model_regression]  
    
for model, name in zip(models, ['Random Forest', 'XGBoost', 'Logistic Regression']):
    y_pred = model.predict(X_test)
    print(f'\n{name}')
    print(classification_report(y_test, y_pred))

expected_columns =list(df_encoded.columns)
expected_columns.remove('Churn')
path_xgboost = Path('model/churn_xgboost.pkl')
path_columns = Path('model/expected_columns.pkl')
joblib.dump(model_xgboost,path_xgboost)
joblib.dump(expected_columns,path_columns)