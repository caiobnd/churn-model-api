from cleanning  import split_data,load_data,encoding,clean_data
from model      import train_logistic_regression,train_xgboost,train_random_forest
from pathlib    import Path

path_df     = Path('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_raw      = load_data(path_df)
df_clean    = clean_data(df_raw)
df_encoded  = encoding(df_clean)
X_train,X_test,y_train,y_test = split_data(df_encoded)