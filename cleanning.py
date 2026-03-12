import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    
    df = pd.read_csv(path)
    return df

def clean_data(df):
    
    df['TotalCharges'] = df['TotalCharges'].replace(' ', float('nan'))
    df.dropna(subset=['TotalCharges'],inplace=True)
    df.drop(columns=['customer_id'],inplace= True)
    
    return df

def encoding(df):
    
    binary_columns  = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling' , 'Churn']
    one_hot_columns = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']
    df.replace(['No internet service', 'No phone service'],'No', inplace=True)
    df_encode                   = pd.get_dummies(df,columns=one_hot_columns,drop_first=True)
    df_encode[binary_columns]   = df[binary_columns].apply(lambda col: col.map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}))
    
    return df_encode
    
def split_data(df):    
    X = df.drop(columns='Churn')
    y = df['Churn']
    return train_test_split(X,y,random_state=42,test_size=0.2)

