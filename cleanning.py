import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    
    df = pd.read_csv(path)
    return df

def clean_data(df):
    
    df['TotalCharges'] = df['TotalCharges'].replace(' ', float('nan'))
    df.dropna(subset=['TotalCharges'],inplace=True)
    df.drop(columns=['customerID'],inplace= True)
    
    return df

def encoding(df):
    
    binary_columns  = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling' , 'Churn', 
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    one_hot_columns = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']
    df['TotalCharges']          = df['TotalCharges'].astype(float)
    df.replace(['No internet service', 'No phone service'],'No', inplace=True)
    df_encoded                  = df.copy()
    df_encoded[binary_columns]  = df_encoded[binary_columns].apply(lambda col: col.map({'Yes':1, 'No': 0, 'Male': 1, 'Female': 0}))
    df_encoded                  = pd.get_dummies(df_encoded,columns=one_hot_columns,drop_first=True)
    df_encoded = df_encoded.apply(lambda col: col.astype(int) if col.dtype == bool else col)
    
    return df_encoded
    
def split_data(df):    
    X = df.drop(columns='Churn')
    y = df['Churn']
    return train_test_split(X,y,random_state=42,test_size=0.2)

