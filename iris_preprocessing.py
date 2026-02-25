import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Modul untuk Kriteria 2
def preprocess_data(df):
    X = df.drop('species', axis=1)
    y = df['species']
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    df_processed = X_scaled.copy()
    df_processed['species'] = y
    return df_processed
