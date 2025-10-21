import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction import FeatureHasher

# Carregar dataset
df = pd.read_csv('all_findings_flat.csv')

# 1. Remover outliers em colunas numéricas consideradas críticas
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    return df[(df[col] >= lim_inf) & (df[col] <= lim_sup)]

cols_outliers = [
    "patch_lines", "patch_added", "patch_removed", "patch_files_touched", "patch_hunks",
    "patch_churn", "patch_net", "prompt_chars", "prompt_lines", "prompt_tokens"
]

for col in cols_outliers:
    df = remove_outliers(df, col)

# 2. Codificação de variáveis categóricas

# Categóricas ordenadas para Label Encoder
label_cols = ['severity', 'confidence']

le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Categóricas textuais de alta cardinalidade para hashing (evitar explosão dimensional)
high_card_cols = [
    'backup_dir', 'repo', 'case', 'report_file', 'filename', 'test_id', 'test_name', 'details'
]
hasher = FeatureHasher(input_type='string', n_features=32)  

# Aplicar hashing and substituir colunas originais por features hash (32 dimensões por coluna)

hashed_features = []
for col in high_card_cols:
    # FeatureHasher precisa de um iterável de iteráveis de strings
    hashed = hasher.transform([[str(val)] for val in df[col]]).toarray()
    hashed_df = pd.DataFrame(hashed,
                             columns=[f'{col}_hash_{i}' for i in range(hashed.shape[1])])
    hashed_features.append(hashed_df)
df_hashed = pd.concat(hashed_features, axis=1)

# Remover colunas textuais originais
df = df.drop(columns=high_card_cols)

# Concatenar features hash com df
df = pd.concat([df, df_hashed], axis=1)

# Categóricas nominais com cardinalidade baixa para One-Hot Encoding
low_card_cols = ['model', 'cwe']

df = pd.get_dummies(df, columns=low_card_cols)

# 3. Normalização das variáveis numéricas
num_cols = [
    "patch_lines", "patch_added", "patch_removed", "patch_files_touched", "patch_hunks",
    "patch_churn", "patch_net", "prompt_chars", "prompt_lines", "prompt_tokens", "temperature",
    "cwe_prevalence_overall", "cwe_severity_score", "cwe_weighted_severity"
]
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 4. Conferência final do dataset
print("Dataset pré-processado com shape:", df.shape)
print("Amostra:\n", df.head())
print("Estatísticas descritivas:\n", df[num_cols].describe())
