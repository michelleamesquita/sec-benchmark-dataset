import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu

# Carregar dataset
df = pd.read_csv('all_findings_flat.csv')

# Definir colunas numéricas e categóricas
col_num = ["patch_lines", "patch_added", "patch_removed", "patch_files_touched", "patch_hunks",
           "patch_churn", "patch_net", "prompt_chars", "prompt_lines", "prompt_tokens",
           "prompt_has_security_guidelines", "temperature", "is_risky", "cwe_prevalence_overall",
           "cwe_severity_score", "cwe_weighted_severity"]
col_cat = df.select_dtypes(include=['object']).columns.tolist()

# 1) Estatísticas descritivas
print("Estatísticas descritivas dos atributos numéricos:\n")
print(df[col_num].describe())

# 2) Distribuição dos atributos (plot de histogramas)
plt.figure(figsize=(15, 20))
for i, col in enumerate(col_num):
    plt.subplot(6, 3, i+1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribuição de {col}')
plt.tight_layout()
plt.show()

# 3) Dados faltantes
print("\nQuantidade de dados faltantes por coluna:")
print(df.isnull().sum())

# 4) Testes de associação entre variáveis categóricas e 'is_risky' (Qui-quadrado)
print("\nTeste Qui-Quadrado para variáveis categóricas:")
for col in col_cat:
    if col != "is_risky":
        ct = pd.crosstab(df[col], df['is_risky'])
        chi2, p, dof, ex = chi2_contingency(ct)
        print(f'{col}: p-valor={p:.4f}')

# 5) Testes entre variáveis numéricas e 'is_risky' (Mann-Whitney)
print("\nTeste Mann-Whitney para variáveis numéricas:")
for col in col_num:
    if col != "is_risky":
        grupo_risky = df[df['is_risky']==1][col]
        grupo_safe = df[df['is_risky']==0][col]
        stat, p = mannwhitneyu(grupo_risky, grupo_safe)
        print(f'{col}: p-valor={p:.4f}')

# 6) Correlação entre variáveis numéricas
plt.figure(figsize=(12, 10))
corr = df[col_num].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação entre atributos numéricos')
plt.show()

# Identificar pares fortemente correlacionados (>0.9 ou <-0.9)
print("\nPares altamente correlacionados:")
for i in range(len(col_num)):
    for j in range(i+1, len(col_num)):
        if abs(corr.iloc[i,j]) > 0.9:
            print(f'{col_num[i]} e {col_num[j]}: correlação = {corr.iloc[i,j]:.2f}')

# 7) Detecção de outliers usando IQR para variáveis numéricas
print("\nQuantidade de outliers detectados por atributo:")
for col in col_num:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lim_inf) | (df[col] > lim_sup)][col].count()
    print(f'{col}: {outliers}')
