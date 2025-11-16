# Resultados da Análise - Security Benchmark Dataset

Este diretório contém os resultados principais da análise de vulnerabilidades em código gerado por LLMs.

## Arquivos de Análise Principal

### SHAP - Interpretabilidade do Modelo

- **`shap_summary_bar.png`**: Importância global das features (modelo principal `is_risky`)
  - Mostra as 15 features mais importantes ordenadas por impacto médio absoluto
  - Inclui features derivadas: `prompt_token_density`, `prompt_density`, `patch_density`, etc.

- **`shap_beeswarm.png`**: Direção e magnitude do impacto das features
  - Cada ponto representa uma amostra
  - Vermelho = valor alto da feature, Azul = valor baixo
  - Direita (positivo) = aumenta risco, Esquerda (negativo) = diminui risco

### SHAP - Análise de Correção vs Problemático

- **`shap_correcao_bar.png`**: Features que distinguem patches de correção de problemáticos
  - Modelo específico para classificar patches seguros que removem código vs patches que introduzem vulnerabilidades

- **`shap_correcao_beeswarm.png`**: Impacto direcionado na classificação correção/problemático

### Análise de CWEs (Common Weakness Enumeration)

- **`top10_cwes_por_modelo.png`**: Top 10 CWEs por modelo de linguagem
  - Gráfico de barras agrupadas mostrando frequência de cada CWE por LLM

- **`heatmap_cwes_modelo.png`**: Heatmap de CWEs x Modelos
  - Visualização matricial da distribuição de vulnerabilidades

### Análise de Patches

- **`correcao_vs_problema_modelo.png`**: Comparação de patches de correção vs problemáticos
  - Taxa de sucesso em correções por modelo
  - Patches que removem código (correção) vs patches que introduzem vulnerabilidades

### Outros

- **`coeficiente.png`**: Feature Importance do Random Forest (método tradicional)
- **`distribuicoes.png`**: Distribuições estatísticas das variáveis do dataset

## Metodologia

Todos os gráficos foram gerados usando:
- **Random Forest Classifier** (100 árvores, max_depth=15, class_weight='balanced')
- **SHAP (SHapley Additive exPlanations)** para interpretabilidade
- **Feature Engineering** com features derivadas (densidades, complexidades, interações)
- **Remoção de features CWE** para evitar data leakage

## Questões de Pesquisa Respondidas

- **QP1**: Random Forest captura relações não-lineares essenciais
- **QP2**: Distribuição de CWEs por modelo (top 10 global e por LLM)
- **QP3**: Relação entre tamanho de patch e risco
- **QP4**: Distinção entre patches de correção e problemáticos

## Para Paper Científico

Use **SHAP** (não apenas Feature Importance) por ser:
- ✅ Teoricamente fundamentado (valores de Shapley, teoria dos jogos)
- ✅ Consistente matematicamente
- ✅ Aceito pela comunidade científica (IEEE, USENIX, ACM)
- ✅ Resolve problema de correlação entre features

