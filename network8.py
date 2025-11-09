import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import combinations
from networkx.algorithms.community import louvain_communities
from scipy.stats import lognorm

df = pd.read_csv('all_findings_flat.csv')
df_gpt = df[(df['model'] == 'gpt-4o') & (df['patch_lines'] > 0)].copy()

grouped = df_gpt.groupby('filename')['cwe'].apply(list).reset_index()

DG = nx.DiGraph()

all_cwes = set()
for cwes in grouped['cwe']:
    clean_cwes = [str(c).strip() for c in cwes if pd.notnull(c) and str(c) != 'nan' and c != '']
    all_cwes.update(clean_cwes)

for cwe in all_cwes:
    DG.add_node(cwe)


# Constrói grafo dirigido: arestas por coocorrência no arquivo
DG = nx.DiGraph()
for idx, row in grouped.iterrows():
    cwes = [str(c).strip() for c in row['cwe'] if pd.notnull(c) and str(c) != 'nan' and c != '']
    cwes = list(set(cwes))
    if len(cwes) > 1:
        for a, b in combinations(cwes, 2):
            for x, y in [(a, b), (b, a)]:
                if DG.has_edge(x, y):
                    DG[x][y]['weight'] += 1
                else:
                    DG.add_edge(x, y, weight=1)

# Contagens iniciais
N0 = DG.number_of_nodes()
E0 = DG.number_of_edges()
print(f"Antes do filtro: N0={N0} nós, E0={E0} arestas")
# Filtro dinâmico: pesos nas arestas
# edge_weights = {}
# for u, v in DG.edges():
#     edge_weights[(u, v)] = edge_weights.get((u, v), 0) + 1
# for (u, v), w in edge_weights.items():
#     DG[u][v]['weight'] = w
weights = [d['weight'] for _, _, d in DG.edges(data=True)]

# OPÇÃO: Ajuste o percentil para controlar quantas arestas manter
# 0 = todas as arestas, 70 = apenas 30% mais fortes, 90 = apenas 10% mais fortes
percentil_filtro = 0  # ← MUDE AQUI: 0 para ver todas as arestas

if percentil_filtro > 0:
    peso_min = np.percentile(weights, percentil_filtro)
    DG.remove_edges_from([(u, v) for u, v, d in DG.edges(data=True) if d.get('weight', 0) < peso_min])
    print(f"✂️  Filtro aplicado: percentil {percentil_filtro} (peso_min={peso_min:.2f})")
else:
    peso_min = 0
    print(f"✅ SEM filtro: mantendo TODAS as arestas")

# Contagens pós-filtro
N1_total = DG.number_of_nodes()
E1 = DG.number_of_edges()
active_nodes = [n for n, d in DG.degree() if d > 0]
N1 = len(active_nodes)
print(f"Após filtro pct70 (peso_min={peso_min:.2f}): N1={N1}/{N1_total} nós ativos/total, E1={E1} arestas")

# MÉTRICAS principais
print(f"Nós (CWEs no código LLM): {list(DG.nodes())}")
print(f"Arestas (relações dirigidas proximidade/impacto): {list(DG.edges(data=True))}")

out_degrees = dict(DG.out_degree())
in_degrees = dict(DG.in_degree())
main_causes = sorted(out_degrees.items(), key=lambda x: -x[1])
main_sinks = sorted(in_degrees.items(), key=lambda x: -x[1])
n_nodes = DG.number_of_nodes()
p_impact = {node: len(nx.descendants(DG, node))/n_nodes for node in DG.nodes}
top_pimpact = sorted(p_impact.items(), key=lambda x: -x[1])
communities = louvain_communities(DG.to_undirected(), seed=42)

# Changeability/Reusability Index
changeability_n = 3
cw_changeability = sum(1 for v in out_degrees.values() if v >= changeability_n)
print(f"Changeability Index (CWEs grau saída >= {changeability_n}): {cw_changeability}")

reusability_n = 3
cw_reusability = sum(1 for v in in_degrees.values() if v >= reusability_n)
print(f"Reusability Index (CWEs grau entrada >= {reusability_n}): {cw_reusability}")

pimpact_thr = 0.2
cw_pimpact = sum(1 for v in p_impact.values() if v >= pimpact_thr)
print(f"P-Impact Index (CWEs com impacto >= {pimpact_thr*100}% do grafo): {cw_pimpact}")
for node, v in top_pimpact[:5]:
    print(f"{node}: {v*100:.1f}% dos nós atingidos")

# Profundidade média (longest path)
def media_profundidade(DG):
    depth_list = []
    for node in DG.nodes:
        lengths = nx.single_source_shortest_path_length(DG, node)
        if lengths:
            depth_list.append(max(lengths.values()))
    return np.mean(depth_list)
print(f"Profundidade média de alcance/transitividade: {media_profundidade(DG):.2f}")

print(f"Clusters estruturais Louvain: {len(communities)}")
for idx, cluster in enumerate(communities):
    print(f'Cluster {idx+1} (tam={len(cluster)}):', sorted(cluster))

# == Diagnóstico de ruído na rede ==
# Nós isolados/grau=1 (potencial ruído)
isolados = [n for n, d in DG.degree() if d == 1]
if isolados:
    print(f"Nós com grau 1 (potencial ruído): {isolados}")
else:
    print("Nenhum nó isolado identificado.")

# Distribuição dos pesos das arestas (para ruído)
pesos = [d['weight'] for _, _, d in DG.edges(data=True)]
plt.figure(figsize=(7,4))
plt.hist(pesos, bins=np.arange(min(pesos), max(pesos)+2)-0.5, color='gray', edgecolor='black')
plt.xlabel('Peso da aresta')
plt.ylabel('Frequência')
plt.title('Distribuição dos pesos das arestas')
plt.tight_layout()
plt.savefig('hist_pesos_arestas.png')
plt.close()

# Clustering coefficient
clustering = nx.average_clustering(DG.to_undirected())
print(f"Coeficiente de agrupamento médio (clustering): {clustering:.2f}")

# Componentes desconectados
components = nx.number_connected_components(DG.to_undirected())
print(f"Número de componentes conectados na rede: {components}")

# Betweenness centrality baixo (outliers)
betweenness = nx.betweenness_centrality(DG)
menor_betweenness = sorted(betweenness.items(), key=lambda x: x[1])[:5]
print("Nós com menor betweenness centrality (potencial ruído):", menor_betweenness)

# == Visualizações ==
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(DG, seed=42, k=0.6, iterations=100)

# Define tamanho dos nós proporcional a out-degree, com escala ajustável
out_degrees = dict(DG.out_degree())
max_out = max(out_degrees.values()) if out_degrees else 1
node_sizes = [300 + 700 * out_degrees.get(n, 0) / max_out for n in DG.nodes()]

nx.draw_networkx_nodes(DG, pos, node_color='skyblue', node_size=node_sizes, edgecolors='black', linewidths=1.2, alpha=0.9)
nx.draw_networkx_edges(DG, pos, arrowsize=20, arrowstyle='-|>', edge_color='gray', width=1.2, alpha=0.7)
nx.draw_networkx_labels(DG, pos, font_size=11, font_weight='bold')

plt.title('Rede de dependências CWE (por arquivo) - melhorada', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig('grafo_dependencias_cwe_filename_pct70_melhorada.png', dpi=300)
plt.close()


def plot_loglog(degree_dict, titulo, arq, cor='b', limiar=1):
    values = np.array(list(degree_dict.values()))
    values = values[values >= limiar]
    unique_val, counts = np.unique(values, return_counts=True)
    pk = counts / counts.sum()
    plt.figure(figsize=(7,6))
    plt.loglog(unique_val, pk, marker='o', linestyle='', color=cor)
    plt.xlabel('k')
    plt.ylabel('p_k (prob)')
    plt.title(titulo)
    plt.grid(True, which="both", ls="--")
    if len(unique_val) > 0:
        k_line = np.linspace(unique_val.min(), unique_val.max(), 100)
        plt.plot(k_line, k_line**(-2.5)/k_line.min()**(-2.5)*pk[0], 'r--', label=r'P(k) $\sim k^{-2.5}$')
        plt.legend()
    plt.tight_layout()
    plt.savefig(arq)
    plt.close()
plot_loglog(out_degrees, 'Log-log Out-degree (Influência/Changeability)', 'loglog_outdegree_filename_pct70.png')
plot_loglog(in_degrees, 'Log-log In-degree (Dependência/Reusability)', 'loglog_indegree_filename_pct70.png', cor='orange')


def plot_lognormal_hist(degree_dict, titulo, nome_arq, cor='blue'):
    vals = np.array(list(degree_dict.values()))
    plt.figure(figsize=(7,4))
    sns.histplot(vals, bins=8, stat='density', color=cor, edgecolor='black', alpha=0.6)
    shape, loc, scale = lognorm.fit(vals, floc=0)
    x_vals = np.linspace(vals.min(), vals.max(), 100)
    plt.plot(x_vals, lognorm.pdf(x_vals, shape, loc, scale), 'r-', lw=2, label='Ajuste Log-normal')
    plt.xlabel('Grau')
    plt.ylabel('Densidade')
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.savefig(nome_arq)
    plt.close()

plot_lognormal_hist(out_degrees, 'Histograma Log-normal Out-degree (Influência)', 'hist_lognormal_outdegree.png', cor='blue')
plot_lognormal_hist(in_degrees, 'Histograma Log-normal In-degree (Dependência)', 'hist_lognormal_indegree.png', cor='orange')


print("\n===== Respostas Pesquisas =====")
print("RQ1: CWEs exercendo maior influência (causas raiz):", main_causes[:5])
print("RQ2: CWEs como dependências críticas (fragilidade):", main_sinks[:5])
print("RQ3: CWEs com maior impacto sistêmico (propagação):", top_pimpact[:5])
print("RQ4: Clusters estruturais detectados:", len(communities), " > maiores exemplos:", [sorted(list(c)) for c in communities[:3]])

# Visualização tradicional (grafo global com força de conexão)
plt.figure(figsize=(16, 12))

# Identificar o nó com maior out-degree (hub central)
hub_node = max(out_degrees.items(), key=lambda x: x[1])[0]

# CWEs que queremos dar destaque no posicionamento
highlight_cwes = ['CWE-78', 'CWE-703', 'CWE-502', 'CWE-22', 'CWE-327']

# Layout inicial com kamada_kawai (melhor distribuição espacial)
pos = nx.kamada_kawai_layout(DG)

# Posicionar estrategicamente os CWEs de interesse
present_highlights = [c for c in highlight_cwes if c in DG.nodes()]
if present_highlights:
    # Hub no centro
    pos[hub_node] = (0, 0)
    
    # Distribuir os 5 CWEs em círculo ao redor do hub
    angle_step = 2 * np.pi / len(present_highlights)
    radius = 0.65  # Distância do centro
    
    for i, cwe in enumerate(present_highlights):
        if cwe != hub_node:  # Se o hub for um dos 5, já está no centro
            angle = i * angle_step
            pos[cwe] = (radius * np.cos(angle), radius * np.sin(angle))
    
    # Afastar outros nós se estiverem muito próximos dos destacados
    for node in DG.nodes():
        if node not in present_highlights and node != hub_node:
            x, y = pos[node]
            distance = np.sqrt(x**2 + y**2)
            if distance < 0.35:  # Se muito perto do centro, afastar
                scale = 0.8 / (distance + 0.1)
                pos[node] = (x * scale, y * scale)

# Tamanho dos nós proporcional ao out-degree (influência)
max_out = max(out_degrees.values()) if out_degrees.values() else 1
min_size = 500
max_size = 5000
node_sizes = [min_size + (max_size - min_size) * (out_degrees.get(n, 0) / max_out) for n in DG.nodes()]

# Desenhar nós
nx.draw_networkx_nodes(
    DG, pos,
    node_size=node_sizes,
    node_color='skyblue',
    edgecolors='darkblue',
    linewidths=2,
    alpha=0.9
)

# Desenhar todas as arestas de forma uniforme
nx.draw_networkx_edges(
    DG, pos,
    edge_color='gray',
    width=1.0,
    alpha=0.4,
    arrows=False
)

# Labels com melhor contraste
nx.draw_networkx_labels(
    DG, pos,
    font_size=13,
    font_weight='bold',
    font_color='black'
)

plt.title('Rede de Dependências de Vulnerabilidades (CWEs), gpt-4o', 
          fontsize=18, pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig('visualizacao_rede_cwe_gpt4o2.jpg', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Visualização principal salva: visualizacao_rede_cwe_gpt4o.jpg")

# ============================================================================
# MATRIZ DE ADJACÊNCIA - Dependências Direcionadas entre CWEs
# ============================================================================
print("\n📊 Criando matriz de adjacência...")

# Criar matriz de adjacência
nodes_sorted = sorted(DG.nodes())
n_nodes = len(nodes_sorted)
adj_matrix = np.zeros((n_nodes, n_nodes))

# Preencher matriz com os pesos das arestas
for i, source in enumerate(nodes_sorted):
    for j, target in enumerate(nodes_sorted):
        if DG.has_edge(source, target):
            adj_matrix[i, j] = DG[source][target]['weight']

# Criar heatmap
plt.figure(figsize=(14, 12))

# Usar escala de cores amarelo para vermelho escuro
sns.heatmap(adj_matrix, 
            xticklabels=nodes_sorted, 
            yticklabels=nodes_sorted,
            cmap='YlOrRd',  # Amarelo -> Laranja -> Vermelho
            cbar_kws={'label': 'Peso da aresta'},
            linewidths=0.5,
            linecolor='white',
            square=True,
            fmt='g')

plt.title('Matriz de Adjacência - Dependências Direcionadas entre CWEs', 
          fontsize=16, pad=15, fontweight='bold')
plt.xlabel('CWE Destino', fontsize=12, fontweight='bold')
plt.ylabel('CWE Origem', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('matriz_adjacencia_cwe.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Matriz de adjacência salva: matriz_adjacencia_cwe.png")

# Visualizações dos maiores clusters Louvain
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#f0c2c2','#b3b3b3','#f7b3a7']
for i, comm in enumerate(sorted(communities, key=lambda x: -len(x))[:3]):  # Top 3 clusters
    subG = DG.subgraph(comm)
    pos = nx.spring_layout(subG, seed=42)
    plt.figure(figsize=(8,7))
    nx.draw_networkx_nodes(subG, pos, node_color=colors[i % len(colors)], alpha=0.8)
    nx.draw_networkx_edges(subG, pos, arrowstyle='-|>', arrowsize=12, alpha=0.58)
    nx.draw_networkx_labels(subG, pos, font_size=10)
    plt.title(f'Comunidade Louvain #{i+1} (tamanho={len(comm)})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'cluster_louvain_{i+1}.jpg')
    plt.close()

# Diagnóstico Power-law (opcional)
try:
    import powerlaw
    def compare_models(data, label):
        data = np.array(list(data.values()))
        data = data[data > 0]
        if data.size < 5 or np.unique(data).size < 2:
            print(f"\n[Powerlaw] Amostra insuficiente para {label}.")
            return
        fit = powerlaw.Fit(data, discrete=True, verbose=False)
        print(f"\nAjuste power-law ({label}):")
        print(f"  alpha: {fit.alpha:.2f}, xmin: {fit.xmin}")
        for alt in ['exponential', 'lognormal', 'truncated_power_law']:
            R, p = fit.distribution_compare('power_law', alt, normalized_ratio=True)
            print(f"  power_law vs {alt:>20s}: R={R:+.3f}, p={p:.3f}")
        scores = {}
        for alt in ['exponential', 'lognormal', 'truncated_power_law']:
            try:
                ll = getattr(fit, alt).loglikelihood
                scores[alt] = ll
            except Exception:
                pass
        if scores:
            best_alt = max(scores, key=scores.get)
            print(f"  Melhor alternativa (sem power-law) por log-likelihood: {best_alt}")
        n = data.size
        def aic_bic(model_name, k):
            try:
                ll = getattr(fit, model_name).loglikelihood
                aic = 2*k - 2*ll
                bic = np.log(n)*k - 2*ll
                return aic, bic
            except Exception:
                return np.nan, np.nan
        aic_bic_tbl = {
            'power_law': aic_bic('power_law', 2),
            'exponential': aic_bic('exponential', 3),
            'lognormal': aic_bic('lognormal', 3),
            'truncated_power_law': aic_bic('truncated_power_law', 3),
        }
        best_aic = min(aic_bic_tbl.items(), key=lambda x: x[1][0] if not np.isnan(x[1][0]) else np.inf)[0]
        best_bic = min(aic_bic_tbl.items(), key=lambda x: x[1][1] if not np.isnan(x[1][1]) else np.inf)[0]
        print("  AIC/BIC (menor melhor):")
        for m, (aic, bic) in aic_bic_tbl.items():
            print(f"    {m:>20s}: AIC={aic:.1f}, BIC={bic:.1f}")
        print(f"  → Melhor por AIC: {best_aic}; Melhor por BIC: {best_bic}")
    compare_models(out_degrees, 'Out-degree')
    compare_models(in_degrees, 'In-degree')
except ImportError:
    print("Pacote powerlaw não instalado. Rode: pip install powerlaw")

# Diagnóstico de ruído global na rede
print("\n===== DIAGNÓSTICO DE RUÍDO/SOLIDEZ =====")
if components > 1:
    print("Rede fragmentada: existência de mais de um componente conectado (potencial ruído/fracionamento).")
else:
    print("Rede totalmente conectada.")

if len(isolados) > 0:
    print(f"Nós com grau 1 detectados (potencial ruído): {isolados}")
else:
    print("Nenhum nó isolado, estrutura coesa.")

if clustering < 0.2:
    print(f"Clustering baixo ({clustering:.2f}), rede pode estar pouco estruturada (potencial ruído).")
else:
    print(f"Clustering médio/alto ({clustering:.2f}), rede apresenta agrupamento e coesão.")

print("Ajuste do grau (lognormal) é típico de dependências reais em redes de vulnerabilidades, reforçando pouca presença de ruído.")
print("Betweenness centrality muito baixo pode aparecer em nós periféricos, mas não há isolamento extremo ou agrupamentos artificiais.")
