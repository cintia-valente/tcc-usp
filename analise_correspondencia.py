import pandas as pd
import prince
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

df_questionario = pd.read_csv('respostas.csv')

# --- Limpeza dos Nomes das Colunas (reproduzindo o seu setup) ---
df_questionario.columns = df_questionario.columns.str.strip()
df_questionario.columns = df_questionario.columns.str.replace('\n', ' ')

# ---Renomeação das colunas---
df_questionario = df_questionario.rename(columns={
    'Gênero': 'Genero',
    'Faixa Etária': 'FaixaEtaria',
    'Escolaridade': 'Escolaridade',
    'A qual ou quais grupo(s) minoritário(s) você pertence?': 'GrupoMinoritario',
    'Tempo de experiência em equipes ágeis': 'ExperienciaAgil',
    'Nas equipes ágeis em que você atuou/atua, a diversidade foi/é valorizada?': 'Diversidade',
    'Nas equipes ágeis em que você atuou/atua, havia/há representatividade de grupos minoritários?': 'Representatividade',
    'Em alguma das equipes ágeis em que você atuou/atua, você já sofreu algum tipo de discriminação?': 'Discriminacao',
    'Nas equipes em que você atuou/atua, você sentiu/sente liberdade para expressar opiniões e dúvidas sem medo de julgamento?': 'SegurancaPsicologica',
    'Como você avalia seu bem-estar emocional nos ambientes ágeis em que atuou/atua?': 'BemEstarEmocional',
    'Você já se sentiu inseguro(a) em alguma equipe ágil devido a estereótipos ou discriminação?': 'Inseguranca',
    'Essas inseguranças afetaram/afetam sua confiança profissional ou desempenho na equipe?': 'ImpactoDesempenho',
    'Nas equipes ágeis em que você atuou/atua, a liderança promoveu/promove um ambiente seguro e acolhedor para os membros de grupos minoritários?': 'LiderancaSegura',
    'A liderança dessas equipes contribuiu/contribui para lidar com situações que impactam a segurança psicológica dos membros?': 'LiderancaSuporte',
    'A liderança dessas equipes promoveu/promove práticas inclusivas?': 'LiderancaInclusiva',
    'Havia/há representatividade de pessoas como você em cargos de liderança nessas equipes?': 'RepresentatividadeNaLideranca'
})

# --- Remoção de Colunas ---
df_questionario = df_questionario.drop(columns=['Carimbo de data/hora'])

if 'GrupoMinoritario' in df_questionario.columns:
    def standardize_multiselect(text):
        if pd.isna(text): # Lida com valores NaN (vazios)
            return text
        # Divide a string pelas vírgulas ou ponto e vírgulas
        # Limpa espaços em branco em cada item e os ordena alfabeticamente
        # Une os itens com ponto e vírgula para uma string padronizada
        return ';'.join(sorted([s.strip() for s in text.replace(',', ';').split(';') if s.strip()]))

    df_questionario['GrupoMinoritario'] = df_questionario['GrupoMinoritario'].apply(standardize_multiselect)

def analise_correspondencia(df, col1, col2, title_text):
    # Cria a tabela de contingência
    tabela = pd.crosstab(df[col1], df[col2])

    # Remove linhas ou colunas que contenham apenas zeros (não contribuem para a CA)
    tabela = tabela.loc[(tabela != 0).any(axis=1)]
    tabela = tabela.loc[:, (tabela != 0).any(axis=0)]

    if tabela.empty or min(tabela.shape) < 2:
        print(f"Aviso: A tabela de contingência para '{col1}' e '{col2}' é degenerada ou vazia após limpeza.")
        print("Não é possível calcular a Análise de Correspondência 2D para esta combinação.")
        print(f"Tabela shape: {tabela.shape}")
        return # Sai da função se a tabela for inadequada

    # Inicializa e ajusta o modelo CA
    # n_components pode ser reduzido se a tabela for muito pequena
    num_components = min(2, min(tabela.shape) - 1)
    if num_components < 1:
        print(f"Aviso: Tabela muito pequena para calcular qualquer componente CA para '{col1}' e '{col2}'.")
        print(f"Tabela shape: {tabela.shape}")
        return

    ca = prince.CA(n_components=num_components, n_iter=10, copy=True, check_input=True, engine='sklearn')
    ca = ca.fit(tabela)

    rows_coords = ca.row_coordinates(tabela)
    cols_coords = ca.column_coordinates(tabela)

    plt.figure(figsize=(14, 10))

    # Determina as coordenadas X e Y de forma robusta
    # Se houver apenas 1 componente, a coordenada Y será 0, efetivamente plotando em 1D
    x_rows = rows_coords.iloc[:, 0]
    y_rows = rows_coords.iloc[:, 1] if rows_coords.shape[1] > 1 else pd.Series(0, index=rows_coords.index)

    # Plot dos pontos azuis (primeira coluna)
    sns.scatterplot(x=x_rows, y=y_rows, s=100, label=col1, color='blue', zorder=2)

    texts_for_adjust = []
    legend_map = {}
    point_labels = []

    # Cria rótulos numerados para as categorias da primeira coluna
    for i, txt in enumerate(tabela.index):
        num_label = str(i + 1)
        legend_map[num_label] = txt
        point_labels.append(num_label)

    # Adiciona os rótulos numerados para ajuste
    for i, label_text in enumerate(point_labels):
        texts_for_adjust.append(plt.annotate(label_text, (x_rows.iloc[i], y_rows.iloc[i]),
                                             fontsize=9, color='blue', ha='left', va='bottom', weight='bold', zorder=3))

    # Determina as coordenadas X e Y para as colunas
    x_cols = cols_coords.iloc[:, 0]
    y_cols = cols_coords.iloc[:, 1] if cols_coords.shape[1] > 1 else pd.Series(0, index=cols_coords.index)

    # Plot dos pontos vermelhos (segunda coluna)
    sns.scatterplot(x=x_cols, y=y_cols, s=100, marker='s', label=col2, color='red', zorder=2)

    # Adiciona os rótulos das categorias da segunda coluna para ajuste
    for i, txt in enumerate(tabela.columns):
        texts_for_adjust.append(plt.annotate(txt, (x_cols.iloc[i] + 0.03, y_cols.iloc[i] - 0.03),
                                             fontsize=10, color='red', ha='left', va='top', zorder=3))

    # Ajusta a posição dos textos para evitar sobreposição
    adjust_text(texts_for_adjust,
                expand_points=(8, 10),
                arrowprops=dict(arrowstyle='-', color='lightgrey', lw=1, alpha=1),
                force_points=2.5,
                force_text=2.5,
                add_points_as_circles=False,
                limiter=2000
               )

    plt.axhline(0, color='grey', lw=0.7)
    plt.axvline(0, color='grey', lw=0.7)
    plt.title(f'Análise de Correspondência: {title_text}')
    plt.subplots_adjust(right=0.5)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 1])

    # Prepara o texto da legenda para as categorias da primeira coluna
    legend_text = f"\nLegenda das Categorias de $\\bf{{{col1}}}$:\n\n"
    for num, name in legend_map.items():
        legend_text += f"$\\bf{{{num}}}$: {name}\n"

    legend_fig = plt.figure("Legenda das Categorias", figsize=(5, 6))
    legend_fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    plt.figtext(0.05, 0.95, legend_text.strip(), ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", ec="black", lw=0.5, alpha=0.8))
    
    plt.show()

# --- Chamadas da função para gerar os gráficos ---
print("Gerando gráfico para 'GrupoMinoritario' x 'SegurancaPsicologica'")
analise_correspondencia(df_questionario, 'GrupoMinoritario', 'SegurancaPsicologica', 'Grupo Minoritário x Segurança psicológica')

print("\nGerando gráfico para 'GrupoMinoritario' x 'ImpactoDesempenho'")
analise_correspondencia(df_questionario, 'GrupoMinoritario', 'ImpactoDesempenho', 'Grupo Minoritário x Impacto no desempenho')

print("Gerando gráfico para 'GrupoMinoritario' x 'LiderancaInclusiva'")
analise_correspondencia(df_questionario, 'GrupoMinoritario', 'LiderancaInclusiva', 'Grupo Minoritário x Liderança inclusiva')

print("\nGerando gráfico para 'Discriminacao' x 'SegurancaPsicologica'")
analise_correspondencia(df_questionario, 'Discriminacao', 'SegurancaPsicologica', 'Discriminação x Segurança psicológica')

print("\nGerando gráfico para 'Discriminacao' x 'ImpactoDesempenho'")
analise_correspondencia(df_questionario, 'Discriminacao', 'ImpactoDesempenho', 'Discriminação x Impacto no desempenho')

print("\nGerando gráfico para 'Genero' x 'BemEstarEmocional'")
analise_correspondencia(df_questionario, 'Genero', 'BemEstarEmocional', 'Gênero x Bem-estar emocional')

print("\nGerando gráfico para 'Genero' x 'SegurancaPsicologica'")
analise_correspondencia(df_questionario, 'Genero', 'SegurancaPsicologica', 'Gênero x Segurança psicológica')

print("\nGerando gráfico para 'FaixaEtaria' x 'SegurancaPsicologica'")
analise_correspondencia(df_questionario, 'FaixaEtaria', 'SegurancaPsicologica', 'Faixa etária x Seguranca psicológica')

print("\nGerando gráfico para 'FaixaEtaria' x 'BemEstarEmocional'")
analise_correspondencia(df_questionario, 'FaixaEtaria', 'BemEstarEmocional', 'Faixa etária x Bem-estar emocional')

print("\nGerando gráfico para 'Escolaridade' x 'SegurancaPsicologica'")
analise_correspondencia(df_questionario, 'Escolaridade', 'BemEstarEmocional', 'Escolaridade x Seguranca psicológica')

print("\nGerando gráfico para 'Escolaridade' x 'BemEstarEmocional'")
analise_correspondencia(df_questionario, 'Escolaridade', 'BemEstarEmocional', 'Escolaridade x Bem-estar emocional')

print("\nGerando gráfico para 'SegurancaPsicologica' x 'BemEstarEmocional'")
analise_correspondencia(df_questionario, 'SegurancaPsicologica', 'BemEstarEmocional', 'Segurança psicológica x Bem-estar emocional')

print("\nGerando gráfico para 'ExperienciaAgil' x 'SegurancaPsicologica'")
analise_correspondencia(df_questionario, 'ExperienciaAgil', 'SegurancaPsicologica', 'Experiência ágil x Segurança psicológica')

print("\nGerando gráfico para 'ExperienciaAgil' x 'BemEstarEmocional'")
analise_correspondencia(df_questionario, 'ExperienciaAgil', 'BemEstarEmocional', 'Experiência ágil x Bem-estar emocional')

print("\nGerando gráfico para 'Diversidade' x 'SegurancaPsicologica'")
analise_correspondencia(df_questionario, 'Diversidade', 'SegurancaPsicologica', 'Diversidade x Segurança psicológica')

print("\nGerando gráfico para 'Representatividade' x 'SegurancaPsicologica'")
analise_correspondencia(df_questionario, 'Representatividade', 'SegurancaPsicologica', 'Representatividade x Segurança psicológica')

print("\nGerando gráfico para 'RepresentatividadeNaLideranca' x 'SegurancaPsicologica'")
analise_correspondencia(df_questionario, 'RepresentatividadeNaLideranca', 'SegurancaPsicologica', 'Representatividade na liderança x Segurança psicológica')

print("\nGerando gráfico para 'LiderancaInclusiva' x 'SegurancaPsicologica'")
analise_correspondencia(df_questionario, 'LiderancaInclusiva', 'SegurancaPsicologica', 'Liderança inclusiva x Segurança psicológica')

print("\nGerando gráfico para 'LiderancaInclusiva' x 'BemEstarEmocional'")
analise_correspondencia(df_questionario, 'LiderancaInclusiva', 'BemEstarEmocional', 'Liderança inclusiva x Bem-estar emocional')

print("\nGerando gráfico para 'Inseguranca' x 'SegurancaPsicologica'")
analise_correspondencia(df_questionario, 'Inseguranca', 'SegurancaPsicologica', 'Insegurança x Segurança psicológica')

print("\nGerando gráfico para 'Inseguranca' x 'ImpactoDesempenho'")
analise_correspondencia(df_questionario, 'Inseguranca', 'ImpactoDesempenho', 'Insegurança x Impacto no desempenho')

print("\nGerando gráfico para 'Representatividade' x 'ImpactoDesempenho'")
analise_correspondencia(df_questionario, 'Representatividade', 'ImpactoDesempenho', 'Representatividade x Impacto no desempenho')