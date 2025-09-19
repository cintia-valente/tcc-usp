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

def analise_correspondencia(df, col1, col2, title_text, label1, label2):
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

    # Determina as coordenadas X e Y
    # Se houver apenas 1 componente, a coordenada Y será 0, efetivamente plotando em 1D
    x_rows = rows_coords.iloc[:, 0]
    y_rows = rows_coords.iloc[:, 1] if rows_coords.shape[1] > 1 else pd.Series(0, index=rows_coords.index)

    # Plot dos pontos azuis (primeira coluna)
    sns.scatterplot(x=x_rows, y=y_rows, s=60, label=label1, color='blue')

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
        texts_for_adjust.append(plt.annotate(
            label_text, 
            (x_rows.iloc[i], y_rows.iloc[i]),
            fontsize=14, 
            color='blue', 
            ha='left', 
            va='bottom', 
            weight='bold')
        )

    # Determina as coordenadas X e Y para as colunas
    x_cols = cols_coords.iloc[:, 0]
    y_cols = cols_coords.iloc[:, 1] if cols_coords.shape[1] > 1 else pd.Series(0, index=cols_coords.index)

    # Plot dos pontos vermelhos (segunda coluna)
    sns.scatterplot(x=x_cols, y=y_cols, s=60, marker='s', label=label2, color='red')

    # Adiciona os rótulos das categorias da segunda coluna para ajuste
    for i, txt in enumerate(tabela.columns):
        texts_for_adjust.append(plt.annotate(
            txt, 
            (x_cols.iloc[i], y_cols.iloc[i]),
            fontsize=12, 
            color='red', 
            ha='left', 
            va='top', 
            weight='bold')
        )

    # Ajusta a posição dos textos para evitar sobreposição
    adjust_text(texts_for_adjust,
                expand_points=(2, 2),
                force_points=6,
                force_text=6,
                add_points_as_circles=False,
                limiter=4000
               )

    plt.axhline(0, color='grey', lw=0.7)
    plt.axvline(0, color='grey', lw=0.7)
    plt.title(f'Análise de Correspondência: {title_text}')
    plt.subplots_adjust(right=0.5)
    plt.grid(True)
    
    plt.legend(fontsize=15)
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
print("Gerando gráfico para 'Grupo Minoritário' x 'Segurança Psicológica'")
analise_correspondencia(df_questionario, 'GrupoMinoritario', 'SegurancaPsicologica', 'Grupo Minoritário x Segurança Psicológica', 'Grupo Minoritário', 'Segurança Psicológica')

print("\nGerando gráfico para 'Grupo Minoritário' x 'Impacto no Desempenho'")
analise_correspondencia(df_questionario, 'GrupoMinoritario', 'ImpactoDesempenho', 'Grupo Minoritário x Impacto no Desempenho', 'Grupo Minoritário', 'Impacto no Desempenho')

print("Gerando gráfico para 'Grupo Minoritário' x 'Liderança Inclusiva'")
analise_correspondencia(df_questionario, 'GrupoMinoritario', 'LiderancaInclusiva', 'Grupo Minoritário x Liderança Inclusiva', 'Grupo Minoritário', 'ILideranca Inclusiva')

print("\nGerando gráfico para 'Discriminacao' x 'Segurança Psicológica'")
analise_correspondencia(df_questionario, 'Discriminacao', 'SegurancaPsicologica', 'Discriminação x Segurança Psicológica', 'Discriminação', 'Segurança Psicológica')

print("\nGerando gráfico para 'Discriminacao' x ' Impacto no Desempenho'")
analise_correspondencia(df_questionario, 'Discriminacao', 'ImpactoDesempenho', 'Discriminação x Impacto no Desempenho', 'Discriminação', 'Impacto no Desempenho')

print("\nGerando gráfico para 'Gênero' x 'Bem-Estar Emocional'")
analise_correspondencia(df_questionario, 'Genero', 'BemEstarEmocional', 'Gênero x Bem-Estar Emocional', 'Gênero', 'Bem-Estar Emocional')

print("\nGerando gráfico para 'Gênero' x 'Segurança Psicológica'")
analise_correspondencia(df_questionario, 'Genero', 'SegurancaPsicologica', 'Gênero x Segurança Psicológica', 'Gênero', 'Segurança Psicológica')

print("\nGerando gráfico para 'Faixa Etária ' x 'Segurança Psicológica'")
analise_correspondencia(df_questionario, 'FaixaEtaria', 'SegurancaPsicologica', 'Faixa Etária x Seguranca Esicológica', 'Faixa Etária', 'Segurança Psicológica')

print("\nGerando gráfico para 'Faixa Etária ' x 'Bem-Estar Emocional'")
analise_correspondencia(df_questionario, 'FaixaEtaria', 'BemEstarEmocional', 'Faixa Etária x Bem-Estar Emocional', 'Faixa etária', 'Bem-Estar Emocional')

print("\nGerando gráfico para 'Escolaridade' x 'Segurança Psicológica'")
analise_correspondencia(df_questionario, 'Escolaridade', 'BemEstarEmocional', 'Escolaridade x Seguranca Esicológica', 'Escolaridade', 'Seguranca Esicológica')

print("\nGerando gráfico para 'Escolaridade' x 'Bem-Estar Emocional'")
analise_correspondencia(df_questionario, 'Escolaridade', 'BemEstarEmocional', 'Escolaridade x Bem-Estar Emocional', 'Escolaridade', 'Bem-Estar Emocional')

print("\nGerando gráfico para 'Segurança Psicológica' x 'Bem-Estar Emocional'")
analise_correspondencia(df_questionario, 'SegurancaPsicologica', 'BemEstarEmocional', 'Segurança Psicológica x Bem-Estar Emocional', 'Segurança Psicológica', 'Bem-Estar Emocional')

print("\nGerando gráfico para 'Experiência Ágil' x 'Segurança Psicológica''")
analise_correspondencia(df_questionario, 'ExperienciaAgil', 'SegurancaPsicologica', 'Experiência Ágil x Segurança Psicológica', 'Experiência Ágil', 'Segurança psicológica')

print("\nGerando gráfico para 'Experiência Ágil' x 'Bem-Estar Emocional'")
analise_correspondencia(df_questionario, 'ExperienciaAgil', 'BemEstarEmocional', 'Experiência Ágil x Bem-Estar Emocional', 'Experiência Ágil', 'Bem-Estar Emocional')

print("\nGerando gráfico para 'Diversidade' x 'Segurança Psicológica'")
analise_correspondencia(df_questionario, 'Diversidade', 'SegurancaPsicologica', 'Diversidade x Segurança Psicológica', 'Diversidade', 'Segurança Psicológica')

print("\nGerando gráfico para 'Representatividade' x 'Segurança Psicológica'")
analise_correspondencia(df_questionario, 'Representatividade', 'SegurancaPsicologica', 'Representatividade x Segurança Psicológica', 'Representatividade', 'Segurança Psicológica')

print("\nGerando gráfico para 'Representatividade na Liderança' x 'Segurança Psicológica'")
analise_correspondencia(df_questionario, 'RepresentatividadeNaLideranca', 'SegurancaPsicologica', 'Representatividade na Liderança x Segurança Psicológica', 'Representatividade na Liderança', 'Segurança Psicológica')

print("\nGerando gráfico para 'Liderança Inclusiva' x 'Segurança Psicológica'")
analise_correspondencia(df_questionario, 'LiderancaInclusiva', 'SegurancaPsicologica', 'Liderança Inclusiva x Segurança Psicológica', 'Liderança Inclusiva', 'Segurança psicológica')

print("\nGerando gráfico para 'Lideranca Inclusiva' x 'Bem-Estar Emocional'")
analise_correspondencia(df_questionario, 'LiderancaInclusiva', 'BemEstarEmocional', 'Liderança Inclusiva x Bem-Estar Emocional', 'Liderança Inclusiva', 'Bem-Estar Emocional')

print("\nGerando gráfico para 'Insegurança' x 'Segurança Psicológica'")
analise_correspondencia(df_questionario, 'Inseguranca', 'SegurancaPsicologica', 'Insegurança x Segurança Psicológica', 'Insegurança', 'Segurança Psicológica')

print("\nGerando gráfico para 'Impacto no Desempenho' x 'Inseguranca'")
analise_correspondencia(df_questionario, 'Inseguranca', 'ImpactoDesempenho', 'Impacto no Desempenho x Inseguranca', 'Impacto no Desempenho', 'Insegurança')

print("\nGerando gráfico para 'Impacto no Desempenho' x 'Representatividade'")
analise_correspondencia(df_questionario, 'ImpactoDesempenho', 'Representatividade', 'Impacto no Desempenho x Representatividade', 'Impacto no Desempenho', 'Representatividade')
