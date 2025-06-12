import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

df = pd.read_csv('respostas.csv')
df.columns = df.columns.str.strip().str.replace('\n', ' ')

df = df.rename(columns={
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

df = df.drop(columns=['Carimbo de data/hora'], errors='ignore')

def testar_associacao(col1, col2):
    tabela = pd.crosstab(df[col1], df[col2])
    if tabela.empty or min(tabela.shape) < 2:
        print(f"{col1} x {col2}: tabela inválida para teste.")
        return

    chi2, p, dof, expected = chi2_contingency(tabela)

    # Mostra a tabela esperada
    df_expected = pd.DataFrame(expected, index=tabela.index, columns=tabela.columns)
    print("Frequências esperadas:")
    print(df_expected.round(2))

    n = tabela.sum().sum()
    phi2 = chi2 / n
    r, k = tabela.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    cramers_v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    print(f"\n{col1} x {col2}")
    print(f"χ² = {chi2:.3f}, gl = {dof}, p-valor = {p:.4f}")
    print(f"V de Cramer = {cramers_v:.3f}")
    if p < 0.05:
        print("→ Associação estatisticamente significativa.")
    else:
        print("→ Sem evidência estatística de associação.")

# Lista de pares a testar (mesma ordem do arquivo de ANACOR)
pares = [
    ('GrupoMinoritario', 'SegurancaPsicologica'),
    ('GrupoMinoritario', 'ImpactoDesempenho'),
    ('GrupoMinoritario', 'LiderancaInclusiva'),
    ('Discriminacao', 'SegurancaPsicologica'),
    ('Discriminacao', 'ImpactoDesempenho'),
    ('Genero', 'BemEstarEmocional'),
    ('Genero', 'SegurancaPsicologica'),
    ('FaixaEtaria', 'SegurancaPsicologica'),
    ('FaixaEtaria', 'BemEstarEmocional'),
    ('Escolaridade', 'SegurancaPsicologica'),
    ('Escolaridade', 'BemEstarEmocional'),
    ('SegurancaPsicologica', 'BemEstarEmocional'),
    ('ExperienciaAgil', 'SegurancaPsicologica'),
    ('ExperienciaAgil', 'BemEstarEmocional'),
    ('Diversidade', 'SegurancaPsicologica'),
    ('Representatividade', 'SegurancaPsicologica'),
    ('RepresentatividadeNaLideranca', 'SegurancaPsicologica'),
    ('LiderancaInclusiva', 'SegurancaPsicologica'),
    ('LiderancaInclusiva', 'BemEstarEmocional'),
    ('Inseguranca', 'SegurancaPsicologica'),
    ('Inseguranca', 'ImpactoDesempenho'),
    ('Representatividade', 'ImpactoDesempenho')
]

# Executa todos os testes
for col1, col2 in pares:
    testar_associacao(col1, col2)
