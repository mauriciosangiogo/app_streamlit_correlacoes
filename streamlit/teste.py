
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns




# Carregar dados
df = pd.read_excel('dados.xlsx')

# Definir as variáveis de interesse
vars_saneamento = ['atendimento_agua_perc', 'extensao_rede_agua', 
                   'atendimento_esgoto_perc', 'coleta_esgoto_perc', 
                   'esgoto_tratado_perc']

vars_doencas = ['tifoide', 'gastroenterite', 'amebiase', 'colera']

# Selecionar apenas as colunas de interesse
# subset() filtra o dataframe para trabalhar apenas com as variáveis necessárias
df_analise = df[vars_saneamento + vars_doencas]

#


# Descritivas

cores_barras = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
    'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
    'tab:olive', 'tab:cyan',
    'lightblue', 'peachpuff', 'lightgreen', 'salmon',
    'plum', 'wheat', 'lightpink', 'silver',
    'yellowgreen', 'paleturquoise'
]

df_sorted = df.sort_values(by='ano', ascending=True)

legenda_y = 'Número de internações'
legenda_x = ''
titulo = 'Número de internações por febre tifoide'

var_x = df_sorted['ano']
var_y = df_sorted['tifoide']

print(df.describe())
