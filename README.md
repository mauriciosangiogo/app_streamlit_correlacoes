# 📊 App de Análise de Correlações

Aplicação web interativa desenvolvida em **Streamlit** para análise estatística de correlações entre variáveis, com visualizações profissionais e exportação de resultados.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

-----

## 🎯 Funcionalidades

- **Upload de dados**: Suporte a arquivos Excel (.xlsx, .xls) e CSV
- **Três métodos de correlação**: Pearson, Spearman e Kendall
- **Seleção flexível de variáveis**: Escolha independente de variáveis dependentes e independentes
- **Matriz de correlação interativa**: Heatmap com anotações de significância estatística
- **Classificação automática**: Força da correlação (Muito Fraca a Muito Forte)
- **Nível de significância configurável**: Alpha ajustável pelo usuário
- **Exportação completa**: Resultados em Excel e visualizações em PNG
- **Gráficos descritivos**: Barras horizontais com estilo limpo e profissional
- **Autenticação**: Sistema de login para acesso controlado

-----

## 🚀 Instalação

### Pré-requisitos

- Python 3.9 ou superior
- pip (gerenciador de pacotes)

### Passos

1. **Clone o repositório**

```bash
git clone https://github.com/seu-usuario/app-correlacoes.git
cd app-correlacoes
```

1. **Crie um ambiente virtual** (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

1. **Instale as dependências**

```bash
pip install -r requirements.txt
```

1. **Execute a aplicação**

```bash
streamlit run app_correlacao.py
```

-----

## 📦 Dependências

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
xlsxwriter>=3.1.0
```

-----

## 🔧 Configuração

### Autenticação (opcional)

Para habilitar o sistema de login no Streamlit Cloud, crie o arquivo `.streamlit/secrets.toml`:

```toml
[credentials]
username = "seu_usuario"
password = "sua_senha"
```

Para deploy local, você pode desabilitar a autenticação comentando a função de login no código.

### Estrutura do Projeto

```
app-correlacoes/
├── app_correlacao.py      # Aplicação principal
├── requirements.txt       # Dependências
├── README.md             # Documentação
├── .streamlit/
│   ├── config.toml       # Configurações do Streamlit
│   └── secrets.toml      # Credenciais (não versionar)
└── .gitignore
```

-----

## 📖 Como Usar

### 1. Upload dos Dados

Faça upload de um arquivo Excel contendo suas variáveis numéricas. A primeira linha deve conter os nomes das colunas.

### 2. Configuração da Análise

Na barra lateral, configure:

- **Método de correlação**: Pearson (dados normais), Spearman (não-paramétrico) ou Kendall (amostras pequenas)
- **Nível de significância (α)**: Padrão 0.05
- **Variáveis dependentes**: Selecione uma ou mais
- **Variáveis independentes**: Selecione uma ou mais

### 3. Interpretação dos Resultados

A aplicação gera:

|Aba                 |Conteúdo                                                    |
|--------------------|------------------------------------------------------------|
|**Heatmap**         |Matriz visual com coeficientes e asteriscos de significância|
|**Tabela Formatada**|Resultados com classificação de força e direção             |
|**P-valores**       |Matriz de significância estatística                         |
|**Dados Brutos**    |Valores numéricos para análise adicional                    |

**Legenda de significância**: `***` p<0.001 · `**` p<0.01 · `*` p<0.05

**Classificação de força**:

- |r| < 0.10: Muito Fraca
- 0.10 ≤ |r| < 0.30: Fraca
- 0.30 ≤ |r| < 0.50: Moderada
- 0.50 ≤ |r| < 0.70: Forte
- |r| ≥ 0.70: Muito Forte

### 4. Exportação

- **Excel**: Todas as tabelas em abas separadas
- **PNG**: Heatmap em alta resolução

-----

## 🌐 Deploy

### Streamlit Community Cloud (Gratuito)

1. Faça push do código para o GitHub
1. Acesse [share.streamlit.io](https://share.streamlit.io)
1. Conecte seu repositório
1. Configure as secrets (credenciais) no painel
1. Deploy automático



## 🧪 Exemplo de Uso

```python
# Dados de exemplo esperados
import pandas as pd

df = pd.DataFrame({
    'ano': [2020, 2021, 2022, 2023],
    'casos_doenca_a': [150, 180, 220, 195],
    'casos_doenca_b': [85, 92, 110, 98],
    'cobertura_vacinal': [78.5, 82.3, 85.1, 88.7],
    'saneamento_pct': [65.2, 68.4, 71.0, 73.5]
})
```

-----

## 📄 Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo <LICENSE> para detalhes.

-----

## 👤 Autor

**DataStrategy Consultoria**

- Análise de dados para pequenas e médias empresas
- Análise de dados acadêmicos
- Dashboards e visualizações interativas
- Automação de relatórios
- Automação na extração e tratamento de dados (ETL)

-----

## 🤝 Contribuições

Contribuições são bem-vindas! Para mudanças significativas, abra uma issue primeiro para discutir o que você gostaria de mudar.

-----

## 📞 Suporte

Para dúvidas ou suporte, entre em contato através do meu e-mail (ms_sangiogo@hotmail.com) ou telefone/whatsapp (53 991627836).
