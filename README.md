# RotaVerde MS — Previsão de resíduos recicláveis por bairro

Protótipo que estima quantas toneladas de recicláveis por mês um bairro gera, a partir de população, domicílios, estabelecimentos comerciais e pontos de coleta. Inclui um painel em Streamlit com simulador interativo e comparação de modelos.

> **Atenção: a base de dados é sintética.** `dados_residuos.csv` tem **12 bairros fictícios** gerados por `gerar_dados.py` (semente fixa). Os números não são medições reais. Servem para demonstrar o fluxo do modelo, não para planejar coleta.

## Estrutura

| Arquivo | Função |
|---|---|
| `gerar_dados.py` | Gera a base sintética `dados_residuos.csv` (reprodutível) |
| `dados_residuos.csv` | 12 bairros fictícios |
| `modelo.py` | Valida os modelos com Leave-One-Out, escolhe o de menor erro e salva `modelo_residuos.pkl` |
| `app.py` | Painel Streamlit: dados, simulador, validação e premissas |
| `.streamlit/config.toml` | Tema claro/escuro do painel |
| `test_rotaverde.py` | Testes automatizados de dados, modelo e interface (`AppTest`) |
| `requirements.txt` | Versões exatas das bibliotecas |

## Como rodar

```powershell
python -m venv .venv
.venv\Scripts\activate        # em alguns Pythons a pasta é .venv\bin
pip install -r requirements.txt
python modelo.py              # treina, valida e salva o modelo
streamlit run app.py
```

Para rodar os testes:

```powershell
python -m unittest -v test_rotaverde.py
```

O `app.py` funciona a partir de qualquer pasta: todos os caminhos partem de `Path(__file__).parent`. Se o `.pkl` não existir, estiver corrompido, tiver sido salvo com outra versão do scikit-learn ou se `dados_residuos.csv` tiver mudado, o app retreina o modelo automaticamente.

## Metodologia

**Validação.** Com 12 amostras, uma divisão treino/teste de 75/25 deixa só 3 bairros no teste e o R² passa a depender da semente sorteada. `python modelo.py` mostra essa variação em 40 sementes. Por isso a avaliação usa **Leave-One-Out**: cada bairro é previsto por um modelo treinado nos outros 11.

**Comparação de modelos.** São comparados, nas mesmas condições:

- RandomForest (4 features)
- Regressão linear (4 features)
- Regressão linear (só população)
- Baseline que prevê a média

Fica como padrão o modelo de menor MAE. Nesta base, que é quase linear, a regressão linear vence o RandomForest. O RandomForest aproxima a tendência por degraus e não extrapola fora da faixa vista no treino.

**Features.** `domicilios` tem correlação de cerca de 0,9999 com `populacao_bairro`. Por isso não entra direto: usamos a derivada `pessoas_por_domicilio`. Isso evita uma importância de variáveis dividida ao acaso entre duas colunas redundantes.

**Simulador.** Usa por padrão o modelo vencedor (só população, exibindo apenas os campos realmente usados por ele) e permite alternar entre os candidatos validados. A população fica travada na faixa do treino até o usuário habilitar a extrapolação; qualquer variável fora da faixa treinada gera alerta, e previsões físicas são limitadas a valores não negativos (`>= 0`).

## Premissas e limitações

- **Fator de aterro evitado = 0,85** é uma *premissa sem fonte citada*. O painel deixa ajustar o valor na aba *Premissas* e atualiza o simulador automaticamente. Substitua por um dado com referência antes de apresentar o resultado.
- Dados agregados por bairro e estáticos: não há série temporal por ponto de coleta.
- Fora do escopo deste código: clusterização (KMeans), detecção de anomalias (IsolationForest), classificação de prioridade e validação temporal (TimeSeriesSplit). São **roadmap**, condicionados a ter dados reais por ponto de coleta ao longo do tempo.
