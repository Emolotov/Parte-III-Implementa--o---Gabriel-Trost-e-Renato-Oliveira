# RotaVerde MS — Previsão de Coleta Seletiva

**Disciplina:** Inteligência Artificial (2026.2) — UFMS Câmpus Coxim  
**Autores:** Gabriel Lemos Trost & Renato Oliveira Alexandre  
**Proposta:** Aplicação de IA para previsão de resíduos recicláveis e roteamento prioritário para prefeituras e cooperativas de Mato Grosso do Sul.

> **Sobre os dados:** `dados_residuos.csv` é uma **base sintética com 12 bairros**, construída para demonstrar o fluxo da solução. Os valores não são medições reais de coleta. Os indicadores dos bairros do ranking (Centro, Vila Bela, Flávio Garcia, Senhor Divino, Silviolândia) também são ilustrativos.

---

## Estrutura dos Arquivos
- `dados_residuos.csv`: base sintética com população, domicílios, comércio local, taxa de adesão e volume reciclável por bairro.
- `modelo.py`: compara RandomForest, regressão linear e um baseline com validação Leave-One-Out e salva o modelo de menor erro.
- `modelo_residuos.pkl`: modelo treinado, com a tabela de validação e a faixa de valores vista no treino.
- `app.py`: interface web em Streamlit com ranking de bairros, simulador, validação e premissas.
- `requirements.txt`: dependências; o scikit-learn fica em versão exata porque o `.pkl` é salvo com ela.

---

## Como Executar

Requer **Python 3.11 ou superior** (testado no 3.11 e no 3.14).

1. Instalar as dependências:
```bash
pip install -r requirements.txt
```

2. Executar o treinamento do modelo (opcional, modelo já incluso):
```bash
python modelo.py
```

3. Iniciar a interface web:
```bash
streamlit run app.py
```

O app pode ser iniciado de qualquer pasta. Se o `.pkl` não existir ou tiver sido salvo com outra versão do scikit-learn, o app retreina o modelo sozinho.

---

## Metodologia

**Validação.** Com 12 bairros, uma divisão treino/teste de 75/25 deixa só 3 amostras de teste. Variando apenas a semente dessa divisão, o R² do RandomForest oscila de valores negativos a quase 1 (`python modelo.py` mostra a faixa). Por isso a avaliação usa **Leave-One-Out**: cada bairro é previsto por um modelo treinado nos outros 11.

**Comparação de modelos (Leave-One-Out):**

| Modelo | MAE (ton/mês) | R² |
|---|---|---|
| Regressão linear (só população) | 0,42 | 0,996 |
| Regressão linear (4 features) | 0,54 | 0,993 |
| RandomForest (4 features) | 2,04 | 0,883 |
| Baseline (prever a média) | 7,59 | −0,190 |

Testamos RandomForest e regressão linear, e o modelo simples venceu nesta base. Todas as variáveis têm correlação acima de 0,97 com o volume, então a relação é praticamente uma reta. O RandomForest aproxima uma reta por degraus e não prevê fora da faixa vista no treino.

**Variáveis.** `domicilios` tem correlação de 0,9997 com `populacao_bairro`. Usada direto, a importância se dividiria ao acaso entre as duas. Ela entra apenas como `pessoas_por_domicilio`.

**Simulador.** A população fica limitada à faixa do treino (3.200 a 21.000 habitantes). Para testar bairros maiores, é preciso ligar a extrapolação, e o painel avisa que o resultado é uma estimativa sem dados que a confirmem.

---

## Premissas e Limitações

- **Aterro evitado = volume × 0,85.** O fator 0,85 é uma *premissa sem fonte citada* e pode ser ajustado na aba Premissas do painel.
- Os dados são agregados por bairro e estáticos: não há série temporal por ponto de coleta.

## Escopo Atual e Roadmap

Este código é a **prova de conceito da frente 1** (previsão de volume por bairro e ranking de prioridade). As frentes abaixo fazem parte da proposta, mas **ainda não estão implementadas** e dependem de dados reais por ponto de coleta ao longo do tempo:

- clusterização de pontos de coleta (KMeans);
- detecção de anomalias no volume coletado (IsolationForest);
- classificação de prioridade de rotas;
- validação temporal (TimeSeriesSplit).
