"""Treinamento e validação do modelo do RotaVerde MS.

Este arquivo responde a uma pergunta simples: dado o perfil de um bairro, quantas
toneladas de recicláveis ele deve gerar por mês?

Nossa base tem só 12 bairros, e isso pede cuidado na hora de avaliar o modelo. Se
separarmos 25% para teste, sobram apenas 3 bairros para medir o desempenho, e o
resultado muda muito dependendo de quais 3 caem no sorteio. Por isso o script faz
três coisas:

  1. mostra o quanto essa avaliação "de sorteio" oscila, testando 40 sementes;
  2. compara os modelos com Leave-One-Out: cada bairro é previsto por um modelo
     que aprendeu só com os outros 11, e repetimos isso para todos os 12;
  3. salva o modelo que errou menos em modelo_residuos.pkl.

Um detalhe sobre as variáveis: população e número de domicílios andam praticamente
juntas (correlação de 0,999), porque cada casa tem em média 3,2 pessoas. Usar as
duas seria contar a mesma informação duas vezes. Por isso transformamos domicílios
em "pessoas por domicílio", que traz algo novo para o modelo.

Para rodar:
    python modelo.py
"""
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split

# Os caminhos partem da pasta deste arquivo, e não da pasta de onde o comando foi
# executado. Assim o projeto funciona mesmo se alguém rodar de outro lugar.
PASTA = Path(__file__).resolve().parent
ARQUIVO_DADOS = PASTA / "dados_residuos.csv"
ARQUIVO_MODELO = PASTA / "modelo_residuos.pkl"

# O que queremos prever: toneladas de recicláveis por mês em cada bairro.
ALVO = "volume_reciclavel_ton"

# As informações que o modelo pode usar para fazer a previsão.
FEATURES = [
    "populacao_bairro",
    "pessoas_por_domicilio",
    "comercio_local",
    "taxa_adesao_historica",
]

# Semente fixa para que o RandomForest dê sempre o mesmo resultado.
SEMENTE = 42


def adicionar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a coluna de pessoas por domicílio a partir das colunas originais."""
    df = df.copy()
    df["pessoas_por_domicilio"] = df["populacao_bairro"] / df["domicilios"]
    return df


def carregar_dados(caminho: Path = ARQUIVO_DADOS) -> pd.DataFrame:
    """Lê o CSV dos bairros e já deixa a coluna derivada pronta."""
    return adicionar_features(pd.read_csv(caminho))


def candidatos() -> dict:
    """Lista os modelos que vamos colocar para competir.

    Cada item guarda o modelo e as colunas que ele enxerga. O baseline, que sempre
    chuta a média, está aqui como régua: um modelo que não ganha dele não serve.
    """
    return {
        "RandomForest (4 features)": (
            RandomForestRegressor(n_estimators=50, random_state=SEMENTE), FEATURES),
        "Regressão linear (4 features)": (LinearRegression(), FEATURES),
        "Regressão linear (só população)": (LinearRegression(), ["populacao_bairro"]),
        "Baseline (prever a média)": (DummyRegressor(strategy="mean"), FEATURES),
    }


def validar(df: pd.DataFrame) -> pd.DataFrame:
    """Avalia todos os candidatos com Leave-One-Out.

    Para cada modelo, prevemos cada bairro usando um modelo treinado sem ele. No
    fim, comparamos as 12 previsões com os valores reais. A tabela sai ordenada do
    menor para o maior erro médio (MAE, em toneladas).
    """
    y = df[ALVO]
    linhas = []
    for nome, (modelo, cols) in candidatos().items():
        pred = cross_val_predict(modelo, df[cols], y, cv=LeaveOneOut())
        linhas.append({
            "modelo": nome,
            "features": ", ".join(cols),
            "mae_ton": mean_absolute_error(y, pred),
            "r2": r2_score(y, pred),
        })
    return pd.DataFrame(linhas).sort_values("mae_ton").reset_index(drop=True)


# As colunas que a primeira versão do projeto usava, com domicílios direto.
FEATURES_VERSAO_ANTIGA = ["populacao_bairro", "domicilios", "comercio_local",
                          "taxa_adesao_historica"]


def estabilidade_divisao_unica(df: pd.DataFrame, n_sementes: int = 40) -> pd.DataFrame:
    """Refaz a avaliação da primeira versão, mudando apenas o sorteio.

    Mesmo modelo, mesmas colunas e mesma divisão 75/25 de antes. A única coisa que
    muda é a semente que decide quais bairros vão para o teste. Serve para mostrar
    por que um único R² não era confiável com tão poucos dados.
    """
    res = []
    for s in range(n_sementes):
        tr, te = train_test_split(df, test_size=0.25, random_state=s)
        m = RandomForestRegressor(n_estimators=50, random_state=SEMENTE)
        m.fit(tr[FEATURES_VERSAO_ANTIGA], tr[ALVO])
        p = m.predict(te[FEATURES_VERSAO_ANTIGA])
        res.append({"semente": s, "r2": r2_score(te[ALVO], p),
                    "mae_ton": mean_absolute_error(te[ALVO], p)})
    return pd.DataFrame(res)


def treinar(df: pd.DataFrame | None = None) -> dict:
    """Escolhe o melhor modelo, treina com todos os bairros e salva em disco.

    Além do modelo, o arquivo salvo leva o que o painel precisa saber sobre ele:
    quais colunas usa, entre quais valores foi treinado e como se saiu na validação.
    """
    df = carregar_dados() if df is None else df
    tabela = validar(df)

    # O baseline fica de fora da escolha: ele só existe para comparação.
    elegiveis = tabela[~tabela["modelo"].str.startswith("Baseline")]
    vencedor = elegiveis.iloc[0]["modelo"]
    modelo, cols = candidatos()[vencedor]

    # Na validação, cada modelo via 11 bairros. O modelo final aprende com os 12.
    modelo.fit(df[cols], df[ALVO])

    pacote = {
        "modelo": modelo,
        "nome_modelo": vencedor,
        "features": cols,
        # O menor e o maior valor de cada coluna no treino. O painel usa isso para
        # avisar quando alguém pede uma previsão fora do que o modelo conhece.
        "faixa_treino": {c: (float(df[c].min()), float(df[c].max()))
                         for c in FEATURES + ["domicilios", ALVO]},
        # Guardamos a tabela como lista de dicionários, e não como DataFrame, para
        # que o arquivo abra em qualquer versão do pandas.
        "validacao": tabela.to_dict("records"),
        "n_amostras": len(df),
        # A regressão linear consegue prolongar a tendência para bairros maiores.
        # O RandomForest não: acima do maior bairro do treino, ele trava.
        "extrapola": not isinstance(modelo, RandomForestRegressor),
        # Se o painel rodar com outro scikit-learn, ele usa isto para saber que
        # precisa treinar de novo.
        "versao_sklearn": sklearn.__version__,
    }
    joblib.dump(pacote, ARQUIVO_MODELO)
    return pacote


def prever(pacote: dict, entradas: pd.DataFrame) -> pd.Series:
    """Estima o volume de recicláveis para um ou mais bairros.

    Recebe as colunas do jeito que estão no CSV. Se faltar a coluna de pessoas por
    domicílio, ela é calculada aqui.
    """
    if "pessoas_por_domicilio" not in entradas:
        entradas = adicionar_features(entradas)
    return pd.Series(pacote["modelo"].predict(entradas[pacote["features"]]),
                     index=entradas.index)


if __name__ == "__main__":
    dados = carregar_dados()
    print(f"Base: {ARQUIVO_DADOS.name} — {len(dados)} amostras\n")

    # Primeiro, mostramos por que domicílios não entra direto no modelo.
    corr = dados["populacao_bairro"].corr(dados["domicilios"])
    print(f"Correlação populacao_bairro x domicilios: {corr:.5f} "
          "-> domicilios entra só como pessoas_por_domicilio\n")

    # Depois, o quanto a avaliação antiga dependia da sorte.
    instab = estabilidade_divisao_unica(dados)
    print("Avaliação antiga (divisão única 75/25) em 40 sementes:")
    print(f"  R²  min {instab.r2.min():.3f} | mediana {instab.r2.median():.3f} | "
          f"max {instab.r2.max():.3f} | negativos: {(instab.r2 < 0).sum()}")
    print(f"  MAE min {instab.mae_ton.min():.2f} | mediana {instab.mae_ton.median():.2f} "
          f"| max {instab.mae_ton.max():.2f} ton\n")

    # Por fim, a comparação justa entre os modelos e o salvamento do vencedor.
    pacote = treinar(dados)
    print("Validação Leave-One-Out (12 previsões fora da amostra):")
    print(pd.DataFrame(pacote["validacao"])[["modelo", "mae_ton", "r2"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nModelo salvo: {pacote['nome_modelo']} -> {ARQUIVO_MODELO.name} "
          f"(scikit-learn {pacote['versao_sklearn']})")
