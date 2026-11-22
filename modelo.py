"""Treina e valida o modelo preditivo do RotaVerde MS.

Compara modelos candidatos por validação Leave-One-Out (LOO) e seleciona o de
menor MAE. Salva o pacote completo (modelo vencedor, candidatos ajustados,
previsões LOO, faixas de treino e metadados de integridade) em
`modelo_residuos.pkl`.

Uso:
    python modelo.py
"""

import hashlib
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

PASTA = Path(__file__).resolve().parent
ARQUIVO_DADOS = PASTA / "dados_residuos.csv"
ARQUIVO_MODELO = PASTA / "modelo_residuos.pkl"

ALVO = "residuos_reciclaveis_ton_mes"
FEATURES = [
    "populacao_bairro",
    "pessoas_por_domicilio",
    "estabelecimentos_comerciais",
    "pontos_coleta",
]
SEMENTE = 42

CHAVES_OBRIGATORIAS = {
    "modelo",
    "modelos",
    "nome_modelo",
    "features",
    "faixa_treino",
    "validacao",
    "previsoes_loo",
    "n_amostras",
    "base_sintetica",
    "versao_sklearn",
    "extrapola",
    "hash_dados",
}


def hash_arquivo(caminho: Path = ARQUIVO_DADOS) -> str:
    """Retorna o SHA-256 hexadecimal do arquivo de dados."""
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def carregar_dados(caminho: Path = ARQUIVO_DADOS) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    return adicionar_features(df)


def adicionar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "domicilios" in df.columns:
        if (df["domicilios"] <= 0).any():
            raise ValueError("`domicilios` deve ser maior que zero.")
        df["pessoas_por_domicilio"] = df["populacao_bairro"] / df["domicilios"]
    return df


def candidatos() -> dict[str, tuple[Any, list[str]]]:
    """Modelos comparados. Cada item: (estimador, lista de features)."""
    return {
        "RandomForest": (
            RandomForestRegressor(n_estimators=200, random_state=SEMENTE),
            FEATURES,
        ),
        "Regressão linear (4 features)": (LinearRegression(), FEATURES),
        "Regressão linear (só população)": (LinearRegression(), ["populacao_bairro"]),
        "Baseline (prever a média)": (DummyRegressor(strategy="mean"), FEATURES),
    }


def validar(df: pd.DataFrame) -> pd.DataFrame:
    """Validação Leave-One-Out: cada amostra é prevista por um modelo que não a viu."""
    tabela, _ = _avaliar_loo(df)
    return tabela


def _avaliar_loo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa LOO para todos os candidatos e devolve (métricas, previsões por amostra)."""
    y = df[ALVO].to_numpy()
    linhas: list[dict[str, Any]] = []
    previsoes = pd.DataFrame(
        {
            "bairro": df["bairro"].to_numpy(),
            "populacao_bairro": df["populacao_bairro"].to_numpy(),
            "real_ton": y,
        }
    )
    for nome, (estimador, cols) in candidatos().items():
        pred = cross_val_predict(estimador, df[cols], y, cv=LeaveOneOut())
        linhas.append(
            {
                "modelo": nome,
                "features": ", ".join(cols),
                "mae_ton": float(mean_absolute_error(y, pred)),
                "r2": float(r2_score(y, pred)),
            }
        )
        if not nome.startswith("Baseline"):
            previsoes[nome] = pred

    tabela = pd.DataFrame(linhas).sort_values("mae_ton").reset_index(drop=True)
    return tabela, previsoes


def treinar(
    df: pd.DataFrame | None = None,
    caminho_dados: Path = ARQUIVO_DADOS,
    caminho_modelo: Path = ARQUIVO_MODELO,
) -> dict[str, Any]:
    dados_df = carregar_dados(caminho_dados) if df is None else adicionar_features(df)
    tabela, previsoes_loo = _avaliar_loo(dados_df)

    elegiveis = tabela[~tabela["modelo"].str.startswith("Baseline")]
    vencedor = str(elegiveis.iloc[0]["modelo"])

    modelos_treinados: dict[str, dict[str, Any]] = {}
    for nome, (estimador, cols) in candidatos().items():
        if nome.startswith("Baseline"):
            continue
        estimador.fit(dados_df[cols], dados_df[ALVO])
        modelos_treinados[nome] = {
            "modelo": estimador,
            "features": list(cols),
            "extrapola": not isinstance(estimador, RandomForestRegressor),
        }

    faixa = {
        col: (float(dados_df[col].min()), float(dados_df[col].max()))
        for col in FEATURES + ["domicilios", ALVO]
    }

    vencedor_info = modelos_treinados[vencedor]
    pacote: dict[str, Any] = {
        "modelo": vencedor_info["modelo"],
        "modelos": modelos_treinados,
        "nome_modelo": vencedor,
        "features": vencedor_info["features"],
        "faixa_treino": faixa,
        "validacao": tabela,
        "previsoes_loo": previsoes_loo,
        "n_amostras": int(len(dados_df)),
        "base_sintetica": True,
        "versao_sklearn": sklearn.__version__,
        "extrapola": vencedor_info["extrapola"],
        "hash_dados": hash_arquivo(caminho_dados) if caminho_dados.exists() else "",
    }
    joblib.dump(pacote, caminho_modelo)
    return pacote


def pacote_valido(
    pacote: Any, caminho_dados: Path = ARQUIVO_DADOS
) -> bool:
    """Verifica se um pacote carregado corresponde ao ambiente e aos dados atuais."""
    if not isinstance(pacote, dict):
        return False
    if not CHAVES_OBRIGATORIAS.issubset(pacote.keys()):
        return False
    if pacote.get("versao_sklearn") != sklearn.__version__:
        return False
    if caminho_dados.exists() and pacote.get("hash_dados") != hash_arquivo(caminho_dados):
        return False
    return True


def carregar_ou_treinar(
    caminho_modelo: Path = ARQUIVO_MODELO,
    caminho_dados: Path = ARQUIVO_DADOS,
) -> dict[str, Any]:
    """Carrega o pacote salvo ou retreina se ausente, corrompido ou desatualizado."""
    if caminho_modelo.exists():
        try:
            pacote = joblib.load(caminho_modelo)
            if pacote_valido(pacote, caminho_dados):
                return pacote
        except Exception:
            pass
    return treinar(caminho_dados=caminho_dados, caminho_modelo=caminho_modelo)


def _info_modelo(pacote: dict[str, Any], nome_modelo: str | None = None) -> dict[str, Any]:
    if nome_modelo is not None and "modelos" in pacote and nome_modelo in pacote["modelos"]:
        return pacote["modelos"][nome_modelo]
    return {
        "modelo": pacote["modelo"],
        "features": pacote["features"],
        "extrapola": pacote["extrapola"],
    }


def features_fora_da_faixa(
    pacote: dict[str, Any],
    entrada: dict[str, float],
    nome_modelo: str | None = None,
) -> list[tuple[str, float, float, float]]:
    """Lista `(feature, valor, min_treino, max_treino)` fora da faixa vista no treino."""
    info = _info_modelo(pacote, nome_modelo)
    linha = pd.DataFrame([entrada])
    if "pessoas_por_domicilio" in info["features"] and "pessoas_por_domicilio" not in linha:
        linha = adicionar_features(linha)

    fora: list[tuple[str, float, float, float]] = []
    for col in info["features"]:
        vmin, vmax = pacote["faixa_treino"][col]
        val = float(linha[col].iloc[0])
        if val < vmin or val > vmax:
            fora.append((col, val, vmin, vmax))
    return fora


def prever(
    pacote: dict[str, Any],
    entrada: dict[str, float],
    nome_modelo: str | None = None,
) -> float:
    """Prevê toneladas/mês de recicláveis (limitado a >= 0,0)."""
    info = _info_modelo(pacote, nome_modelo)
    cols = info["features"]
    linha = pd.DataFrame([entrada])
    if "pessoas_por_domicilio" in cols and "pessoas_por_domicilio" not in linha:
        linha = adicionar_features(linha)

    for col in ("populacao_bairro", "estabelecimentos_comerciais", "pontos_coleta"):
        if col in cols and float(linha[col].iloc[0]) < 0:
            raise ValueError(f"`{col}` não pode ser negativo.")

    bruto = float(info["modelo"].predict(linha[cols])[0])
    return max(0.0, bruto)


def estabilidade_divisao_unica(df: pd.DataFrame, n_sementes: int = 40) -> pd.DataFrame:
    """Mostra quanto o R² de uma divisão única 75/25 varia com a semente sorteada."""
    from sklearn.model_selection import train_test_split

    res = []
    for s in range(n_sementes):
        tr, te = train_test_split(df, test_size=0.25, random_state=s)
        m = RandomForestRegressor(n_estimators=200, random_state=SEMENTE)
        m.fit(tr[FEATURES], tr[ALVO])
        p = m.predict(te[FEATURES])
        res.append(
            {
                "semente": s,
                "r2": float(r2_score(te[ALVO], p)),
                "mae_ton": float(mean_absolute_error(te[ALVO], p)),
            }
        )
    return pd.DataFrame(res)


if __name__ == "__main__":
    dados = carregar_dados()
    print(f"Base: {ARQUIVO_DADOS.name} — {len(dados)} amostras (SINTÉTICA)\n")

    corr = float(dados[["populacao_bairro", "domicilios"]].corr().iloc[0, 1])
    print(
        f"Correlação populacao_bairro x domicilios: {corr:.5f} "
        "-> domicilios fica fora das features\n"
    )

    instab = estabilidade_divisao_unica(dados)
    print("Divisão única 75/25 em 40 sementes (por que NÃO usamos):")
    print(
        f"  R²  min {instab.r2.min():.3f} | mediana {instab.r2.median():.3f} | "
        f"max {instab.r2.max():.3f} | negativos: {(instab.r2 < 0).sum()}"
    )
    print(
        f"  MAE min {instab.mae_ton.min():.2f} | mediana {instab.mae_ton.median():.2f} | "
        f"max {instab.mae_ton.max():.2f} ton\n"
    )

    pacote = treinar(dados)
    print(f"Validação Leave-One-Out ({len(dados)} previsões fora da amostra):")
    print(
        pacote["validacao"][["modelo", "mae_ton", "r2"]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )
    )
    print(
        f"\nModelo salvo: {pacote['nome_modelo']} -> {ARQUIVO_MODELO.name} "
        f"(scikit-learn {pacote['versao_sklearn']})"
    )
