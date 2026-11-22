"""Gera a base SINTÉTICA dados_residuos.csv usada pelo protótipo RotaVerde MS.

Os valores não vêm de medição real: são produzidos por uma relação aproximadamente
linear entre população e resíduos recicláveis, com ruído pequeno. O script existe para
que qualquer pessoa reproduza exatamente a mesma base (semente fixa).

Uso:
    python gerar_dados.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PASTA = Path(__file__).resolve().parent
ARQUIVO = PASTA / "dados_residuos.csv"
SEMENTE = 2026

BAIRROS = [
    "Jardim Aurora",
    "Vila Esperança",
    "Parque dos Ipês",
    "Residencial Pantanal",
    "Vila Nova Serra",
    "Jardim das Araras",
    "Centro Norte",
    "Bairro do Lago",
    "Parque Bodoquena",
    "Vila Maracaju",
    "Jardim Taquari",
    "Setor Pioneiros",
]
POPULACAO = [
    2200,
    3600,
    4900,
    6300,
    7800,
    9100,
    10800,
    12500,
    14600,
    16700,
    18900,
    21000,
]

# Premissas da geração (não são dados oficiais):
KG_POR_HAB_MES = 1.32  # recicláveis gerados por habitante/mês
PESSOAS_POR_DOMICILIO = 3.2
ESTAB_POR_MIL_HAB = 9.5  # estabelecimentos comerciais por mil habitantes
TON_POR_ESTAB_MES = 0.012


def gerar() -> pd.DataFrame:
    if len(BAIRROS) != len(POPULACAO):
        raise ValueError("BAIRROS e POPULACAO devem ter o mesmo comprimento.")

    rng = np.random.default_rng(SEMENTE)
    pop = np.array(POPULACAO, dtype=float)

    domicilios = np.maximum(
        1.0, np.round(pop / (PESSOAS_POR_DOMICILIO + rng.normal(0, 0.04, pop.size)))
    )
    estab = np.maximum(
        0.0, np.round(pop / 1000 * ESTAB_POR_MIL_HAB * rng.normal(1, 0.05, pop.size))
    )
    pontos = np.maximum(1.0, np.round(pop / 2500 + rng.normal(0, 0.4, pop.size)))

    residuos = np.maximum(
        0.0,
        pop * KG_POR_HAB_MES / 1000
        + estab * TON_POR_ESTAB_MES
        + rng.normal(0, 0.35, pop.size),
    )

    return pd.DataFrame(
        {
            "bairro": BAIRROS,
            "populacao_bairro": pop.astype(int),
            "domicilios": domicilios.astype(int),
            "estabelecimentos_comerciais": estab.astype(int),
            "pontos_coleta": pontos.astype(int),
            "residuos_reciclaveis_ton_mes": np.round(residuos, 2),
        }
    )


if __name__ == "__main__":
    df = gerar()
    df.to_csv(ARQUIVO, index=False)
    print(df.to_string(index=False))
    print(f"\nBase sintética salva em {ARQUIVO} ({len(df)} amostras)")
