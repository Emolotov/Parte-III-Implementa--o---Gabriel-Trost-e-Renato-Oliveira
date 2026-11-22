"""Painel Streamlit do RotaVerde MS.

Uso:
    streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from modelo import (
    ALVO,
    carregar_dados,
    carregar_ou_treinar,
    features_fora_da_faixa,
    prever,
)

# Premissa SEM fonte citada: fração dos recicláveis coletados que deixa de ir ao aterro.
# Substitua por um valor com referência (ex.: relatório municipal ou SNIS) antes de
# apresentar como resultado.
FATOR_ATERRO_EVITADO_PADRAO = 0.85

ROTULOS_FEATURES = {
    "populacao_bairro": ("População do bairro", 0),
    "pessoas_por_domicilio": ("Pessoas por domicílio", 2),
    "estabelecimentos_comerciais": ("Estabelecimentos comerciais", 0),
    "pontos_coleta": ("Pontos de coleta", 0),
}


def br(valor: float, casas: int = 0) -> str:
    """Formata número no padrão brasileiro: 12.345,67 (sem -0,00)."""
    if abs(valor) < 0.5 * (10.0 ** (-casas)):
        valor = 0.0
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


st.set_page_config(
    page_title="RotaVerde MS",
    page_icon=":material/recycling:",
    layout="wide",
)


@st.cache_resource
def carregar_modelo() -> dict:
    return carregar_ou_treinar()


@st.cache_data
def dados() -> pd.DataFrame:
    return carregar_dados()


pacote = carregar_modelo()
df = dados()
pop_min, pop_max = pacote["faixa_treino"]["populacao_bairro"]
media = df.mean(numeric_only=True)
modelos_disponiveis = list(pacote["modelos"].keys())
melhor_modelo = pacote["nome_modelo"]

# Inicialização centralizada de session_state
st.session_state.setdefault("fator_aterro", FATOR_ATERRO_EVITADO_PADRAO)
st.session_state.setdefault("sim_modelo", melhor_modelo)
st.session_state.setdefault("sim_extrapolar", False)
st.session_state.setdefault("sim_pop", float(round(media["populacao_bairro"], -2)))
st.session_state.setdefault("sim_dom", float(round(media["domicilios"])))
st.session_state.setdefault("sim_estab", float(round(media["estabelecimentos_comerciais"])))
st.session_state.setdefault("sim_pontos", float(round(media["pontos_coleta"])))
st.session_state.setdefault("ultima_simulacao", None)

st.title("RotaVerde MS", icon=":material/recycling:")
st.caption(
    f"Previsão de resíduos recicláveis por bairro · protótipo · "
    f":orange-badge[Base sintética] {pacote['n_amostras']} amostras fictícias (não são medições reais)"
)

aba_dados, aba_sim, aba_val, aba_prem = st.tabs(
    [
        ":material/table_chart: Dados",
        ":material/calculate: Simulador",
        ":material/verified: Validação do modelo",
        ":material/tune: Premissas",
    ]
)

# --------------------------------------------------------------------------- Dados
with aba_dados:
    st.subheader("Base de dados", icon=":material/table_chart:")
    st.info(
        "Base gerada por `gerar_dados.py` com semente fixa para demonstrar o fluxo do modelo. "
        "Os números não representam medições reais de bairros.",
        title="Base sintética para demonstração",
        icon=":material/science:",
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Bairros na base",
        br(len(df), 0),
        border=True,
        height="stretch",
        icon=":material/location_city:",
    )
    k2.metric(
        "População total",
        br(float(df["populacao_bairro"].sum()), 0),
        border=True,
        height="stretch",
        icon=":material/groups:",
    )
    k3.metric(
        "Recicláveis totais",
        f"{br(float(df[ALVO].sum()), 2)} ton/mês",
        border=True,
        height="stretch",
        icon=":material/recycling:",
    )
    k4.metric(
        "Média por bairro",
        f"{br(float(df[ALVO].mean()), 2)} ton/mês",
        border=True,
        height="stretch",
        icon=":material/analytics:",
    )

    st.dataframe(
        df.drop(columns="pessoas_por_domicilio"),
        hide_index=True,
        column_config={
            "bairro": st.column_config.TextColumn("Bairro"),
            "populacao_bairro": st.column_config.NumberColumn(
                "População", format="localized"
            ),
            "domicilios": st.column_config.NumberColumn(
                "Domicílios", format="localized"
            ),
            "estabelecimentos_comerciais": st.column_config.NumberColumn(
                "Estabelecimentos comerciais", format="localized"
            ),
            "pontos_coleta": st.column_config.NumberColumn(
                "Pontos de coleta", format="localized"
            ),
            ALVO: st.column_config.NumberColumn(
                "Recicláveis (ton/mês)", format="%.2f"
            ),
        },
        alt="Tabela com os bairros sintéticos, população, domicílios, estabelecimentos comerciais, pontos de coleta e recicláveis mensais",
    )

    pontos_base = (
        alt.Chart(df)
        .mark_circle(size=90)
        .encode(
            x=alt.X("populacao_bairro:Q", title="População do bairro"),
            y=alt.Y(f"{ALVO}:Q", title="Recicláveis (ton/mês)"),
            tooltip=[
                alt.Tooltip("bairro:N", title="Bairro"),
                alt.Tooltip("populacao_bairro:Q", title="População", format=","),
                alt.Tooltip("domicilios:Q", title="Domicílios", format=","),
                alt.Tooltip(
                    "estabelecimentos_comerciais:Q",
                    title="Estabelecimentos",
                    format=",",
                ),
                alt.Tooltip("pontos_coleta:Q", title="Pontos de coleta", format=","),
                alt.Tooltip(f"{ALVO}:Q", title="Recicláveis (ton/mês)", format=".2f"),
            ],
        )
    )
    linha_tendencia = (
        alt.Chart(df)
        .transform_regression("populacao_bairro", ALVO)
        .mark_line(strokeDash=[5, 4], opacity=0.7)
        .encode(x="populacao_bairro:Q", y=f"{ALVO}:Q")
    )
    grafico_dados = (pontos_base + linha_tendencia).properties(
        title="Relação entre população do bairro e resíduos recicláveis",
        height=360,
    )
    st.altair_chart(
        grafico_dados,
        alt="Gráfico de dispersão mostrando crescimento quase linear de resíduos recicláveis em função da população do bairro",
    )

# ------------------------------------------------------------------------ Simulador
with aba_sim:
    st.subheader("Simulador de bairro", icon=":material/calculate:")

    modelo_selecionado = st.segmented_control(
        "Modelo para simulação",
        options=modelos_disponiveis,
        required=True,
        key="sim_modelo",
        help="Por padrão usa o modelo de menor erro na validação Leave-One-Out.",
    )
    if modelo_selecionado is None:
        modelo_selecionado = melhor_modelo

    info_modelo = pacote["modelos"][modelo_selecionado]
    features_ativas = info_modelo["features"]
    mae_ativo = float(
        pacote["validacao"].set_index("modelo").loc[modelo_selecionado, "mae_ton"]
    )
    badge_vencedor = " :green-badge[Menor erro LOO]" if modelo_selecionado == melhor_modelo else ""
    st.markdown(
        f"Modelo ativo: **{modelo_selecionado}**{badge_vencedor} · "
        f"MAE na validação: **{br(mae_ativo, 3)} ton/mês**"
    )

    extrapolar = st.toggle(
        "Permitir população fora da faixa treinada (extrapolação)",
        key="sim_extrapolar",
        help=f"O modelo só viu bairros entre {br(pop_min, 0)} e {br(pop_max, 0)} habitantes.",
    )
    limite_inf, limite_sup = (0.0, 500_000.0) if extrapolar else (pop_min, pop_max)

    # Garante que o valor em session_state fique dentro dos limites antes de criar o widget
    if st.session_state["sim_pop"] < limite_inf:
        st.session_state["sim_pop"] = limite_inf
    elif st.session_state["sim_pop"] > limite_sup:
        st.session_state["sim_pop"] = limite_sup

    usa_extras = len(features_ativas) > 1

    with st.form("simulador"):
        if usa_extras:
            c1, c2 = st.columns(2)
            populacao = c1.number_input(
                "População do bairro",
                min_value=limite_inf,
                max_value=limite_sup,
                step=100.0,
                format="%.0f",
                key="sim_pop",
            )
            domicilios = c2.number_input(
                "Domicílios",
                min_value=1.0,
                step=10.0,
                format="%.0f",
                key="sim_dom",
            )
            estabelecimentos = c1.number_input(
                "Estabelecimentos comerciais",
                min_value=0.0,
                step=1.0,
                format="%.0f",
                key="sim_estab",
            )
            pontos = c2.number_input(
                "Pontos de coleta",
                min_value=1.0,
                step=1.0,
                format="%.0f",
                key="sim_pontos",
            )
        else:
            populacao = st.number_input(
                "População do bairro",
                min_value=limite_inf,
                max_value=limite_sup,
                step=100.0,
                format="%.0f",
                key="sim_pop",
            )
            domicilios = float(st.session_state["sim_dom"])
            estabelecimentos = float(st.session_state["sim_estab"])
            pontos = float(st.session_state["sim_pontos"])
            st.caption(
                "Este modelo usa apenas a população do bairro. "
                "Domicílios, estabelecimentos e pontos de coleta não reduziram o erro na validação "
                "(selecione um modelo com 4 features acima para simulá-los)."
            )

        enviar = st.form_submit_button(
            "Prever",
            type="primary",
            icon=":material/play_arrow:",
            key="sim_submit",
        )

    if enviar:
        st.session_state["ultima_simulacao"] = {
            "modelo": modelo_selecionado,
            "entrada": {
                "populacao_bairro": float(populacao),
                "domicilios": float(domicilios),
                "estabelecimentos_comerciais": float(estabelecimentos),
                "pontos_coleta": float(pontos),
            },
        }

    simulacao = st.session_state["ultima_simulacao"]
    if simulacao is not None:
        nome_sim = simulacao["modelo"]
        entrada_sim = simulacao["entrada"]
        info_sim = pacote["modelos"][nome_sim]

        fora = features_fora_da_faixa(pacote, entrada_sim, nome_modelo=nome_sim)
        if fora:
            detalhes = "; ".join(
                f"{ROTULOS_FEATURES[col][0]} = {br(val, ROTULOS_FEATURES[col][1])} "
                f"(treino: {br(vmin, ROTULOS_FEATURES[col][1])} a {br(vmax, ROTULOS_FEATURES[col][1])})"
                for col, val, vmin, vmax in fora
            )
            if info_sim["extrapola"]:
                explicacao = (
                    "A regressão linear prolonga a tendência observada, mas não há "
                    "dados que confirmem esse comportamento — trate como estimativa."
                )
            else:
                explicacao = (
                    "O RandomForest não extrapola: a previsão tende a ficar travada "
                    "perto dos extremos vistos no treino."
                )
            st.warning(
                f"Valores fora da faixa treinada: {detalhes}. {explicacao}",
                title="Extrapolação fora da faixa treinada",
                icon=":material/warning:",
            )

        ton = prever(pacote, entrada_sim, nome_modelo=nome_sim)
        fator = float(st.session_state["fator_aterro"])
        mae_sim = float(
            pacote["validacao"].set_index("modelo").loc[nome_sim, "mae_ton"]
        )
        faixa_inf = max(0.0, ton - mae_sim)
        faixa_sup = ton + mae_sim

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Recicláveis previstos",
            f"{br(ton, 2)} ton/mês",
            border=True,
            height="stretch",
            icon=":material/recycling:",
            help=f"Erro médio absoluto na validação Leave-One-Out: ±{br(mae_sim, 2)} ton/mês.",
        )
        m2.metric(
            "Aterro evitado (premissa)",
            f"{br(ton * fator, 2)} ton/mês",
            border=True,
            height="stretch",
            icon=":material/eco:",
            help=f"Previsão × {br(fator, 2)} ({br(fator * 100, 0)}%). Fator ajustável na aba Premissas.",
        )
        m3.metric(
            "Intervalo esperado (±MAE)",
            f"{br(faixa_inf, 2)} a {br(faixa_sup, 2)} ton/mês",
            border=True,
            height="stretch",
            icon=":material/straighten:",
            help="Intervalo indicativo centrado na previsão usando o MAE fora da amostra.",
        )
        st.caption(
            f"Simulação com **{nome_sim}** para população de "
            f"{br(entrada_sim['populacao_bairro'], 0)} habitantes · "
            f"fator de aterro evitado: {br(fator, 2)}."
        )

        ponto_sim = pd.DataFrame(
            [
                {
                    "bairro": "Bairro simulado",
                    "populacao_bairro": entrada_sim["populacao_bairro"],
                    ALVO: ton,
                    "origem": "Simulação",
                }
            ]
        )
        base_comp = df[["bairro", "populacao_bairro", ALVO]].assign(
            origem="Base de treino"
        )
        dados_comp = pd.concat([base_comp, ponto_sim], ignore_index=True)

        grafico_sim = (
            alt.Chart(dados_comp)
            .mark_circle()
            .encode(
                x=alt.X("populacao_bairro:Q", title="População do bairro"),
                y=alt.Y(f"{ALVO}:Q", title="Recicláveis (ton/mês)"),
                color=alt.Color(
                    "origem:N",
                    title="Origem",
                    scale=alt.Scale(domain=["Base de treino", "Simulação"]),
                ),
                size=alt.condition(
                    alt.datum.origem == "Simulação", alt.value(180), alt.value(70)
                ),
                tooltip=[
                    alt.Tooltip("bairro:N", title="Bairro"),
                    alt.Tooltip("origem:N", title="Origem"),
                    alt.Tooltip("populacao_bairro:Q", title="População", format=","),
                    alt.Tooltip(f"{ALVO}:Q", title="Recicláveis (ton/mês)", format=".2f"),
                ],
            )
            .properties(
                title="Posição do bairro simulado em relação à base de treino",
                height=300,
            )
        )
        st.altair_chart(
            grafico_sim,
            alt="Comparação visual entre o bairro simulado e os bairros da base de treino por população e recicláveis",
        )

# ------------------------------------------------------------------------ Validação
with aba_val:
    st.subheader("Validação Leave-One-Out", icon=":material/verified:")
    n_amostras = pacote["n_amostras"]
    n_teste_75_25 = max(1, round(n_amostras * 0.25))
    st.markdown(
        f"Com apenas {n_amostras} amostras, uma divisão treino/teste única (75/25) "
        f"deixaria cerca de {n_teste_75_25} bairros no teste e o resultado dependeria da semente sorteada. "
        f"Por isso cada bairro é previsto por um modelo treinado nos outros {n_amostras - 1}, e todos os "
        "candidatos são comparados nas mesmas condições."
    )

    tabela = pacote["validacao"].copy()
    st.dataframe(
        tabela,
        hide_index=True,
        column_config={
            "modelo": st.column_config.TextColumn("Modelo"),
            "features": st.column_config.TextColumn("Features"),
            "mae_ton": st.column_config.NumberColumn("MAE (ton)", format="%.3f"),
            "r2": st.column_config.NumberColumn("R²", format="%.3f"),
        },
        alt="Comparação de MAE e R² dos modelos candidatos na validação Leave-One-Out",
    )

    linha_vencedor = tabela.set_index("modelo").loc[melhor_modelo]
    mae_vencedor = float(linha_vencedor["mae_ton"])
    r2_vencedor = float(linha_vencedor["r2"])
    if pacote["extrapola"]:
        motivo = (
            "Nesta base quase linear, a regressão linear supera o RandomForest, "
            "que aproxima a tendência por degraus."
        )
    else:
        motivo = (
            "O modelo baseado em árvores obteve o menor erro absoluto médio "
            "na validação fora da amostra."
        )
    st.success(
        f"MAE de **{br(mae_vencedor, 3)} ton/mês** e R² de **{br(r2_vencedor, 3)}**. {motivo}",
        title=f"Modelo escolhido: {melhor_modelo}",
        icon=":material/check_circle:",
    )

    previsoes_loo = pacote["previsoes_loo"]
    colunas_modelos = [
        c for c in previsoes_loo.columns if c not in {"bairro", "populacao_bairro", "real_ton"}
    ]
    loo_longo = previsoes_loo.melt(
        id_vars=["bairro", "real_ton"],
        value_vars=colunas_modelos,
        var_name="Modelo",
        value_name="Previsto",
    )
    max_eixo = float(max(loo_longo["real_ton"].max(), loo_longo["Previsto"].max())) * 1.05
    linha_ideal = (
        alt.Chart(pd.DataFrame({"x": [0.0, max_eixo], "y": [0.0, max_eixo]}))
        .mark_line(strokeDash=[4, 4], color="#94A3B8")
        .encode(x="x:Q", y="y:Q")
    )
    pontos_loo = (
        alt.Chart(loo_longo)
        .mark_circle(size=80)
        .encode(
            x=alt.X("real_ton:Q", title="Valor real da base (ton/mês)"),
            y=alt.Y("Previsto:Q", title="Previsão Leave-One-Out (ton/mês)"),
            color=alt.Color("Modelo:N", title="Modelo"),
            tooltip=[
                alt.Tooltip("bairro:N", title="Bairro"),
                alt.Tooltip("Modelo:N", title="Modelo"),
                alt.Tooltip("real_ton:Q", title="Real (ton/mês)", format=".2f"),
                alt.Tooltip("Previsto:Q", title="Previsto (ton/mês)", format=".2f"),
            ],
        )
    )
    grafico_val = (linha_ideal + pontos_loo).properties(
        title="Real vs. previsto fora da amostra (Leave-One-Out)",
        height=360,
    )
    st.altair_chart(
        grafico_val,
        alt="Gráfico comparando valores reais e previsões Leave-One-Out de cada modelo em relação à linha ideal",
    )

    corr_dom = float(df[["populacao_bairro", "domicilios"]].corr().iloc[0, 1])
    st.caption(
        f"`domicilios` não entra diretamente como feature: tem correlação {br(corr_dom, 4)} com a população "
        "e tornaria a importância das variáveis arbitrária. Ela é usada apenas como "
        "`pessoas_por_domicilio`."
    )

# ------------------------------------------------------------------------ Premissas
with aba_prem:
    st.subheader("Premissas do painel", icon=":material/tune:")
    st.slider(
        "Fração dos recicláveis que deixa de ir ao aterro",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="fator_aterro",
        help="Atualiza automaticamente o indicador de aterro evitado no simulador.",
    )
    st.warning(
        f"O valor padrão {br(FATOR_ATERRO_EVITADO_PADRAO, 2)} é uma premissa sem fonte "
        "citada. Substitua por um dado com referência (ex.: SNIS ou relatório municipal) "
        "antes de apresentar o indicador de aterro evitado como resultado.",
        title="Premissa sem fonte citada",
        icon=":material/warning:",
    )

    st.table(
        {
            ":material/science: Tipo da base": f"Sintética ({pacote['n_amostras']} bairros fictícios)",
            ":material/groups: Faixa de população treinada": f"{br(pop_min, 0)} a {br(pop_max, 0)} habitantes",
            ":material/verified: Modelo padrão": melhor_modelo,
            ":material/tune: Fator de aterro evitado atual": f"{br(float(st.session_state['fator_aterro']), 2)} ({br(float(st.session_state['fator_aterro']) * 100, 0)}%)",
            ":material/memory: scikit-learn": pacote["versao_sklearn"],
        },
        border="horizontal",
        width="content",
        alt="Resumo técnico das premissas, faixa de treino e versão do scikit-learn",
    )
