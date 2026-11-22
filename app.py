"""Painel do RotaVerde MS, feito em Streamlit.

É aqui que o modelo vira algo que a prefeitura ou a cooperativa consegue usar. O
painel tem quatro abas:

  - Ranking: ordena os bairros pelo volume previsto, para ajudar a decidir por onde
    a coleta começa;
  - Simulador: estima o volume de um bairro novo a partir dos dados que a pessoa
    digitar;
  - Validação: mostra como cada modelo se saiu e por que escolhemos o atual;
  - Premissas: deixa claro o que é suposição nossa, e não dado medido.

Para abrir:
    streamlit run app.py
"""
import joblib
import pandas as pd
import sklearn
import streamlit as st

from modelo import ARQUIVO_MODELO, prever, treinar

# Quanto do reciclável coletado deixa de ir para o aterro. O 0,85 é uma suposição
# nossa e ainda não tem fonte. Antes de apresentar o "aterro evitado" como
# resultado, vale trocar por um número com referência. A pessoa também pode ajustar
# esse valor na aba Premissas.
FATOR_ATERRO_PADRAO = 0.85

# Nomes amigáveis das colunas, usados nos avisos que aparecem para o usuário.
ROTULOS = {
    "populacao_bairro": "população",
    "pessoas_por_domicilio": "pessoas por domicílio",
    "comercio_local": "estabelecimentos comerciais",
    "taxa_adesao_historica": "taxa de adesão",
}

st.set_page_config(page_title="RotaVerde MS — Previsão de Coleta Seletiva", layout="wide")


def br(valor: float, casas: int = 0) -> str:
    """Escreve o número do jeito brasileiro: 12.345,67 em vez de 12,345.67."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


@st.cache_resource
def carregar_modelo() -> dict:
    """Abre o modelo salvo ou, se não der, treina um novo na hora.

    O treino acontece quando o arquivo não existe, não abre, está no formato da
    primeira versão ou foi salvo com outro scikit-learn. Como a base é pequena, isso
    leva poucos segundos, e o painel nunca fica parado por causa do arquivo.
    O @st.cache_resource faz isso uma vez só, e não a cada clique.
    """
    try:
        pacote = joblib.load(ARQUIVO_MODELO)
        if isinstance(pacote, dict) and pacote.get("versao_sklearn") == sklearn.__version__:
            return pacote
    except Exception:
        pass
    return treinar()


# Informações do modelo que várias abas usam.
pacote = carregar_modelo()
faixa = pacote["faixa_treino"]
pop_min, pop_max = faixa["populacao_bairro"]
validacao = pd.DataFrame(pacote["validacao"]).set_index("modelo")
mae = float(validacao.loc[pacote["nome_modelo"], "mae_ton"])

# O fator de aterro vem do controle da aba Premissas. Na primeira abertura, ele
# ainda não existe, então usamos o valor padrão.
fator = st.session_state.get("fator_aterro", FATOR_ATERRO_PADRAO)

st.title("♻️ RotaVerde MS — Painel Preditivo de Coleta Seletiva")
st.caption("UFMS Câmpus Coxim | Tecnologia Social e Inteligência Artificial aplicada à "
           "Gestão de Resíduos")
st.info(
    f"Protótipo treinado em uma **base sintética de {pacote['n_amostras']} bairros**. "
    f"Modelo em uso: **{pacote['nome_modelo']}**, erro médio de ±{br(mae, 2)} ton/mês "
    "na validação Leave-One-Out."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Ranking de Prioridade de Bairros",
    "🔮 Previsão Individual de Região",
    "📊 Validação do Modelo",
    "⚙️ Premissas",
])

# ---------------------------------------------------------------------------
# Aba 1: ranking dos bairros de Coxim pelo volume previsto
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Planejamento Semanal para Cooperativas e Prefeituras")
    st.write("Ranking dos bairros com maior volume previsto para otimizar as rotas de "
             "coleta e evitar descarte em aterros.")

    # Números ilustrativos para demonstrar o ranking. Não são dados oficiais.
    bairros_padrao = pd.DataFrame({
        'Bairro': ['Centro', 'Vila Bela', 'Flávio Garcia', 'Senhor Divino', 'Silviolândia'],
        'populacao_bairro': [12000, 7500, 9200, 4800, 3100],
        'domicilios': [3800, 2300, 2900, 1500, 900],
        'comercio_local': [95, 30, 60, 22, 10],
        'taxa_adesao_historica': [0.65, 0.45, 0.58, 0.40, 0.32]
    })

    bairros_padrao['Reciclável Estimado (ton/mês)'] = prever(pacote, bairros_padrao).round(2)
    bairros_padrao['Aterro Evitado Estimado (ton/mês)'] = (
        bairros_padrao['Reciclável Estimado (ton/mês)'] * fator).round(2)

    # Quem gera mais reciclável aparece primeiro e recebe a prioridade 1.
    tabela_ranking = bairros_padrao.sort_values(by='Reciclável Estimado (ton/mês)',
                                                ascending=False)
    tabela_ranking.insert(0, 'Prioridade', range(1, len(tabela_ranking) + 1))
    st.dataframe(
        tabela_ranking[['Prioridade', 'Bairro', 'Reciclável Estimado (ton/mês)',
                        'Aterro Evitado Estimado (ton/mês)']],
        width="stretch", hide_index=True,
        column_config={
            'Reciclável Estimado (ton/mês)': st.column_config.NumberColumn(format="%.2f"),
            'Aterro Evitado Estimado (ton/mês)': st.column_config.NumberColumn(format="%.2f"),
        },
    )

    col1, col2 = st.columns(2)
    col1.metric("Volume Total Mensal Estimado",
                f"{br(bairros_padrao['Reciclável Estimado (ton/mês)'].sum(), 2)} ton/mês")
    col2.metric("Impacto Direto (Desvio de Aterro)",
                f"{br(bairros_padrao['Aterro Evitado Estimado (ton/mês)'].sum(), 2)} ton/mês",
                help=f"Volume estimado × {br(fator, 2)} (premissa, ver aba Premissas).")

    # Se algum bairro do ranking for menor ou maior que os bairros do treino,
    # avisamos: ali o modelo está adivinhando além do que já viu.
    fora_faixa = bairros_padrao.loc[
        ~bairros_padrao['populacao_bairro'].between(pop_min, pop_max), 'Bairro'].tolist()
    if fora_faixa:
        st.caption(f"⚠️ População fora da faixa treinada ({br(pop_min)} a {br(pop_max)}): "
                   f"{', '.join(fora_faixa)}. A estimativa desses bairros é extrapolação.")
    st.caption("Indicadores dos bairros são ilustrativos. O aterro evitado usa um fator "
               f"de {br(fator, 2)} que é premissa, não dado medido.")

# ---------------------------------------------------------------------------
# Aba 2: simulador para um bairro digitado pelo usuário
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Simulador para Novos Bairros / Pilotos Territoriais")

    # Por padrão, a população fica presa à faixa que o modelo conhece. Quem quiser
    # testar uma cidade maior precisa ligar esta opção de propósito, e aí recebe
    # um aviso junto com o resultado.
    extrapolar = st.toggle(
        "Permitir população fora da faixa treinada (extrapolação)", value=False,
        help=f"O modelo só viu bairros entre {br(pop_min)} e {br(pop_max)} habitantes.",
    )
    pop_lim = (0, 500_000) if extrapolar else (int(pop_min), int(pop_max))

    with st.form("form_previsao"):
        nome = st.text_input("Nome do Bairro / Zona", "Bairro Piloto")
        c1, c2 = st.columns(2)
        pop = c1.number_input("População estimada", min_value=pop_lim[0],
                              max_value=pop_lim[1], value=5000, step=100)
        domicilios = c2.number_input("Número de domicílios", min_value=1, value=1500,
                                     step=50)
        c3, c4 = st.columns(2)
        comercios = c3.number_input("Estabelecimentos comerciais", min_value=0, value=25,
                                    step=5)
        # A pessoa escolhe em porcentagem (0 a 100). Mais abaixo, dividimos por 100,
        # porque no CSV a taxa está entre 0 e 1.
        adesao_pct = c4.slider("Taxa de adesão histórica à coleta (%)", 0, 100, 50, step=5)

        btn = st.form_submit_button("Calcular Estimativa")

    # Sendo transparentes: se o modelo escolhido ignora algum campo, dizemos isso,
    # para ninguém achar que mexer nele muda o resultado.
    nao_usadas = [ROTULOS[f] for f in ROTULOS if f not in pacote["features"]]
    if nao_usadas:
        st.caption(f"O modelo escolhido na validação não usa {', '.join(nao_usadas)}: "
                   "esses campos não alteram a estimativa.")

    if btn:
        entrada = pd.DataFrame([[pop, domicilios, comercios, adesao_pct / 100]],
                               columns=['populacao_bairro', 'domicilios', 'comercio_local',
                                        'taxa_adesao_historica'])
        entrada["pessoas_por_domicilio"] = entrada["populacao_bairro"] / entrada["domicilios"]

        # Conferimos cada valor que o modelo usa contra a faixa do treino.
        fora = [ROTULOS[f] for f in pacote["features"]
                if not faixa[f][0] <= float(entrada[f].iloc[0]) <= faixa[f][1]]
        if fora:
            aviso = (f"Fora da faixa usada no treino: {', '.join(fora)}. ")
            if pacote["extrapola"]:
                aviso += ("A regressão linear prolonga a tendência observada, mas não há "
                          "dados que confirmem esse comportamento. Trate como estimativa.")
            else:
                aviso += ("O RandomForest não extrapola: o valor fica travado perto do "
                          "maior bairro do treino e sai subestimado.")
            st.warning(aviso)

        prev = float(prever(pacote, entrada).iloc[0])
        st.success(f"Para o **{nome}**, a estimativa mensal de coleta é de "
                   f"**{br(prev, 2)} toneladas** de recicláveis "
                   f"(±{br(mae, 2)} ton, erro médio na validação).")
        st.metric("Aterro evitado (premissa)", f"{br(prev * fator, 2)} ton/mês")

# ---------------------------------------------------------------------------
# Aba 3: como os modelos se saíram e por que ficamos com este
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Validação Leave-One-Out")
    st.write(
        f"Com {pacote['n_amostras']} bairros, uma divisão treino/teste única deixaria só 3 "
        "no teste, e o resultado dependeria da semente sorteada. Por isso cada bairro é "
        "previsto por um modelo treinado nos outros, e todos os modelos são comparados "
        "nas mesmas condições."
    )
    tabela = pd.DataFrame(pacote["validacao"]).rename(columns={
        "modelo": "Modelo", "features": "Features", "mae_ton": "MAE (ton)", "r2": "R²"})
    st.dataframe(tabela, width="stretch", hide_index=True,
                 column_config={"MAE (ton)": st.column_config.NumberColumn(format="%.3f"),
                                "R²": st.column_config.NumberColumn(format="%.3f")})
    st.success(
        f"Modelo escolhido: **{pacote['nome_modelo']}** (menor MAE). Testamos "
        "RandomForest e regressão linear: nesta base, quase linear, o modelo simples "
        "venceu. O RandomForest aproxima a tendência por degraus e não extrapola."
    )
    st.caption("`domicilios` tem correlação ~0,999 com a população, então entra apenas "
               "como pessoas por domicílio. Assim a contribuição das variáveis não fica "
               "dividida ao acaso entre duas colunas redundantes.")

# ---------------------------------------------------------------------------
# Aba 4: tudo o que é suposição nossa, em um lugar só
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Premissas do painel")
    # O key liga este controle ao st.session_state. É assim que as outras abas
    # ficam sabendo do valor escolhido.
    st.slider("Fração do reciclável coletado que deixa de ir ao aterro", 0.0, 1.0,
              FATOR_ATERRO_PADRAO, 0.01, key="fator_aterro")
    st.warning(f"O valor padrão {br(FATOR_ATERRO_PADRAO, 2)} é uma **premissa sem fonte "
               "citada**. Substitua por um dado com referência antes de apresentar o "
               "aterro evitado como resultado.")
    st.markdown(
        f"- Base sintética com {pacote['n_amostras']} bairros\n"
        f"- População treinada: {br(pop_min)} a {br(pop_max)} habitantes\n"
        f"- Volume treinado: {br(faixa['volume_reciclavel_ton'][0], 1)} a "
        f"{br(faixa['volume_reciclavel_ton'][1], 1)} ton/mês\n"
        f"- scikit-learn {pacote['versao_sklearn']}"
    )
