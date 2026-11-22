"""Testes automatizados para gerar_dados.py, modelo.py e app.py."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import gerar_dados
import modelo


class TestGerarDados(unittest.TestCase):
    def test_reprodutibilidade_csv(self) -> None:
        df_csv = pd.read_csv(gerar_dados.ARQUIVO)
        df_gen = gerar_dados.gerar()
        pd.testing.assert_frame_equal(df_csv, df_gen)

    def test_valores_fisicamente_validos(self) -> None:
        df = gerar_dados.gerar()
        self.assertTrue((df["populacao_bairro"] > 0).all())
        self.assertTrue((df["domicilios"] > 0).all())
        self.assertTrue((df["estabelecimentos_comerciais"] >= 0).all())
        self.assertTrue((df["pontos_coleta"] >= 1).all())
        self.assertTrue((df[modelo.ALVO] >= 0).all())


class TestModelo(unittest.TestCase):
    def test_adicionar_features_rejeita_domicilios_zero(self) -> None:
        df = pd.DataFrame([{"populacao_bairro": 1000, "domicilios": 0}])
        with self.assertRaises(ValueError):
            modelo.adicionar_features(df)

    def test_treinar_e_prever_nao_negativo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            caminho_pkl = Path(tmp) / "modelo.pkl"
            pacote = modelo.treinar(caminho_modelo=caminho_pkl)
            self.assertTrue(caminho_pkl.exists())
            self.assertTrue(modelo.pacote_valido(pacote))

            pred_zero = modelo.prever(
                pacote,
                {
                    "populacao_bairro": 0.0,
                    "domicilios": 1.0,
                    "estabelecimentos_comerciais": 0.0,
                    "pontos_coleta": 1.0,
                },
            )
            self.assertGreaterEqual(pred_zero, 0.0)

            pred_rf = modelo.prever(
                pacote,
                {
                    "populacao_bairro": 10000.0,
                    "domicilios": 3100.0,
                    "estabelecimentos_comerciais": 95.0,
                    "pontos_coleta": 4.0,
                },
                nome_modelo="RandomForest",
            )
            self.assertGreater(pred_rf, 0.0)

    def test_carregar_ou_treinar_recupera_pickle_corrompido(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            caminho_pkl = Path(tmp) / "corrompido.pkl"
            caminho_pkl.write_bytes(b"dados-invalidos")
            pacote = modelo.carregar_ou_treinar(caminho_modelo=caminho_pkl)
            self.assertTrue(modelo.pacote_valido(pacote))

    def test_features_fora_da_faixa_detecta_extrapolacao(self) -> None:
        pacote = modelo.carregar_ou_treinar()
        fora_pop = modelo.features_fora_da_faixa(
            pacote, {"populacao_bairro": 100_000.0}
        )
        self.assertEqual(len(fora_pop), 1)
        self.assertEqual(fora_pop[0][0], "populacao_bairro")

        fora_4 = modelo.features_fora_da_faixa(
            pacote,
            {
                "populacao_bairro": 10_000.0,
                "domicilios": 3_125.0,
                "estabelecimentos_comerciais": 999.0,
                "pontos_coleta": 4.0,
            },
            nome_modelo="Regressão linear (4 features)",
        )
        nomes_fora = [item[0] for item in fora_4]
        self.assertIn("estabelecimentos_comerciais", nomes_fora)


class TestApp(unittest.TestCase):
    def test_fluxo_completo_simulador_e_premissas(self) -> None:
        at = AppTest.from_file("app.py", default_timeout=60).run()
        self.assertFalse(at.exception)

        # Submete simulação padrão (modelo vencedor: só população)
        at.number_input(key="sim_pop").set_value(12_000.0)
        at.form_submit_button[0].click().run()
        self.assertFalse(at.exception)
        self.assertIsNotNone(at.session_state["ultima_simulacao"])

        metricas_antes = {m.label: m.value for m in at.metric}
        self.assertIn("Recicláveis previstos", metricas_antes)
        self.assertIn("Aterro evitado (premissa)", metricas_antes)

        # Alterar fator de aterro em Premissas mantém resultado e atualiza métrica
        at.slider(key="fator_aterro").set_value(0.50).run()
        self.assertFalse(at.exception)
        metricas_depois = {m.label: m.value for m in at.metric}
        self.assertIn("Aterro evitado (premissa)", metricas_depois)
        self.assertNotEqual(
            metricas_antes["Aterro evitado (premissa)"],
            metricas_depois["Aterro evitado (premissa)"],
        )

    def test_extrapolacao_liga_e_desliga_sem_erro(self) -> None:
        at = AppTest.from_file("app.py", default_timeout=60).run()
        self.assertFalse(at.exception)

        # Liga extrapolação e testa valor muito alto
        at.toggle(key="sim_extrapolar").set_value(True).run()
        self.assertFalse(at.exception)
        at.number_input(key="sim_pop").set_value(100_000.0)
        at.form_submit_button[0].click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("fora da faixa" in w.value for w in at.warning))

        # Desliga extrapolação: valor em session_state deve ser ajustado sem crash
        at.toggle(key="sim_extrapolar").set_value(False).run()
        self.assertFalse(at.exception)

    def test_troca_para_modelo_4_features(self) -> None:
        at = AppTest.from_file("app.py", default_timeout=60).run()
        self.assertFalse(at.exception)

        at.segmented_control(key="sim_modelo").set_value("RandomForest").run()
        self.assertFalse(at.exception)
        at.number_input(key="sim_dom").set_value(3200.0)
        at.number_input(key="sim_estab").set_value(100.0)
        at.number_input(key="sim_pontos").set_value(5.0)
        at.form_submit_button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(
            at.session_state["ultima_simulacao"]["modelo"], "RandomForest"
        )


if __name__ == "__main__":
    unittest.main()
