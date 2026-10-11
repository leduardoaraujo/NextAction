"""Empacota um classificador treinado para publicação no Hugging Face Hub."""

import json
import shutil
import zipfile
from pathlib import Path

from .avaliacao import avaliar, metricas
from .dados import carregar_exemplos
from .modelo import enxugar, novo_modelo


ROOT = Path(__file__).resolve().parent.parent
EXTERNOS = ROOT / "dados" / "externos"
SGD = (
    "[Schema-Guided Dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)"
    " (Rastogi et al., 2020, CC BY-SA 4.0)"
)
CENARIOS = {
    "pesquisa": {
        "dados": ROOT / "dados" / "exemplos.csv",
        "idioma": "pt",
        "decisao": "qual deve ser a próxima ação de um agente de pesquisa",
        "exemplo": "Preciso saber a cotação de hoje e ainda não consultei nenhuma fonte",
        "origem_dados": "exemplos **sintéticos**, escritos à mão; não vêm de usuários reais",
    },
    "passagens": {
        "dados": ROOT / "dados" / "passagens.csv",
        "idioma": "pt",
        "decisao": "qual deve ser a próxima etapa de um agente que ajuda a encontrar passagens aéreas",
        "exemplo": "Tenho origem GRU destino REC ida 12/11 e ainda não consultei voos",
        "origem_dados": "exemplos **sintéticos**, escritos à mão; não vêm de usuários reais",
    },
    "agente": {
        "dados": EXTERNOS / "sgd_generico_train.csv",
        "teste": EXTERNOS / "sgd_generico_test.csv",
        "idioma": "en",
        "licenca": "cc-by-sa-4.0",
        "min_df": 5,
        "copiar_dados": False,
        "decisao": "o que um assistente deve fazer em seguida numa conversa orientada a tarefas",
        "exemplo": (
            "[intencao SearchOnewayFlight] [preenchidos destination_airport origin_airport] "
            "[faltando departure_date] [pedidos nenhum] || "
            "usuario: I need a one way flight from New York to Seattle."
        ),
        "origem_dados": (
            f"turnos do assistente do {SGD}: conversas entre pessoas e um assistente virtual em "
            "20 domínios, escritas por colaboradores a partir de roteiros simulados. Gere o CSV "
            "com `python -m opendecision importar-sgd`"
        ),
    },
}
ARQUIVO_MODELO = "model.skops"
ARQUIVO_DADOS = "dados_treino.csv"


def exportar(cenario: str, destino: Path, repo: str | None = None, licenca: str | None = None) -> dict:
    """Treina o cenário, mede o desempenho e grava modelo skops e model card em ``destino``."""
    # Importado aqui para que skops só seja exigido por quem exporta.
    import skops.io as sio

    if cenario not in CENARIOS:
        raise ValueError(f"Cenário desconhecido: {cenario}. Use {', '.join(CENARIOS)}.")
    config = CENARIOS[cenario]
    if not config["dados"].is_file():
        raise FileNotFoundError(f"{config['dados']} não existe; gere-o antes (veja o readme).")
    modelo, avaliacao = treinar_e_avaliar(config)

    destino.mkdir(parents=True, exist_ok=True)
    sio.dump(modelo, destino / ARQUIVO_MODELO, compression=zipfile.ZIP_DEFLATED, compresslevel=9)
    if sio.get_untrusted_types(file=destino / ARQUIVO_MODELO):
        raise ValueError("O arquivo skops exige tipos não confiáveis; revise antes de publicar.")
    if config.get("copiar_dados", True):
        shutil.copyfile(config["dados"], destino / ARQUIVO_DADOS)
    card = model_card(
        cenario, list(modelo.classes_), avaliacao,
        repo or f"<usuario>/opendecision-{cenario}", licenca or config.get("licenca"),
    )
    (destino / "README.md").write_text(card, encoding="utf-8")
    return {
        "pasta": str(destino), "arquivos": sorted(p.name for p in destino.iterdir()),
        "metodo": avaliacao["metodo"], "acuracia": avaliacao["acuracia"],
    }


def treinar_e_avaliar(config: dict):
    estados, acoes = carregar_exemplos(config["dados"])
    min_df = config.get("min_df", 1)
    modelo = enxugar(novo_modelo(min_df).fit(estados, acoes))
    if "teste" not in config:
        return modelo, {**avaliar(config["dados"], min_df=min_df), "exemplos_de_treino": len(estados)}
    # Com teste separado, o modelo publicado é o próprio modelo avaliado.
    teste_x, teste_y = carregar_exemplos(config["teste"])
    majoritaria = max(set(acoes), key=acoes.count)
    resultado = metricas(teste_y, modelo.predict_proba(teste_x), list(modelo.classes_), majoritaria)
    metodo = f"split de teste separado ({config['teste'].name})"
    return modelo, {"metodo": metodo, "exemplos_de_treino": len(estados), **resultado}


def model_card(cenario: str, acoes: list[str], avaliacao: dict, repo: str, licenca: str | None) -> str:
    config = CENARIOS[cenario]
    metadados = [f"language: {config['idioma']}", "library_name: sklearn", "pipeline_tag: text-classification"]
    if licenca:
        metadados.append(f"license: {licenca}")
    metadados += ["tags:", "- sklearn", "- skops", "- text-classification", "- agents"]
    lista_acoes = "\n".join(f"- `{acao}`" for acao in acoes)
    exemplo = json.dumps(config["exemplo"], ensure_ascii=False)
    idioma = "português" if config["idioma"] == "pt" else "inglês"
    por_acao = "\n".join(
        f"| `{acao}` | {m['precisao']} | {m['revocacao']} | {m['f1']} | {m['suporte']} |"
        for acao, m in avaliacao["por_acao"].items()
    )
    limiares = "\n".join(
        f"| {l['limiar']} | {l['automatizadas']:.1%} | "
        f"{'-' if l['acuracia_automatizadas'] is None else format(l['acuracia_automatizadas'], '.1%')} |"
        for l in avaliacao["limiares"]
    )
    dados_publicados = (
        f"O arquivo `{ARQUIVO_DADOS}` acompanha o modelo." if config.get("copiar_dados", True)
        else "Os dados de treino não são redistribuídos aqui; use a fonte original."
    )
    aviso = f"\n\n> {avaliacao['aviso']}" if "aviso" in avaliacao else ""
    return f"""---
{chr(10).join(metadados)}
---

# OpenDecision: {cenario}

Classificador de texto que responde **{config["decisao"]}**.

Entrada: uma descrição do estado atual em {idioma}. Saída: uma das ações abaixo, com probabilidades estimadas.

{lista_acoes}

O modelo apenas **sugere** a ação. O aplicativo que a executa deve validar as condições
necessárias e pedir revisão humana quando a confiança for baixa.

## Como usar

```python
from huggingface_hub import hf_hub_download
import skops.io as sio

caminho = hf_hub_download("{repo}", "{ARQUIVO_MODELO}")
modelo = sio.load(caminho)  # o arquivo não exige tipos fora da lista confiável do skops

estado = {exemplo}
probabilidades = dict(zip(modelo.classes_, modelo.predict_proba([estado])[0]))
print(max(probabilidades, key=probabilidades.get), probabilidades)
```

Dependências: `scikit-learn`, `skops` e `huggingface_hub`. Use uma versão de scikit-learn
compatível com a de treino para evitar avisos de incompatibilidade.

## Arquitetura

Pipeline scikit-learn: `TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", min_df={config.get("min_df", 1)})` seguido de
`LogisticRegression(max_iter=1000)`. Código de treino, avaliação e API: [OpenDecision](https://github.com/leduardoaraujo/OpenDecision).

## Dados de treino

{avaliacao["exemplos_de_treino"]} {config["origem_dados"]}. {dados_publicados}

## Avaliação

Método: {avaliacao["metodo"]}, com {avaliacao["exemplos_avaliados"]} exemplos avaliados.
Acurácia **{avaliacao["acuracia"]}**; escolher sempre a ação mais comum daria {avaliacao["acuracia_classe_majoritaria"]}.
Erro de calibração esperado: {avaliacao["erro_calibracao_esperado"]}.{aviso}

| Ação | Precisão | Revocação | F1 | Exemplos |
| --- | --- | --- | --- | --- |
{por_acao}

Limiar de revisão: decisões com confiança abaixo do limiar vão para uma pessoa.

| Limiar | Decisões automáticas | Acurácia nas automáticas |
| --- | --- | --- |
{limiares}

## Limitações

- O modelo reconhece padrões de palavras; não entende regras de negócio nem verifica fatos.
- As probabilidades da regressão logística não foram calibradas.
- Idioma único ({idioma}); o conjunto de ações é fixo.
- Não use a previsão para executar ações irreversíveis (por exemplo, compras) sem confirmação explícita.
"""
