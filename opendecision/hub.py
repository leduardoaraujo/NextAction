"""Empacota um classificador treinado para publicação no Hugging Face Hub."""

import json
import shutil
from pathlib import Path

from .modelo import ajustar


ROOT = Path(__file__).resolve().parent.parent
CENARIOS = {
    "pesquisa": {
        "dados": ROOT / "dados" / "exemplos.csv",
        "decisao": "qual deve ser a próxima ação de um agente de pesquisa",
        "exemplo": "Preciso saber a cotação de hoje e ainda não consultei nenhuma fonte",
    },
    "passagens": {
        "dados": ROOT / "dados" / "passagens.csv",
        "decisao": "qual deve ser a próxima etapa de um agente que ajuda a encontrar passagens aéreas",
        "exemplo": "Tenho origem GRU destino REC ida 12/11 e ainda não consultei voos",
    },
}
ARQUIVO_MODELO = "model.skops"
ARQUIVO_DADOS = "dados_treino.csv"


def exportar(cenario: str, destino: Path, repo: str | None = None, licenca: str | None = None) -> dict:
    """Treina a partir do CSV do cenário e grava modelo skops, dados e model card em ``destino``."""
    # Importado aqui para que skops só seja exigido por quem exporta.
    import skops.io as sio

    if cenario not in CENARIOS:
        raise ValueError(f"Cenário desconhecido: {cenario}. Use {', '.join(CENARIOS)}.")
    config = CENARIOS[cenario]
    modelo, resumo = ajustar(config["dados"])

    destino.mkdir(parents=True, exist_ok=True)
    sio.dump(modelo, destino / ARQUIVO_MODELO)
    if sio.get_untrusted_types(file=destino / ARQUIVO_MODELO):
        raise ValueError("O arquivo skops exige tipos não confiáveis; revise antes de publicar.")
    shutil.copyfile(config["dados"], destino / ARQUIVO_DADOS)
    card = model_card(cenario, resumo, repo or f"<usuario>/opendecision-{cenario}", licenca)
    (destino / "README.md").write_text(card, encoding="utf-8")
    return {"pasta": str(destino), "arquivos": sorted(p.name for p in destino.iterdir()), **resumo}


def model_card(cenario: str, resumo: dict, repo: str, licenca: str | None) -> str:
    config = CENARIOS[cenario]
    metadados = ["language: pt", "library_name: sklearn", "pipeline_tag: text-classification"]
    if licenca:
        metadados.append(f"license: {licenca}")
    metadados += ["tags:", "- sklearn", "- skops", "- text-classification", "- agents"]
    acoes = "\n".join(f"- `{acao}`" for acao in resumo["acoes"])
    exemplo = json.dumps(config["exemplo"], ensure_ascii=False)
    return f"""---
{chr(10).join(metadados)}
---

# OpenDecision: {cenario}

Classificador de texto que responde **{config["decisao"]}**.

Entrada: uma descrição do estado atual em português. Saída: uma das ações abaixo, com probabilidades estimadas.

{acoes}

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

Pipeline scikit-learn: `TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")` seguido de
`LogisticRegression(max_iter=1000)`. Código de treino e API: projeto OpenDecision.

## Dados de treino

`{ARQUIVO_DADOS}` traz {resumo["exemplos"]} exemplos **sintéticos**, escritos à mão, com as colunas
`estado` e `acao`. Eles não vêm de usuários reais.

## Avaliação

Em uma divisão estratificada com {resumo["exemplos_de_teste"]} exemplos de teste (`random_state=42`),
a acurácia foi {resumo["acuracia_no_teste"]}. Com tão poucos casos, esse número **não** indica
desempenho real. O modelo publicado foi ajustado novamente em todos os exemplos.

## Limitações

- Base pequena e sintética: o modelo reconhece principalmente palavras parecidas com as dos exemplos.
- As probabilidades da regressão logística não foram calibradas.
- Somente português; o conjunto de ações é fixo.
- Não use a previsão para executar ações irreversíveis (por exemplo, compras) sem confirmação explícita.
"""
