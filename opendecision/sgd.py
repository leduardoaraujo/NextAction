"""Converte o Schema-Guided Dialogue (Google, CC BY-SA 4.0) em exemplos ``estado,acao``.

Cada turno do assistente vira um exemplo: o estado é a conversa até ali, mais o estado do
diálogo (intenção, campos preenchidos e obrigatórios que faltam), e a ação é o que o
assistente fez. Os atos do SGD são agrupados em ações genéricas de um agente.
"""

import csv
import json
from collections import Counter
from pathlib import Path


# Ordem de prioridade quando um turno tem vários atos.
ACOES_GENERICAS = {
    "NOTIFY_SUCCESS": "executar",
    "NOTIFY_FAILURE": "executar",
    "CONFIRM": "confirmar",
    "REQUEST": "perguntar",
    "OFFER_INTENT": "sugerir_proxima_etapa",
    "OFFER": "apresentar_opcao",
    "INFORM_COUNT": "apresentar_opcao",
    "INFORM": "responder",
    "REQ_MORE": "oferecer_mais_ajuda",
    "GOODBYE": "encerrar",
}
PARA_PASSAGENS = {
    "perguntar": "perguntar_dados",
    "buscar": "buscar_voos",
    "apresentar_opcao": "apresentar_opcoes",
    "confirmar": "pedir_confirmacao",
}
ESQUEMAS = ("generico", "passagens")


def acao_do_turno(turno: dict) -> str | None:
    atos = {acao["act"] for frame in turno["frames"] for acao in frame["actions"]}
    busca = any(
        "service_call" in frame and not atos & {"NOTIFY_SUCCESS", "NOTIFY_FAILURE"}
        for frame in turno["frames"]
    )
    if busca:
        # O assistente consultou o serviço antes de responder: a decisão foi buscar.
        return "buscar"
    return next((acao for ato, acao in ACOES_GENERICAS.items() if ato in atos), None)


def descrever_estado(historico: list[dict], obrigatorios: dict, contexto: int, com_estado: bool) -> str:
    partes = []
    ultimo_usuario = next((t for t in reversed(historico) if t["speaker"] == "USER"), None)
    if com_estado and ultimo_usuario:
        for frame in ultimo_usuario["frames"]:
            estado = frame.get("state")
            if not estado or estado["active_intent"] == "NONE":
                continue
            intencao = estado["active_intent"]
            preenchidos = sorted(estado["slot_values"])
            faltando = [c for c in obrigatorios.get((frame["service"], intencao), []) if c not in preenchidos]
            partes.append(
                f"[intencao {intencao}] [preenchidos {' '.join(preenchidos) or 'nenhum'}] "
                f"[faltando {' '.join(faltando) or 'nenhum'}] "
                f"[pedidos {' '.join(estado['requested_slots']) or 'nenhum'}]"
            )
    for turno in historico[-contexto:]:
        quem = "usuario" if turno["speaker"] == "USER" else "assistente"
        partes.append(f"{quem}: {turno['utterance']}")
    return " || ".join(partes)


def converter(
    pasta_split: Path, destino: Path, esquema: str = "generico",
    contexto: int = 3, com_estado: bool = True,
) -> dict:
    if esquema not in ESQUEMAS:
        raise ValueError(f"Esquema desconhecido: {esquema}. Use {', '.join(ESQUEMAS)}.")
    arquivos = sorted(pasta_split.glob("dialogues_*.json"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum dialogues_*.json em {pasta_split}.")
    esquema_servicos = json.loads((pasta_split / "schema.json").read_text(encoding="utf-8"))
    obrigatorios = {
        (servico["service_name"], intencao["name"]): intencao["required_slots"]
        for servico in esquema_servicos for intencao in servico["intents"]
    }

    contagem = Counter()
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="") as saida:
        escritor = csv.writer(saida)
        escritor.writerow(["estado", "acao", "dialogo"])
        for arquivo in arquivos:
            for dialogo in json.loads(arquivo.read_text(encoding="utf-8")):
                if esquema == "passagens" and not any(s.startswith("Flights") for s in dialogo["services"]):
                    continue
                turnos = dialogo["turns"]
                for indice, turno in enumerate(turnos):
                    if turno["speaker"] != "SYSTEM" or indice == 0:
                        continue
                    acao = acao_do_turno(turno)
                    if esquema == "passagens":
                        voos = all(f["service"].startswith("Flights") for f in turno["frames"])
                        acao = PARA_PASSAGENS.get(acao) if voos else None
                    if acao is None:
                        continue
                    estado = descrever_estado(turnos[:indice], obrigatorios, contexto, com_estado)
                    escritor.writerow([estado, acao, dialogo["dialogue_id"]])
                    contagem[acao] += 1
    return {"arquivo": str(destino), "exemplos": sum(contagem.values()), "por_acao": dict(contagem.most_common())}
