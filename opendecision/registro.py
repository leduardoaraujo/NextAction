"""Registro append-only de decisões e do feedback que vira dado de treino."""

import csv
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


_trava = threading.Lock()


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _anexar(arquivo: Path, registro: dict) -> None:
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    linha = json.dumps(registro, ensure_ascii=False)
    with _trava, arquivo.open("a", encoding="utf-8") as saida:
        saida.write(linha + "\n")


def ler(arquivo: Path) -> list[dict]:
    if not arquivo.is_file():
        return []
    with arquivo.open(encoding="utf-8") as entrada:
        return [json.loads(linha) for linha in entrada if linha.strip()]


def registrar_decisao(arquivo: Path, cenario: str, estado: str, decisao: dict) -> str:
    decisao_id = uuid.uuid4().hex
    _anexar(arquivo, {
        "tipo": "decisao", "id": decisao_id, "em": _agora(), "cenario": cenario,
        "estado": estado, "sugerida": decisao["proxima_acao"],
        "probabilidades": decisao["probabilidades_estimadas"],
        "pedir_revisao": decisao["pedir_revisao"],
    })
    return decisao_id


def registrar_feedback(
    arquivo: Path, decisao_id: str, acao_final: str,
    resultado: str | None = None, comentario: str | None = None,
) -> dict:
    """Anota a ação que de fato deveria ser tomada; o último feedback de cada decisão prevalece."""
    if not any(r["tipo"] == "decisao" and r["id"] == decisao_id for r in ler(arquivo)):
        raise KeyError(decisao_id)
    registro = {"tipo": "feedback", "id": decisao_id, "em": _agora(), "acao_final": acao_final}
    if resultado is not None:
        registro["resultado"] = resultado
    if comentario is not None:
        registro["comentario"] = comentario
    _anexar(arquivo, registro)
    return registro


def exportar_rotulos(arquivo: Path, destino: Path, cenario: str) -> dict:
    """Gera CSV ``estado,acao`` em ordem cronológica, só com decisões que receberam feedback."""
    registros = ler(arquivo)
    feedback = {r["id"]: r for r in registros if r["tipo"] == "feedback"}
    decisoes = [r for r in registros if r["tipo"] == "decisao" and r["cenario"] == cenario]
    rotuladas = [(d, feedback[d["id"]]) for d in decisoes if d["id"] in feedback]
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="") as saida:
        escritor = csv.writer(saida)
        escritor.writerow(["estado", "acao", "em", "sugerida"])
        for decisao, nota in rotuladas:
            escritor.writerow([decisao["estado"], nota["acao_final"], decisao["em"], decisao["sugerida"]])
    corrigidas = sum(d["sugerida"] != n["acao_final"] for d, n in rotuladas)
    return {
        "arquivo": str(destino), "decisoes": len(decisoes), "rotuladas": len(rotuladas),
        "sugestoes_corrigidas": corrigidas,
    }
