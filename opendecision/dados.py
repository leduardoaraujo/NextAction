"""Leitura e validação dos exemplos rotulados."""

import csv
from collections import Counter
from pathlib import Path


def carregar_exemplos(caminho: Path) -> tuple[list[str], list[str]]:
    with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        if not leitor.fieldnames or not {"estado", "acao"}.issubset(leitor.fieldnames):
            raise ValueError("O CSV precisa das colunas 'estado' e 'acao'.")
        linhas = [(linha["estado"].strip(), linha["acao"].strip()) for linha in leitor]

    if not linhas or any(not estado or not acao for estado, acao in linhas):
        raise ValueError("Todos os exemplos precisam de estado e acao preenchidos.")
    estados, acoes = zip(*linhas)
    contagens = Counter(acoes)
    if len(contagens) < 2 or min(contagens.values()) < 4:
        raise ValueError("Use pelo menos duas ações e quatro exemplos por ação.")
    return list(estados), list(acoes)
