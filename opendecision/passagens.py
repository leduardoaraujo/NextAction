"""Avaliação estruturada de ofertas de passagens fictícias."""

from datetime import date
from math import isfinite


PESOS_PADRAO = {"preco": 0.5, "duracao": 0.3, "escalas": 0.2}


def _numero(valor, campo: str, *, minimo: float = 0, inteiro: bool = False) -> float:
    tipos = (int,) if inteiro else (int, float)
    if isinstance(valor, bool) or not isinstance(valor, tipos) or not isfinite(float(valor)):
        raise ValueError(f"'{campo}' precisa ser um número válido.")
    if valor < minimo:
        raise ValueError(f"'{campo}' precisa ser maior ou igual a {minimo}.")
    return float(valor)


def _data(valor, campo: str) -> date:
    if not isinstance(valor, str):
        raise ValueError(f"'{campo}' precisa estar no formato AAAA-MM-DD.")
    try:
        return date.fromisoformat(valor)
    except ValueError as erro:
        raise ValueError(f"'{campo}' precisa estar no formato AAAA-MM-DD.") from erro


def avaliar_passagens(cenario: dict) -> dict:
    """Escolhe a próxima etapa e ordena ofertas elegíveis por critérios explícitos.

    A pontuação é uma utilidade relativa, não uma probabilidade ou previsão de ML.
    """
    if not isinstance(cenario, dict) or not isinstance(cenario.get("pedido"), dict):
        raise ValueError("O JSON precisa conter um objeto 'pedido'.")
    pedido = cenario["pedido"]
    obrigatorios = ("origem", "destino", "data_ida", "passageiros")
    faltantes = [campo for campo in obrigatorios if pedido.get(campo) in (None, "")]
    if faltantes:
        return {"acao": "perguntar_dados", "campos_faltantes": faltantes}

    for campo in ("origem", "destino"):
        if not isinstance(pedido[campo], str) or not pedido[campo].strip():
            raise ValueError(f"'{campo}' precisa ser um texto não vazio.")
    if pedido["origem"].strip().upper() == pedido["destino"].strip().upper():
        raise ValueError("Origem e destino precisam ser diferentes.")
    data_ida = _data(pedido["data_ida"], "data_ida")
    if pedido.get("data_volta") is not None and _data(pedido["data_volta"], "data_volta") < data_ida:
        raise ValueError("'data_volta' não pode ser anterior a 'data_ida'.")
    _numero(pedido["passageiros"], "passageiros", minimo=1, inteiro=True)

    orcamento = pedido.get("orcamento_max")
    if orcamento is not None:
        _numero(orcamento, "orcamento_max", minimo=0.01)
    max_escalas = pedido.get("max_escalas")
    if max_escalas is not None:
        _numero(max_escalas, "max_escalas", inteiro=True)
    bagagem = pedido.get("bagagem_obrigatoria", False)
    if not isinstance(bagagem, bool):
        raise ValueError("'bagagem_obrigatoria' precisa ser verdadeiro ou falso.")

    pesos = pedido.get("pesos", PESOS_PADRAO)
    if not isinstance(pesos, dict) or set(pesos) != set(PESOS_PADRAO):
        raise ValueError("'pesos' precisa conter preco, duracao e escalas.")
    pesos = {chave: _numero(pesos[chave], f"pesos.{chave}") for chave in PESOS_PADRAO}
    soma = sum(pesos.values())
    if soma == 0:
        raise ValueError("Pelo menos um peso precisa ser maior que zero.")
    pesos = {chave: valor / soma for chave, valor in pesos.items()}

    ofertas = cenario.get("ofertas", [])
    if not isinstance(ofertas, list):
        raise ValueError("'ofertas' precisa ser uma lista.")
    if not ofertas:
        return {"acao": "buscar_voos", "criterios": pedido}

    elegiveis, descartadas, ids = [], [], set()
    for oferta in ofertas:
        if not isinstance(oferta, dict) or not isinstance(oferta.get("id"), str) or not oferta["id"].strip():
            raise ValueError("Cada oferta precisa de um 'id' de texto não vazio.")
        if oferta["id"] in ids:
            raise ValueError(f"ID de oferta repetido: {oferta['id']}.")
        ids.add(oferta["id"])
        for campo in ("preco_total", "duracao_min"):
            _numero(oferta.get(campo), campo, minimo=0.01)
        _numero(oferta.get("escalas"), "escalas", inteiro=True)
        if not isinstance(oferta.get("bagagem_incluida"), bool):
            raise ValueError("'bagagem_incluida' precisa ser verdadeiro ou falso.")

        motivos = []
        if orcamento is not None and oferta["preco_total"] > orcamento:
            motivos.append("acima_do_orcamento")
        if max_escalas is not None and oferta["escalas"] > max_escalas:
            motivos.append("escalas_acima_do_limite")
        if bagagem and not oferta["bagagem_incluida"]:
            motivos.append("sem_bagagem_exigida")
        if motivos:
            descartadas.append({"id": oferta["id"], "motivos": motivos})
        else:
            elegiveis.append(oferta)

    if not elegiveis:
        return {"acao": "rever_criterios", "descartadas": descartadas}

    menor_preco = min(oferta["preco_total"] for oferta in elegiveis)
    menor_duracao = min(oferta["duracao_min"] for oferta in elegiveis)
    ranking = []
    for oferta in elegiveis:
        componentes = {
            "preco": menor_preco / oferta["preco_total"],
            "duracao": menor_duracao / oferta["duracao_min"],
            "escalas": 1 / (1 + oferta["escalas"]),
        }
        pontuacao = sum(pesos[chave] * valor for chave, valor in componentes.items())
        ranking.append((pontuacao, oferta, componentes))
    ranking.sort(key=lambda item: (-item[0], item[1]["preco_total"], item[1]["duracao_min"], item[1]["id"]))
    ordenadas = [
        {**oferta, "pontuacao": round(pontuacao, 4),
         "componentes": {chave: round(valor, 4) for chave, valor in componentes.items()}}
        for pontuacao, oferta, componentes in ranking
    ]

    escolhida = cenario.get("oferta_escolhida")
    if escolhida is not None:
        if escolhida not in ids:
            raise ValueError("'oferta_escolhida' não corresponde a uma oferta informada.")
        selecionada = next((oferta for oferta in ordenadas if oferta["id"] == escolhida), None)
        if selecionada is None:
            return {"acao": "rever_criterios", "oferta_escolhida": escolhida, "descartadas": descartadas}
        return {"acao": "pedir_confirmacao", "oferta_escolhida": selecionada,
                "aviso": "Confirme os detalhes e o preço atual antes de qualquer compra."}

    return {"acao": "apresentar_opcoes", "recomendacao": ordenadas[0]["id"],
            "ranking": ordenadas, "descartadas": descartadas,
            "pesos_aplicados": {chave: round(valor, 4) for chave, valor in pesos.items()}}
