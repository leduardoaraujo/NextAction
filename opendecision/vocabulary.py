"""Canonical action labels and state format, with Portuguese aliases."""

import re

ACTIONS = (
    "ask", "search", "present_options", "answer", "suggest_next_step",
    "confirm", "execute", "offer_more_help", "end_conversation", "revise_criteria",
)
ACTION_ALIASES = {
    "perguntar": "ask",
    "perguntar_dados": "ask",
    "buscar": "search",
    "buscar_voos": "search",
    "pesquisar": "search",
    "apresentar_opcao": "present_options",
    "apresentar_opcoes": "present_options",
    "responder": "answer",
    "sugerir_proxima_etapa": "suggest_next_step",
    "confirmar": "confirm",
    "pedir_confirmacao": "confirm",
    "executar": "execute",
    "oferecer_mais_ajuda": "offer_more_help",
    "encerrar": "end_conversation",
    "rever_criterios": "revise_criteria",
}

STATE_TAGS = {"intencao": "intent", "preenchidos": "filled", "faltando": "missing", "pedidos": "requested"}
SPEAKERS = {"usuario": "user", "usuário": "user", "assistente": "assistant"}

_TAG = re.compile(r"\[(intent|filled|missing|requested|intencao|preenchidos|faltando|pedidos)\b([^\]]*)\]")
_SPEAKER = re.compile(r"(^|\|\|\s*)(usuario|usuário|assistente):", re.IGNORECASE)


def normalize_action(action: str) -> str:
    """Return the canonical English label; unknown labels pass through unchanged."""
    action = action.strip()
    return ACTION_ALIASES.get(action, action)


def normalize_state(state: str) -> str:
    """Rewrite Portuguese state tags and speaker labels to the English format.

    ``[faltando data] || usuario: oi`` becomes ``[missing data] || user: oi``. The free text
    itself is never translated.
    """
    def tag(match: re.Match) -> str:
        name = STATE_TAGS.get(match.group(1), match.group(1))
        values = re.sub(r"\bnenhum\b", "none", match.group(2))
        return f"[{name}{values}]"

    def speaker(match: re.Match) -> str:
        return f"{match.group(1)}{SPEAKERS[match.group(2).lower()]}:"

    return _SPEAKER.sub(speaker, _TAG.sub(tag, state.strip()))


ACTION_DESCRIPTIONS = {
    "ask": "ask the user for missing information",
    "search": "query a tool, API or database before replying",
    "present_options": "present options from results already fetched",
    "answer": "answer a question about an item or the task",
    "suggest_next_step": "propose the next step, such as booking what was found",
    "confirm": "confirm the details with the user before a transaction",
    "execute": "run the transaction (booking, payment, transfer) and report the outcome",
    "offer_more_help": "ask whether the user needs anything else",
    "end_conversation": "close the conversation",
    "revise_criteria": "ask the user to relax criteria that no option satisfies",
}
