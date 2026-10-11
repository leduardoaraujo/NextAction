"""Human-readable messages in English and Portuguese.

Field names, action labels and other machine-readable values stay in English in every
language; only text meant for people is translated.
"""

import os

LANGUAGES = ("en", "pt")

MESSAGES = {
    # Data and training
    "csv_missing_columns": {
        "en": "The CSV needs 'state' and 'action' columns (or 'estado' and 'acao').",
        "pt": "O CSV precisa das colunas 'state' e 'action' (ou 'estado' e 'acao').",
    },
    "csv_empty_fields": {
        "en": "Every example needs a non-empty state and action.",
        "pt": "Todos os exemplos precisam de estado e ação preenchidos.",
    },
    "csv_too_few_examples": {
        "en": "Use at least two actions and four examples per action.",
        "pt": "Use pelo menos duas ações e quatro exemplos por ação.",
    },
    "train_note": {
        "en": "Run `opendecision evaluate` to measure performance before relying on this model.",
        "pt": "Rode `opendecision evaluate` para medir o desempenho antes de confiar neste modelo.",
    },
    "threshold_range": {
        "en": "The threshold must be between 0 and 1.",
        "pt": "O limiar precisa estar entre 0 e 1.",
    },
    "unknown_scenario": {
        "en": "Unknown scenario: {name}. Use one of: {options}.",
        "pt": "Cenário desconhecido: {name}. Use um destes: {options}.",
    },
    "data_file_missing": {
        "en": "{path} does not exist; create it first (see the docs).",
        "pt": "{path} não existe; gere o arquivo antes (veja a documentação).",
    },
    "model_file_missing": {
        "en": "Model file {path} not found. Train the scenario first.",
        "pt": "Arquivo de modelo {path} não encontrado. Treine o cenário antes.",
    },
    "config_missing_field": {
        "en": "{path} is missing the required field '{field}'.",
        "pt": "{path} não tem o campo obrigatório '{field}'.",
    },
    "config_invalid_value": {
        "en": "Invalid value '{value}' for {field}. Use one of: {options}.",
        "pt": "Valor inválido '{value}' para {field}. Use um destes: {options}.",
    },
    "policy_invalid_risk": {
        "en": "Invalid risk '{risk}' for action '{action}'. Use one of: {options}.",
        "pt": "Risco inválido '{risk}' para a ação '{action}'. Use um destes: {options}.",
    },
    "tuning_empty_grid": {
        "en": "The [tuning] section has no parameter lists to search.",
        "pt": "A seção [tuning] não tem listas de parâmetros para testar.",
    },
    # Evaluation
    "test_modes_exclusive": {
        "en": "Use either --test or --recent-fraction, not both.",
        "pt": "Use --test ou --recent-fraction, não os dois.",
    },
    "fraction_range": {
        "en": "The recent fraction must be between 0 and 1.",
        "pt": "A fração recente precisa estar entre 0 e 1.",
    },
    "train_needs_two_actions": {
        "en": "The training data needs at least two actions.",
        "pt": "O treino precisa de pelo menos duas ações.",
    },
    "method_cross_validation": {
        "en": "stratified cross-validation with {folds} folds",
        "pt": "validação cruzada estratificada com {folds} dobras",
    },
    "method_test_file": {
        "en": "separate test set: {path}",
        "pt": "teste separado: {path}",
    },
    "method_recent": {
        "en": "most recent {count} examples of the file",
        "pt": "últimos {count} exemplos do arquivo",
    },
    "small_evaluation_warning": {
        "en": "Only {count} examples evaluated: metrics vary a lot with so few cases.",
        "pt": "Só {count} exemplos avaliados: as métricas variam muito com poucos casos.",
    },
    # Decision log and API
    "decision_not_found": {"en": "Decision not found.", "pt": "Decisão não encontrada."},
    "log_disabled": {
        "en": "The decision log is disabled.",
        "pt": "O registro de decisões está desativado.",
    },
    "model_unavailable": {
        "en": "Model '{scenario}' is unavailable. Train the scenario before calling it.",
        "pt": "Modelo '{scenario}' indisponível. Treine o cenário antes de consultar.",
    },
    # Flight ranking
    "number_invalid": {
        "en": "'{field}' must be a valid number.",
        "pt": "'{field}' precisa ser um número válido.",
    },
    "number_minimum": {
        "en": "'{field}' must be greater than or equal to {minimum}.",
        "pt": "'{field}' precisa ser maior ou igual a {minimum}.",
    },
    "date_format": {
        "en": "'{field}' must use the YYYY-MM-DD format.",
        "pt": "'{field}' precisa estar no formato AAAA-MM-DD.",
    },
    "text_required": {
        "en": "'{field}' must be non-empty text.",
        "pt": "'{field}' precisa ser um texto não vazio.",
    },
    "boolean_required": {
        "en": "'{field}' must be true or false.",
        "pt": "'{field}' precisa ser verdadeiro ou falso.",
    },
    "request_required": {
        "en": "The JSON needs a 'request' object.",
        "pt": "O JSON precisa conter um objeto 'request' (ou 'pedido').",
    },
    "same_origin_destination": {
        "en": "Origin and destination must be different.",
        "pt": "Origem e destino precisam ser diferentes.",
    },
    "return_before_departure": {
        "en": "'return_date' cannot be earlier than 'departure_date'.",
        "pt": "A data de volta não pode ser anterior à data de ida.",
    },
    "weights_keys": {
        "en": "'weights' must contain price, duration and stops.",
        "pt": "'weights' precisa conter price, duration e stops.",
    },
    "weights_zero": {
        "en": "At least one weight must be greater than zero.",
        "pt": "Pelo menos um peso precisa ser maior que zero.",
    },
    "offers_list": {"en": "'offers' must be a list.", "pt": "'offers' precisa ser uma lista."},
    "offer_id_required": {
        "en": "Every offer needs a non-empty text 'id'.",
        "pt": "Cada oferta precisa de um 'id' de texto não vazio.",
    },
    "offer_id_duplicate": {
        "en": "Duplicate offer id: {id}.",
        "pt": "ID de oferta repetido: {id}.",
    },
    "selected_offer_unknown": {
        "en": "'selected_offer' does not match any offer provided.",
        "pt": "A oferta escolhida não corresponde a nenhuma oferta informada.",
    },
    "confirm_notice": {
        "en": "Confirm the details and the current price before any purchase.",
        "pt": "Confirme os detalhes e o preço atual antes de qualquer compra.",
    },
    # SGD import and Hub export
    "sgd_no_dialogues": {
        "en": "No dialogues_*.json found in {path}.",
        "pt": "Nenhum dialogues_*.json em {path}.",
    },
    "unknown_schema": {
        "en": "Unknown schema: {name}. Use one of: {options}.",
        "pt": "Esquema desconhecido: {name}. Use um destes: {options}.",
    },
    "skops_untrusted": {
        "en": "The skops file needs untrusted types; review it before publishing.",
        "pt": "O arquivo skops exige tipos não confiáveis; revise antes de publicar.",
    },
    # Progress stages and job states
    "stage_vectorize": {"en": "Vectorizing text", "pt": "Vetorizando texto"},
    "stage_fit": {"en": "Fitting model", "pt": "Ajustando modelo"},
    "stage_cross_validation": {"en": "Cross-validation", "pt": "Validação cruzada"},
    "stage_predict": {"en": "Predicting", "pt": "Prevendo"},
    "stage_import": {"en": "Importing dialogues", "pt": "Importando diálogos"},
    "stage_translate": {"en": "Translating", "pt": "Traduzindo"},
    "stage_tune": {"en": "Tuning hyperparameters", "pt": "Ajustando hiperparâmetros"},
    "stage_calibrate": {"en": "Calibrating policy", "pt": "Calibrando política"},
    "stage_encode": {"en": "Encoding text", "pt": "Codificando texto"},
    "status_running": {"en": "running", "pt": "rodando"},
    "status_done": {"en": "done", "pt": "concluído"},
    "status_failed": {"en": "failed", "pt": "falhou"},
    "status_stale": {"en": "stale", "pt": "parado"},
    "no_jobs": {"en": "No jobs yet.", "pt": "Nenhum job ainda."},
    # CLI help
    "cli_description": {
        "en": "OpenDecision: predict an agent's next action and know when to ask a human.",
        "pt": "OpenDecision: prevê a próxima ação de um agente e sabe quando pedir revisão humana.",
    },
    "cli_lang": {
        "en": "language for messages: en or pt (default: $OPENDECISION_LANG or en)",
        "pt": "idioma das mensagens: en ou pt (padrão: $OPENDECISION_LANG ou en)",
    },
    "cli_scenario": {"en": "scenario: {options}", "pt": "cenário: {options}"},
    "cli_train": {"en": "train a scenario from a labeled CSV", "pt": "treina um cenário a partir de um CSV rotulado"},
    "cli_decide": {"en": "choose the next action for a state", "pt": "escolhe a próxima ação para um estado"},
    "cli_evaluate": {
        "en": "measure per-action errors, calibration and review thresholds",
        "pt": "mede erros por ação, calibração e limiares de revisão",
    },
    "cli_import_sgd": {
        "en": "convert a Schema-Guided Dialogue split into a state,action CSV",
        "pt": "converte um split do Schema-Guided Dialogue em CSV state,action",
    },
    "cli_export_labels": {
        "en": "turn decisions with feedback into training data",
        "pt": "converte decisões com feedback em dados de treino",
    },
    "cli_export_hf": {
        "en": "write a skops model and model card for the Hugging Face Hub",
        "pt": "gera modelo skops e model card para o Hugging Face Hub",
    },
    "cli_rank_flights": {
        "en": "filter and rank the flight offers of a trip described in JSON",
        "pt": "filtra e ordena as ofertas de uma viagem descrita em JSON",
    },
    "cli_data": {"en": "labeled CSV (default: the scenario's data)", "pt": "CSV rotulado (padrão: dados do cenário)"},
    "cli_model": {"en": "model file (default: the scenario's model)", "pt": "arquivo do modelo (padrão: modelo do cenário)"},
    "cli_min_df": {
        "en": "drop terms seen in fewer examples than this",
        "pt": "descarta termos vistos em menos exemplos que isso",
    },
    "cli_state": {"en": "text describing the current situation", "pt": "texto que descreve a situação atual"},
    "cli_threshold": {
        "en": "minimum confidence to act without review (default: the policy's per-action thresholds)",
        "pt": "confiança mínima para agir sem revisão (padrão: limiares por ação da política)",
    },
    "cli_test": {"en": "separate test CSV", "pt": "CSV de teste separado"},
    "cli_recent_fraction": {
        "en": "use the last rows of the CSV as the test set (chronological order)",
        "pt": "usa as últimas linhas do CSV como teste (ordem cronológica)",
    },
    "cli_split": {"en": "SGD train, dev or test folder", "pt": "pasta train, dev ou test do SGD"},
    "cli_output": {"en": "output path", "pt": "caminho de saída"},
    "cli_schema": {
        "en": "generic (all domains) or flights (flight turns only)",
        "pt": "generic (todos os domínios) ou flights (só turnos de voos)",
    },
    "cli_context": {"en": "previous turns to include", "pt": "turnos anteriores incluídos"},
    "cli_no_state": {"en": "use only the conversation text", "pt": "usa só o texto da conversa"},
    "cli_log": {"en": "decision log (JSONL)", "pt": "registro de decisões (JSONL)"},
    "cli_repo": {
        "en": "Hub repository, e.g. user/opendecision-agent",
        "pt": "repositório no Hub, por exemplo usuario/opendecision-agent",
    },
    "cli_license": {"en": "Hub license id, e.g. mit", "pt": "identificador de licença do Hub, por exemplo mit"},
    "cli_jobs": {"en": "show training and import jobs with their progress", "pt": "mostra os jobs de treino e importação com o progresso"},
    "cli_watch": {"en": "refresh every second until interrupted", "pt": "atualiza a cada segundo até ser interrompido"},
    "cli_no_progress": {"en": "do not draw progress bars", "pt": "não desenha barras de progresso"},
    "cli_tune": {
        "en": "search the [tuning] grid of a scenario for the best model parameters",
        "pt": "procura na grade [tuning] do cenário os melhores parâmetros do modelo",
    },
    "cli_calibrate": {
        "en": "choose per-action thresholds that minimize the policy's expected cost",
        "pt": "escolhe limiares por ação que minimizam o custo esperado da política",
    },
    "cli_write": {
        "en": "save the result into configs/<scenario>.toml",
        "pt": "grava o resultado em configs/<cenario>.toml",
    },
    "cli_trip": {"en": "JSON file with the request and offers", "pt": "arquivo JSON com o pedido e as ofertas"},
}


def resolve_language(value: str | None = None) -> str:
    """Map 'pt', 'pt-BR', 'pt_BR', 'en-US'... to a supported language, falling back to the env var."""
    for candidate in (value, os.environ.get("OPENDECISION_LANG"), "en"):
        if candidate:
            code = candidate.strip().lower().replace("_", "-").split("-")[0]
            if code in LANGUAGES:
                return code
    return "en"


def language_from_header(accept_language: str | None) -> str:
    """Pick the first supported language listed in an Accept-Language header."""
    for part in (accept_language or "").split(","):
        code = part.split(";")[0].strip().lower().split("-")[0]
        if code in LANGUAGES:
            return code
    return resolve_language()


def t(key: str, lang: str | None = None, **fmt) -> str:
    return MESSAGES[key][resolve_language(lang)].format(**fmt)


class LocalizedError(ValueError):
    """A validation error whose message can be rendered in any supported language."""

    def __init__(self, key: str, **fmt):
        self.key, self.fmt = key, fmt
        super().__init__(t(key, "en", **fmt))

    def message(self, lang: str | None = None) -> str:
        return t(self.key, lang, **self.fmt)
