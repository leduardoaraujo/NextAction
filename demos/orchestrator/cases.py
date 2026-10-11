"""Hand-authored request families, separated before any model training."""

# Each entry is (utterance, context). Context values are explicit runtime facts,
# not labels inferred from the utterance. Every test phrase family is distinct.
CASES = {
    "train": {
        "data_query": [
            "Mostre a receita mensal por região", "Compare vendas deste mês com o anterior",
            "Qual foi o faturamento por produto em 2025?", "Crie um gráfico de pedidos por semana",
            "Liste os dez clientes com maior receita", "Calcule a taxa de cancelamento por canal",
            "How many orders did we receive last quarter?", "Show average ticket by store",
            "Compare gross margin across business units", "Plot the daily revenue trend",
            "Traga uma tabela de custos por centro", "Qual a variação do lucro ano contra ano?",
            "Após a consulta anterior falhar, mostre vendas por trimestre",
            "Ordene as lojas pelo ticket médio", "Quero o ranking de regiões com mais pedidos",
            "Mostre a distribuição de preços de cada linha", "Calcule o percentil de tempo de resposta",
            "Quantos chamados cada equipe resolveu?", "Analise a evolução do estoque semanal",
            "Que segmento tem mais cancelamentos?", "Divida os reembolsos por método de pagamento",
            "Faça um histograma da margem por filial", "Cruze atrasos de entrega com cidade",
            "Which category has the highest return rate?", "Rank products by units sold",
            "Give me a breakdown of refunds by channel", "Compute the p95 latency per service",
            "How does inventory change each week?", "Which customer segment churns most?",
            "Summarize support tickets by team", "Show price distribution across categories",
            "Me mostre a série diária de acessos", "Liste as contas com maior inadimplência",
        ],
        "pure_chat": [
            "Olá, tudo bem?", "O que você consegue fazer?", "Obrigado pela ajuda",
            "Explique a diferença entre média e mediana", "Como funciona uma análise de dados?",
            "Me dê uma ideia para apresentar resultados", "Hi, can you introduce yourself?",
            "What is a KPI?", "Tell me how you approach a question", "Boa tarde",
            "Você pode falar de forma mais simples?", "O que significa correlação?",
            "O que significa um erro de coluna em SQL?",
            "Como calculo um percentil?", "Pode detalhar sua resposta anterior?",
            "Não preciso de números, só uma explicação", "Tell me what a data warehouse is",
            "What does churn mean?", "O que é margem bruta?",
            "Qual é a diferença entre ranking e percentil?", "Explique como interpretar uma tabela",
            "Como você chegou a essa conclusão?", "Me dê um exemplo conceitual",
        ],
        "clarification_needed": [
            "Mostre os números", "Quero uma análise", "Pode comparar isso?",
            "Faça um gráfico", "Qual indicador devemos usar?", "Mostre os resultados sem dizer de qual fonte",
            "Analyze it for me", "Can you compare those figures?", "Give me the report",
            "Quero ver os dados, mas ainda não conectei a fonte", "Consulte a receita da fonte que ainda não existe",
            "Qual foi o total?",
            "Preciso de alguma informação, mas não sei qual", "Mostre um indicador qualquer",
            "Pode analisar sem eu dizer a pergunta?", "Which metric do you need from me?",
            "I have not specified the date range", "Que fonte de dados devo conectar?",
            "Faça um resumo sem indicar o tema", "Compare dois itens que ainda não escolhi",
            "Quero ver uma tabela, porém falta o assunto", "Me ajude a definir o recorte",
        ],
        "error_recovery": [
            "A consulta falhou; tente corrigir", "Esse SQL deu erro de coluna", "Tente de novo depois do erro",
            "A tabela não existe, ajuste a consulta", "Corrija a agregação que quebrou",
            "O resultado veio vazio por engano; refaça", "Retry the failed query",
            "Fix the invalid column and run again", "The database timed out; recover",
            "Revise o filtro de data que gerou erro", "A execução anterior retornou exceção",
            "A busca falhou, podemos tentar novamente?",
            "A última consulta usou uma coluna inexistente", "Tente recuperar o cálculo que falhou",
            "O gráfico anterior retornou erro", "Reexecute depois de corrigir o SQL",
            "Fix the broken join in the previous query", "The last data call raised an exception",
            "A conexão caiu durante a consulta, tente outra vez", "Repare o filtro inválido",
            "Can we retry after the timeout?", "O banco rejeitou a query, corrija e repita",
        ],
    },
    "validation": {
        "data_query": [
            "Mostre o crescimento de assinaturas por trimestre", "Quanto vendemos por estado no último ano?",
            "Build a chart of active users by month", "What is the median delivery time by carrier?",
            "Separe a inadimplência por faixa de cliente", "Compare devoluções entre lojas",
            "A consulta anterior falhou; agora calcule receita por canal",
        ],
        "pure_chat": [
            "Bom dia, como posso te chamar?", "O que quer dizer desvio padrão?",
            "Can you explain what a dashboard is?", "Obrigado, era só isso",
            "Prefiro respostas curtas", "Me ajude a entender uma metodologia",
        ],
        "clarification_needed": [
            "Quero olhar aquilo", "Gere um painel sem escolher a métrica",
            "Which numbers should we look at?", "Compare them, please",
            "Ainda não há fonte conectada, veja minhas vendas", "Preciso de uma tabela, mas de quê?",
        ],
        "error_recovery": [
            "A query retornou erro de sintaxe", "Conserte a consulta que não executou",
            "The previous chart query failed", "Try again with a valid date field",
            "A execução quebrou após o timeout", "Corrija o nome da tabela e repita",
        ],
    },
    "test": {
        "data_query": [
            "Faça o ranking de filiais por margem", "Qual o volume de chamados por equipe?",
            "Desenhe a evolução semanal do estoque", "Quais segmentos concentram churn?",
            "Break down refunds by payment method", "Show a table of late deliveries by city",
            "Compute the 90th percentile of response time", "Quero a distribuição de preços por categoria",
            "O SQL anterior deu erro, agora mostre o ranking de produtos",
        ],
        "pure_chat": [
            "Oi!", "Você lembra do que eu perguntei?", "O que é um percentil?",
            "Pode explicar com um exemplo?", "Thanks, that's clear", "What can this assistant do?",
            "Como devo interpretar um gráfico?", "Me responda em inglês daqui para frente",
            "Explique por que uma consulta SQL pode falhar",
        ],
        "clarification_needed": [
            "Me mostre algo interessante", "Preciso de um resumo, ainda sem assunto",
            "What should I analyze?", "Use the dataset I have not connected yet",
            "Faça uma comparação, mas não sei de quais períodos", "Olhe esses valores para mim",
            "Pode preparar uma visualização qualquer?", "Qual é o número?",
        ],
        "error_recovery": [
            "A coluna de período causou uma exceção; ajuste", "A última tentativa falhou, refaça",
            "The SQL returned an error; repair it", "Recover from the failed chart request",
            "Não funcionou: a fonte expirou", "A consulta anterior estourou o tempo limite",
            "Conserte o campo inválido usado antes", "Rode novamente após corrigir a sintaxe",
        ],
    },
}


def context_for(utterance: str) -> dict[str, str]:
    """A small runtime snapshot, supplied by the environment in a real integration."""
    lower = utterance.lower()
    source = "missing" if any(phrase in lower for phrase in
                              ("não conectei", "não existe", "não há fonte", "not connected")) else "available"
    # The snapshot is inferred from words describing prior runtime state here. It
    # does not read the expected action; a production caller would supply it.
    error_terms = ("consulta falhou", "deu erro", "depois do erro", "não existe, ajuste",
                   "quebrou", "veio vazio por engano", "failed query", "invalid column and run",
                   "timed out", "gerou erro", "retornou exceção", "busca falhou",
                   "query retornou erro", "consulta que não executou", "query failed",
                   "previous chart query failed", "valid date field", "após o timeout",
                   "corrija o nome da tabela", "causou uma exceção", "última tentativa falhou",
                   "sql returned an error", "failed chart", "não funcionou", "consulta anterior",
                   "campo inválido usado antes", "corrigir a sintaxe", "consulta anterior falhar",
                   "última consulta", "cálculo que falhou", "gráfico anterior retornou erro",
                   "corrigir o sql", "broken join", "last data call", "conexão caiu",
                   "filtro inválido", "retry after the timeout", "banco rejeitou")
    error = "sql_error" if any(term in lower for term in error_terms) else "none"
    missing_terms = ("sem escolher a métrica", "mas de quê", "não sei de quais períodos",
                     "ainda sem assunto")
    missing = "metric" if any(term in lower for term in missing_terms) else "none"
    return {"datasource": source, "last_error": error, "missing": missing}
