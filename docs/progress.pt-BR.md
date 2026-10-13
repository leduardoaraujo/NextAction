# Acompanhando o progresso

Treino, avaliação, importações e exportações rodam como **tarefas**. Cada tarefa desenha uma barra de progresso no terminal e grava `jobs/<id>.json`, para que você possa acompanhá-la de outro terminal ou do navegador.

## No terminal

Comandos longos (`train`, `evaluate`, `import-sgd`, `export-hf`) mostram uma barra por etapa:

```text
Vectorizing text  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Fitting model     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
```

`--no-progress` esconde as barras; o arquivo da tarefa continua sendo gravado. Para acompanhar tarefas iniciadas em outro lugar:

```bash
nextaction jobs            # list recent jobs
nextaction jobs --watch    # refresh every second (Ctrl+C to stop)
nextaction --lang pt tarefas --acompanhar
```

```text
 id                      job          status   stage          %    ETA
 20261010-230805-b5465b  train agent  running  Fitting model  5%
```

## No navegador

Com a API rodando (`uvicorn nextaction.api:app`), abra `http://127.0.0.1:8000/dashboard`. A página consulta `GET /v1/jobs` a cada dois segundos e mostra a etapa, a porcentagem, o tempo restante estimado, o status e o erro de cada tarefa. Ela segue o idioma do navegador (inglês ou português) e o tema claro ou escuro do sistema.

## Arquivo da tarefa

```json
{
  "id": "20261010-230805-b5465b", "name": "train agent", "status": "running",
  "stage": "fit", "done": 50, "total": 1000, "percent": 5.0, "eta_s": null,
  "started_at": "2026-10-11T02:08:05+00:00", "updated_at": "2026-10-11T02:08:25+00:00",
  "error": null, "pid": 13096
}
```

- `status` é `running`, `done` ou `failed`. Uma tarefa em execução sem atualizações por cinco minutos aparece como `stale`, o que geralmente significa que o processo foi encerrado à força.
- `GET /v1/jobs` acrescenta `stage_label` e `status_label` no idioma de `Accept-Language`.
- Os arquivos são gravados de forma atômica, no máximo duas vezes por segundo. A pasta é ignorada pelo git; defina `NEXTACTION_JOBS_DIR` para mudá-la de lugar.

## O que as etapas medem

| Etapa | Unidade de progresso |
| --- | --- |
| `import` | arquivos de diálogo lidos |
| `vectorize` | o ajuste do TF-IDF sobre todos os exemplos (informado no início e no fim) |
| `fit` | iterações do solver em relação a `max_iter`, em blocos de 50; salta para 100% quando o solver converge antes, por isso nenhum tempo restante é mostrado |
| `cross_validation` | folds concluídos |
| `predict` | previsões de teste |

## Informando o progresso no seu próprio código

```python
from nextaction.progress import Job, stage

with Job("my import"):
    with stage("import", total=len(files)) as progress:
        for path in files:
            ...
            progress.advance()
```

Fora de um `Job`, `stage` não faz nada, então as funções de biblioteca podem informar o progresso sem nenhuma condição.
