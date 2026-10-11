"""Contratos JSON da API pública."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActionRequest(StrictModel):
    cenario: Literal["pesquisa", "passagens", "agente"]
    estado: str = Field(min_length=1)
    limiar: float = Field(default=0.70, ge=0, le=1)


class ActionResponse(StrictModel):
    tipo: Literal["escolha"]
    proxima_acao: str
    probabilidades_estimadas: dict[str, float]
    pedir_revisao: bool
    decisao_id: str | None = None


class FeedbackRequest(StrictModel):
    acao_final: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    resultado: str | None = Field(default=None, max_length=500)
    comentario: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(StrictModel):
    decisao_id: str
    acao_final: str
    registrado_em: str


class Pesos(StrictModel):
    preco: float = Field(ge=0)
    duracao: float = Field(ge=0)
    escalas: float = Field(ge=0)


class PedidoPassagem(StrictModel):
    origem: str | None = None
    destino: str | None = None
    data_ida: str | None = None
    data_volta: str | None = None
    passageiros: int | None = Field(default=None, ge=1)
    orcamento_max: float | None = Field(default=None, gt=0)
    max_escalas: int | None = Field(default=None, ge=0)
    bagagem_obrigatoria: bool = False
    pesos: Pesos | None = None


class OfertaPassagem(StrictModel):
    id: str = Field(min_length=1)
    preco_total: float = Field(gt=0)
    duracao_min: float = Field(gt=0)
    escalas: int = Field(ge=0)
    bagagem_incluida: bool


class FlightRequest(StrictModel):
    pedido: PedidoPassagem
    ofertas: list[OfertaPassagem] = Field(default_factory=list)
    oferta_escolhida: str | None = None


class OfertaPontuada(OfertaPassagem):
    pontuacao: float
    componentes: dict[str, float]


class OfertaDescartada(StrictModel):
    id: str
    motivos: list[str]


class FlightResponse(StrictModel):
    acao: Literal[
        "perguntar_dados", "buscar_voos", "apresentar_opcoes",
        "rever_criterios", "pedir_confirmacao",
    ]
    campos_faltantes: list[str] | None = None
    criterios: PedidoPassagem | None = None
    recomendacao: str | None = None
    ranking: list[OfertaPontuada] | None = None
    descartadas: list[OfertaDescartada] | None = None
    pesos_aplicados: dict[str, float] | None = None
    oferta_escolhida: str | OfertaPontuada | None = None
    aviso: str | None = None
