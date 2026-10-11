use reqwest::blocking::Client;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::BTreeMap, env, error::Error, fs, time::Duration};

#[derive(Serialize)]
struct ActionRequest<'a> {
    cenario: &'a str,
    estado: &'a str,
    limiar: f64,
}

#[derive(Deserialize, Serialize)]
struct ActionResponse {
    tipo: String,
    proxima_acao: String,
    probabilidades_estimadas: BTreeMap<String, f64>,
    pedir_revisao: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    decisao_id: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct FlightRequest {
    pedido: TripRequest,
    #[serde(default)]
    ofertas: Vec<FlightOffer>,
    #[serde(skip_serializing_if = "Option::is_none")]
    oferta_escolhida: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct TripRequest {
    origem: Option<String>,
    destino: Option<String>,
    data_ida: Option<String>,
    data_volta: Option<String>,
    passageiros: Option<u32>,
    orcamento_max: Option<f64>,
    max_escalas: Option<u32>,
    bagagem_obrigatoria: Option<bool>,
    pesos: Option<Weights>,
}

#[derive(Deserialize, Serialize)]
struct Weights {
    preco: f64,
    duracao: f64,
    escalas: f64,
}

#[derive(Deserialize, Serialize)]
struct FlightOffer {
    id: String,
    preco_total: f64,
    duracao_min: f64,
    escalas: u32,
    bagagem_incluida: bool,
}

#[derive(Deserialize, Serialize)]
struct RankedFlight {
    #[serde(flatten)]
    oferta: FlightOffer,
    pontuacao: f64,
    componentes: BTreeMap<String, f64>,
}

#[derive(Deserialize, Serialize)]
struct DiscardedFlight {
    id: String,
    motivos: Vec<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum FlightAction {
    PerguntarDados,
    BuscarVoos,
    ApresentarOpcoes,
    ReverCriterios,
    PedirConfirmacao,
}

#[derive(Deserialize, Serialize)]
struct FlightResponse {
    acao: FlightAction,
    #[serde(skip_serializing_if = "Option::is_none")]
    campos_faltantes: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    criterios: Option<TripRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recomendacao: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ranking: Option<Vec<RankedFlight>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    descartadas: Option<Vec<DiscardedFlight>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pesos_aplicados: Option<Weights>,
    #[serde(skip_serializing_if = "Option::is_none")]
    oferta_escolhida: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aviso: Option<String>,
}

fn post<Request: Serialize, Response: DeserializeOwned>(
    client: &Client,
    base_url: &str,
    path: &str,
    body: &Request,
) -> Result<Response, Box<dyn Error>> {
    let url = format!("{}{path}", base_url.trim_end_matches('/'));
    let response = match client.post(url).json(body).send() {
        Ok(response) => response,
        Err(error) if error.is_connect() || error.is_timeout() => {
            return Err(format!(
                "Não foi possível conectar à API em {base_url}. Inicie o servidor em outro terminal com \
                 '.\\.venv\\Scripts\\python -m uvicorn opendecision.api:app --host 127.0.0.1 --port 8000' \
                 ou ajuste OPENDECISION_API_URL. Detalhe: {error}"
            )
            .into());
        }
        Err(error) => return Err(error.into()),
    };
    let status = response.status();
    if !status.is_success() {
        return Err(format!("API retornou {status}: {}", response.text()?).into());
    }
    Ok(response.json::<Response>()?)
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let base_url =
        env::var("OPENDECISION_API_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
    let client = Client::builder().timeout(Duration::from_secs(10)).build()?;

    match args.get(1).map(String::as_str) {
        Some("action") if args.len() == 4 || args.len() == 5 => {
            let limiar = args.get(4).map_or(Ok(0.70), |v| v.parse::<f64>())?;
            let request = ActionRequest {
                cenario: &args[2],
                estado: &args[3],
                limiar,
            };
            let result: ActionResponse = post(&client, &base_url, "/v1/actions/decide", &request)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Some("flights") if args.len() == 3 => {
            let request: FlightRequest = serde_json::from_str(&fs::read_to_string(&args[2])?)?;
            let result: FlightResponse =
                post(&client, &base_url, "/v1/flights/evaluate", &request)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        _ => {
            return Err("Uso: opendecision-client action <pesquisa|passagens> <estado> [limiar] | flights <arquivo.json>".into());
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Erro: {error}");
        std::process::exit(1);
    }
}
