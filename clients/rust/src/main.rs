use reqwest::blocking::Client;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::BTreeMap, env, error::Error, fs, time::Duration};

// Requests are sent with the English field names; trip files may use either language.

#[derive(Serialize)]
struct ActionRequest<'a> {
    scenario: &'a str,
    state: &'a str,
    threshold: f64,
}

#[derive(Deserialize, Serialize)]
struct ActionResponse {
    #[serde(rename = "type")]
    kind: String,
    action: String,
    probabilities: BTreeMap<String, f64>,
    needs_review: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    decision_id: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct FlightRequest {
    #[serde(alias = "pedido")]
    request: TripRequest,
    #[serde(default, alias = "ofertas")]
    offers: Vec<FlightOffer>,
    #[serde(alias = "oferta_escolhida", skip_serializing_if = "Option::is_none")]
    selected_offer: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct TripRequest {
    #[serde(alias = "origem")]
    origin: Option<String>,
    #[serde(alias = "destino")]
    destination: Option<String>,
    #[serde(alias = "data_ida")]
    departure_date: Option<String>,
    #[serde(alias = "data_volta")]
    return_date: Option<String>,
    #[serde(alias = "passageiros")]
    passengers: Option<u32>,
    #[serde(alias = "orcamento_max")]
    max_budget: Option<f64>,
    #[serde(alias = "max_escalas")]
    max_stops: Option<u32>,
    #[serde(alias = "bagagem_obrigatoria")]
    baggage_required: Option<bool>,
    #[serde(alias = "pesos")]
    weights: Option<Weights>,
}

#[derive(Deserialize, Serialize)]
struct Weights {
    #[serde(alias = "preco")]
    price: f64,
    #[serde(alias = "duracao")]
    duration: f64,
    #[serde(alias = "escalas")]
    stops: f64,
}

#[derive(Deserialize, Serialize)]
struct FlightOffer {
    id: String,
    #[serde(alias = "preco_total")]
    total_price: f64,
    #[serde(alias = "duracao_min")]
    duration_min: f64,
    #[serde(alias = "escalas")]
    stops: u32,
    #[serde(alias = "bagagem_incluida")]
    baggage_included: bool,
}

#[derive(Deserialize, Serialize)]
struct RankedFlight {
    #[serde(flatten)]
    offer: FlightOffer,
    score: f64,
    components: BTreeMap<String, f64>,
}

#[derive(Deserialize, Serialize)]
struct DiscardedFlight {
    id: String,
    reasons: Vec<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum FlightAction {
    Ask,
    Search,
    PresentOptions,
    ReviseCriteria,
    Confirm,
}

#[derive(Deserialize, Serialize)]
struct FlightResponse {
    action: FlightAction,
    #[serde(skip_serializing_if = "Option::is_none")]
    missing_fields: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    criteria: Option<TripRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recommendation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ranking: Option<Vec<RankedFlight>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    discarded: Option<Vec<DiscardedFlight>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    applied_weights: Option<Weights>,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_offer: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    notice: Option<String>,
}

fn post<Request: Serialize, Response: DeserializeOwned>(
    client: &Client,
    base_url: &str,
    path: &str,
    body: &Request,
) -> Result<Response, Box<dyn Error>> {
    let url = format!("{}{path}", base_url.trim_end_matches('/'));
    let language = env::var("OPENDECISION_LANG").unwrap_or_else(|_| "en".to_string());
    let response = match client.post(url).header("Accept-Language", language).json(body).send() {
        Ok(response) => response,
        Err(error) if error.is_connect() || error.is_timeout() => {
            return Err(format!(
                "Could not connect to the API at {base_url}. Start the server in another terminal with \
                 'uvicorn opendecision.api:app --host 127.0.0.1 --port 8000' or set OPENDECISION_API_URL. \
                 Details: {error}"
            )
            .into());
        }
        Err(error) => return Err(error.into()),
    };
    let status = response.status();
    if !status.is_success() {
        return Err(format!("API returned {status}: {}", response.text()?).into());
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
            let threshold = args.get(4).map_or(Ok(0.70), |v| v.parse::<f64>())?;
            let request = ActionRequest {
                scenario: &args[2],
                state: &args[3],
                threshold,
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
            return Err(
                "Usage: opendecision-client action <agent|flights|research> <state> [threshold] | flights <trip.json>"
                    .into(),
            );
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}
