# Identificação de objetos em imagens

O comando `inspect-image` recebe uma imagem e detecta pessoas, animais e outros objetos. Ele salva uma cópia com caixas e rótulos e imprime JSON com classe, confiança, coordenadas e contagens por grupo.

```powershell
pip install -e ".[vision]"
nextaction inspecionar-imagem caminho/para/foto.jpg
nextaction inspect-image caminho/para/foto.jpg --output resultados/foto_marcada.png --threshold 0.6
```

Por padrão, a imagem marcada é salva ao lado da original com sufixo `_detected.png`. O primeiro uso baixa os pesos do modelo [YOLOS Tiny](https://huggingface.co/hustvl/yolos-tiny) do Hugging Face; depois a inferência é local. A foto não é enviada ao Hub. O modelo detecta classes do COCO, incluindo `person`, `cat`, `dog`, `bird`, veículos e objetos cotidianos. Os rótulos mais comuns aparecem em português sobre a imagem; o JSON também preserva o nome original em inglês.

Exemplo resumido de saída:

```json
{
  "counts": {"people": 0, "animals": 2, "objects": 0},
  "detections": [
    {"label": "cat", "label_pt": "gato", "category": "animals", "confidence": 0.988,
     "box": {"xmin": 151, "ymin": 138, "xmax": 432, "ymax": 342}}
  ]
}
```

Esse fluxo usa um detector de objetos pré-treinado, separado do classificador de próxima ação do NextAction. Os resultados dependem da foto, do limiar de confiança e das classes que o modelo conhece; não detectar algo não prova que ele esteja ausente.
