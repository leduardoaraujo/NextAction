"""Detect common objects in an image and write an annotated copy."""

from collections import Counter
from pathlib import Path

ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
LABEL_PT = {
    "person": "pessoa", "bird": "pássaro", "cat": "gato", "dog": "cachorro",
    "horse": "cavalo", "sheep": "ovelha", "cow": "vaca", "elephant": "elefante",
    "bear": "urso", "zebra": "zebra", "giraffe": "girafa", "car": "carro",
    "bicycle": "bicicleta", "motorcycle": "moto", "bus": "ônibus", "truck": "caminhão",
    "chair": "cadeira", "couch": "sofá", "dining table": "mesa", "bottle": "garrafa",
    "cup": "xícara", "book": "livro", "cell phone": "celular", "laptop": "notebook",
}
MODEL_ID = "hustvl/yolos-tiny"


def category(label: str) -> str:
    if label == "person":
        return "people"
    if label in ANIMALS:
        return "animals"
    return "objects"


def label_font(image_font, size: int):
    """A system font with accents for Portuguese labels (sofá, pássaro), else Pillow's default."""
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "Helvetica.ttc", "LiberationSans-Regular.ttf"):
        try:
            return image_font.truetype(name, size)
        except OSError:
            continue
    return image_font.load_default(size=size)


def inspect_image(image_path: Path, output_path: Path | None = None, threshold: float = 0.5) -> dict:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be greater than 0 and at most 1")
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    output_path = output_path or image_path.with_name(image_path.stem + "_detected.png")
    if output_path.resolve() == image_path.resolve():
        raise ValueError("output image must differ from input image")

    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from transformers import pipeline

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    detector = pipeline("object-detection", model=MODEL_ID, device=-1)
    raw = detector(image, threshold=threshold)
    detections = []
    for item in raw:
        label = str(item["label"])
        box = {key: int(item["box"][key]) for key in ("xmin", "ymin", "xmax", "ymax")}
        detections.append({"label": label, "label_pt": LABEL_PT.get(label, label),
                           "category": category(label), "confidence": round(float(item["score"]), 3),
                           "box": box})
    detections.sort(key=lambda item: item["confidence"], reverse=True)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = label_font(ImageFont, size=max(14, min(image.size) // 35))
    colors = {"people": "#37c976", "animals": "#ffc247", "objects": "#60a5fa"}
    for item in detections:
        box = item["box"]
        color = colors[item["category"]]
        draw.rectangle((box["xmin"], box["ymin"], box["xmax"], box["ymax"]),
                       outline=color, width=max(2, min(image.size) // 200))
        text = f"{item['label_pt']} {item['confidence']:.0%}"
        bounds = draw.textbbox((0, 0), text, font=font)
        text_height = bounds[3] - bounds[1] + 6
        top = max(0, box["ymin"] - text_height)
        width = bounds[2] - bounds[0] + 10
        draw.rectangle((box["xmin"], top, min(image.width, box["xmin"] + width), top + text_height), fill=color)
        draw.text((box["xmin"] + 5, top + 2), text, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    counts = Counter(item["category"] for item in detections)
    return {"image": str(image_path), "annotated_image": str(output_path), "model": MODEL_ID,
            "threshold": threshold, "counts": {key: counts[key] for key in ("people", "animals", "objects")},
            "detections": detections}
