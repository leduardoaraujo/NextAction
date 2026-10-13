# Object detection in images

The `inspect-image` command takes an image and detects people, animals and other objects. It saves a copy with boxes and labels and prints JSON with the class, confidence, coordinates and counts per group.

```powershell
pip install -e ".[vision]"
nextaction inspecionar-imagem caminho/para/foto.jpg
nextaction inspect-image caminho/para/foto.jpg --output resultados/foto_marcada.png --threshold 0.6
```

By default, the annotated image is saved next to the original with the `_detected.png` suffix. The first run downloads the [YOLOS Tiny](https://huggingface.co/hustvl/yolos-tiny) weights from Hugging Face; after that, inference is local. The photo is not sent to the Hub. The model detects COCO classes, including `person`, `cat`, `dog`, `bird`, vehicles and everyday objects. The most common labels are drawn on the image in Portuguese; the JSON also keeps the original English name.

Shortened output example:

```json
{
  "counts": {"people": 0, "animals": 2, "objects": 0},
  "detections": [
    {"label": "cat", "label_pt": "gato", "category": "animals", "confidence": 0.988,
     "box": {"xmin": 151, "ymin": 138, "xmax": 432, "ymax": 342}}
  ]
}
```

This flow uses a pretrained object detector, separate from NextAction's next-action classifier. Results depend on the photo, the confidence threshold and the classes the model knows; not detecting something does not prove it is absent.

## Examples

Four free photos from Wikimedia Commons (public domain or CC0), run with the default threshold of 0.5. The images below are the command's annotated output: green for people, yellow for animals, blue for objects.

| Photo | People | Animals | Objects | What to notice |
| --- | --- | --- | --- | --- |
| Cows in a field | 0 | 2 | 0 | both cows found, one at 99% and one at 77% |
| People and a dog in a park | 10 | 1 | 29 | the dog at 98%; most people found; parked cars counted many times |
| Family picnic | 8 | 0 | 0 | about five people, counted as eight because of overlapping boxes |
| Cat and dog on a couch | 0 | 3 | 1 | the cat is labeled as a dog, and the dog gets two boxes |

![Two cows grazing, each in a yellow box](image-examples/cows.jpg)

![People sitting and standing in a park with a dog; people in green, cars and bags in blue, the dog in yellow](image-examples/park_with_dog.jpg)

![A family having a picnic on a hillside, with overlapping green boxes](image-examples/family_picnic.jpg)

![A cat and a dog sleeping on a couch, both labeled as dogs](image-examples/cat_and_dog.jpg)

The small YOLOS Tiny model is fast and runs on a CPU, but the examples show its limits: it can confuse similar animals and draw several boxes on one object, which inflates the counts. Treat the counts as estimates and check the annotated image when the number matters.

Photo credits (all free to use, no attribution required): [cows](https://commons.wikimedia.org/wiki/File:Cows_in_a_Field.jpg) by KaanaveniSaikiran (CC0); [park](https://commons.wikimedia.org/wiki/File:People_Talking_on_Grass_with_Dog_at_Minsheng_Park_20150503.jpg) by 玄史生 (CC0); [picnic](https://commons.wikimedia.org/wiki/File:Family_picnic_Kulanaokuaiki_Campground_NPS_Photo_Janice_Wei_%2843852400890%29.jpg) by Janice Wei, Hawai‘i Volcanoes NPS (public domain); [cat and dog](https://commons.wikimedia.org/wiki/File:Cat_and_dog_sleeping_together.jpeg) (public domain).
