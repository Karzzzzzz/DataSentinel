"""Advanced open-vocabulary visual analysis for DataSentinel.

Uses pretrained CLIP for broad visual classification and image embeddings. Optional
BLIP captions and DETR object detection are loaded lazily to keep startup light.
The scores are ranking signals, not calibrated probabilities.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import re

import numpy as np

MODEL_NAME = "openai/clip-vit-base-patch32"
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
OBJECT_MODEL_NAME = "facebook/detr-resnet-50"
# ImageNet-1k classifier adds a dependable baseline for everyday objects and animals.
IMAGENET_MODEL_NAME = "google/vit-base-patch16-224"

# Each label is a visible content description, not a statement about a person's identity.
TAXONOMY: dict[str, dict[str, Any]] = {
    "People & Activities": {
        "prompt": "a photograph whose main subject is one or more people",
        "domain": "People & Activities",
        "labels": [
            "single person portrait", "group of people", "family group", "person smiling",
            "person laughing", "person visibly crying with tears", "person frowning",
            "person with a surprised facial expression", "person with a neutral facial expression",
            "person speaking", "person reading a book", "person writing notes",
            "student studying at a desk", "teacher teaching a class", "person using a laptop",
            "person using a smartphone", "person cooking", "person dancing", "person running",
            "person playing a sport", "person holding a baby", "person presenting at a meeting",
            "person wearing a uniform", "person sitting at a desk", "person standing outdoors",
        ],
    },
    "Animals & Wildlife": {
        "prompt": "a photograph whose main subject is an animal or wildlife",
        "domain": "Animals & Wildlife",
        "labels": [
            "dog", "cat", "horse", "cow", "buffalo", "sheep", "goat", "pig", "rabbit",
            "hamster", "squirrel", "deer", "fox", "wolf", "bear", "lion", "tiger",
            "leopard", "cheetah", "elephant", "giraffe", "zebra", "rhinoceros", "hippopotamus",
            "monkey", "chimpanzee", "gorilla", "orangutan", "panda", "koala", "kangaroo",
            "camel", "llama", "dolphin", "whale", "shark", "seal", "sea lion", "octopus",
            "turtle", "tortoise", "snake", "lizard", "crocodile", "frog", "toad", "butterfly",
            "bee", "ant", "spider", "ladybug", "eagle", "owl", "parrot", "peacock", "crow",
            "pigeon", "sparrow", "duck", "goose", "chicken", "rooster", "flamingo", "penguin",
            "swan", "woodpecker", "kingfisher", "fish", "goldfish", "jellyfish", "crab", "lobster",
        ],
    },
    "Plants & Trees": {
        "prompt": "a photograph whose main subject is a plant, tree, flower or crop",
        "domain": "Plants & Agriculture",
        "labels": [
            "oak tree", "maple tree", "pine tree", "spruce tree", "fir tree", "cedar tree",
            "birch tree", "willow tree", "banyan tree", "peepal tree", "neem tree", "mango tree",
            "coconut palm tree", "date palm tree", "banana plant", "papaya tree", "guava tree",
            "apple tree", "orange tree", "lemon tree", "cherry blossom tree", "eucalyptus tree",
            "bamboo", "rose plant", "sunflower", "tulip", "orchid", "lotus flower", "water lily",
            "hibiscus", "marigold", "jasmine", "lavender", " daisy flower", "lily flower",
            "cactus", "succulent", "fern", "moss", "ivy", "grass", "wheat crop", "rice paddy",
            "maize crop", "sugarcane crop", "cotton plant", "tomato plant", "potato plant",
            "tea plantation", "coffee plant", "vegetable garden", "flower garden", "indoor houseplant",
        ],
    },
    "Buildings & Landmarks": {
        "prompt": "a photograph whose main subject is a building, structure or landmark",
        "domain": "Architecture & Places",
        "labels": [
            "house", "apartment building", "skyscraper", "office building", "school building",
            "university building", "hospital building", "shopping mall", "factory building",
            "warehouse", "hotel building", "restaurant building", "church", "mosque", "Hindu temple",
            "Buddhist temple", "palace", "castle", "fort", "museum", "library", "stadium",
            "bridge", "tower", "monument", "statue", "train station", "airport terminal", "lighthouse",
            "Taj Mahal in Agra India", "Eiffel Tower in Paris France", "Statue of Liberty in New York",
            "Big Ben Elizabeth Tower in London", "Colosseum in Rome", "Pyramids of Giza in Egypt",
            "Great Wall of China", "Machu Picchu in Peru", "Sydney Opera House in Australia",
            "Burj Khalifa in Dubai", "Golden Gate Bridge in San Francisco", "Christ the Redeemer in Rio",
            "Angkor Wat in Cambodia", "Petra Treasury in Jordan", "Mount Rushmore in the United States",
            "Sagrada Familia in Barcelona", "Leaning Tower of Pisa in Italy", "Brandenburg Gate in Berlin",
            "India Gate in New Delhi", "Gateway of India in Mumbai", "Charminar in Hyderabad",
        ],
    },
    "Nature & Landscapes": {
        "prompt": "a photograph whose main subject is a natural landscape or outdoor environment",
        "domain": "Nature & Environment",
        "labels": [
            "mountain landscape", "snowy mountains", "beach", "ocean", "river", "waterfall", "lake",
            "forest", "tropical rainforest", "desert", "grassland", "meadow", "wetland", "swamp",
            "volcano", "canyon", "cliff", "cave", "island", "glacier", "snow field", "sunset sky",
            "sunrise sky", "night sky with stars", "clouds", "rainstorm", "rainbow", "autumn forest",
            "spring garden", "farmland", "terraced fields", "park", "botanical garden", "urban street",
            "city skyline", "village landscape", "rural landscape", "underwater coral reef",
        ],
    },
    "Food & Beverage": {
        "prompt": "a photograph whose main subject is food, ingredients or a drink",
        "domain": "Food & Beverage",
        "labels": [
            "cooked meal", "pizza", "burger", "sandwich", "pasta", "noodles", "rice dish", "biryani",
            "curry", "salad", "soup", "bread", "cake", "pastry", "ice cream", "chocolate", "fruit",
            "vegetables", "meat", "seafood", "eggs", "coffee", "tea", "juice", "soft drink", "water bottle",
            "restaurant food", "grocery products", "spices", "cooking ingredients", "dessert", "snack food",
        ],
    },
    "Electronics & Technology": {
        "prompt": "a photograph whose main subject is electronic equipment or technology",
        "domain": "IT & Electronics",
        "labels": [
            "laptop computer", "desktop computer", "computer monitor", "computer keyboard", "computer mouse",
            "smartphone", "mobile phone", "tablet computer", "television", "digital camera", "video camera",
            "headphones", "earbuds", "speaker", "microphone", "printer", "scanner", "router", "modem",
            "server rack", "computer motherboard", "computer circuit board", "microchip", "graphics card",
            "hard drive", "USB flash drive", "game console", "gaming controller", "smartwatch", "drone",
            "robot", "3D printer", "projector", "calculator", "electronic sensor", "Arduino board",
            "Raspberry Pi board", "circuit prototype", "solar panel", "battery", "power bank", "cable and adapter",
        ],
    },
    "Study & Office Supplies": {
        "prompt": "a photograph whose main subject is study material, books or office supplies",
        "domain": "Education & Office",
        "labels": [
            "textbook", "notebook", "open book", "stack of books", "printed worksheet", "handwritten notes",
            "exam paper", "document folder", "paper file", "pen", "pencil", "marker", "highlighter",
            "eraser", "ruler", "geometry set", "school bag", "backpack", "whiteboard", "blackboard",
            "chalkboard writing", "classroom desk", "laboratory equipment", "microscope", "telescope",
            "scientific calculator", "study table", "office stationery", "filing cabinet", "clipboard",
        ],
    },
    "Vehicles & Transport": {
        "prompt": "a photograph whose main subject is a vehicle or transport infrastructure",
        "domain": "Transport & Automotive",
        "labels": [
            "car", "sports car", "taxi", "bus", "school bus", "truck", "van", "motorcycle", "scooter",
            "bicycle", "electric bicycle", "train", "metro train", "tram", "airplane", "helicopter", "boat",
            "ship", "yacht", "tractor", "ambulance", "fire engine", "police car", "auto rickshaw",
            "traffic lights", "road intersection", "railway track", "parking lot", "fuel station",
        ],
    },
    "Sports & Recreation": {
        "prompt": "a photograph whose main subject is sport, exercise or recreation",
        "domain": "Sports & Recreation",
        "labels": [
            "football match", "cricket match", "basketball game", "tennis match", "badminton game",
            "volleyball game", "hockey game", "golf", "swimming", "running race", "cycling race",
            "gym workout", "yoga", "skateboarding", "skiing", "surfing", "boxing", "wrestling",
            "sports equipment", "football", "cricket bat", "basketball", "tennis racket", "medal or trophy",
        ],
    },
    "Furniture & Home": {
        "prompt": "a photograph whose main subject is furniture or home interior",
        "domain": "Household & Interiors",
        "labels": [
            "sofa", "armchair", "dining table", "chair", "bed", "wardrobe", "bookshelf", "desk",
            "kitchen interior", "living room", "bedroom interior", "bathroom interior", "home decor",
            "refrigerator", "washing machine", "microwave oven", "air conditioner", "ceiling fan",
            "lamp", "curtains", "carpet", "household appliance", "garden furniture",
        ],
    },
    "Clothing & Accessories": {
        "prompt": "a photograph whose main subject is clothing, footwear or accessories",
        "domain": "Fashion & Retail",
        "labels": [
            "shirt", "t-shirt", "trousers", "jeans", "dress", "saree", "kurta", "jacket", "coat",
            "shoes", "sneakers", "boots", "sandals", "handbag", "backpack", "watch", "glasses",
            "hat", "helmet", "jewellery", "ring", "necklace", "scarf", "uniform",
        ],
    },
    "Documents & Text": {
        "prompt": "a photograph whose main subject is a printed or handwritten document",
        "domain": "Documents & Records",
        "labels": [
            "invoice document", "receipt document", "identity card", "passport", "certificate", "medical report",
            "prescription document", "bank statement", "contract document", "form", "business card",
            "newspaper page", "magazine page", "printed report", "handwritten letter", "white paper document",
        ],
    },
    "Tools & Industrial": {
        "prompt": "a photograph whose main subject is tools, machinery or industrial equipment",
        "domain": "Engineering & Industry",
        "labels": [
            "hammer", "screwdriver", "wrench", "drill", "saw", "toolbox", "power tools", "industrial machine",
            "robotic arm", "factory machinery", "3D printer", "welding equipment", "construction equipment",
            "excavator", "crane", "safety helmet", "protective equipment", "measuring instrument",
        ],
    },
    "Art & Design": {
        "prompt": "a photograph whose main subject is art, craft, graphics or design",
        "domain": "Art & Design",
        "labels": [
            "painting", "drawing", "sketch", "sculpture", "pottery", "handicraft", "poster design",
            "infographic", "chart", "graph", "diagram", "architectural drawing", "digital illustration",
            "abstract artwork", "photograph of artwork", "wall mural", "map", "blueprint",
        ],
    },
    "Other / Unclear": {
        "prompt": "a photograph of an object or scene not covered by the other categories",
        "domain": "Unknown / Needs Review",
        "labels": ["miscellaneous object", "indoor scene", "outdoor scene", "abstract image", "unclear image"],
    },
}

EXPRESSION_PROMPTS = [
    ("smiling", "a person visibly smiling with the corners of the mouth raised"),
    ("laughing", "a person visibly laughing with an open smiling mouth"),
    ("visible crying cues", "a person visibly crying with tears on the face"),
    ("frowning / downturned mouth", "a person visibly frowning with a downturned mouth"),
    ("wide eyes / raised eyebrows", "a person with visibly widened eyes and raised eyebrows"),
    ("furrowed brows / tense facial appearance", "a person with furrowed eyebrows and a visibly tense face"),
    ("neutral facial cue", "a person with a relaxed neutral facial expression"),
    ("eyes closed", "a person with their eyes visibly closed"),
    ("not clear", "a person whose facial expression cannot be clearly seen"),
]


def _pil_image(image_bytes: bytes):
    from PIL import Image, ImageOps
    import io
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((1024, 1024))
    return image


@lru_cache(maxsize=1)
def _load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return processor, model, device


def _clip_rank(image, prompts: list[str]) -> tuple[list[tuple[int, float]], np.ndarray]:
    import torch
    processor, model, device = _load_clip()
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    text_inputs = processor(text=prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze(0)
        probabilities = torch.softmax(similarity * model.logit_scale.exp(), dim=0)
    probs = probabilities.detach().cpu().numpy().astype(float)
    sims = similarity.detach().cpu().numpy().astype(float)
    order = np.argsort(probs)[::-1]
    ranked = [(int(i), float(probs[i])) for i in order]
    return ranked, image_features[0].detach().cpu().numpy().astype(float)


@lru_cache(maxsize=1)
def _load_captioner():
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-to-text", model=CAPTION_MODEL_NAME, device=device)


def _caption_image(image) -> str:
    captioner = _load_captioner()
    result = captioner(image, max_new_tokens=70)
    if isinstance(result, list) and result:
        return str(result[0].get("generated_text", "")).strip()
    return ""


@lru_cache(maxsize=1)
def _load_object_detector():
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("object-detection", model=OBJECT_MODEL_NAME, device=device)


def _detect_objects(image, threshold: float = 0.55) -> list[dict[str, Any]]:
    detector = _load_object_detector()
    outputs = detector(image)
    result = []
    seen = set()
    for item in outputs:
        score = float(item.get("score", 0.0))
        label = str(item.get("label", "unknown"))
        if score < threshold:
            continue
        key = label.lower()
        if key not in seen:
            result.append({"label": label, "score": round(score, 3), "box": item.get("box", {})})
            seen.add(key)
    return result


# ImageNet has 1,000 familiar object classes, including many dog breeds.
# This second model prevents a single incorrect CLIP family choice from hiding
# a basic class such as dog, cat, bird, car, or laptop.
_DOG_TERMS = (
    "retriever", "terrier", "spaniel", "hound", "poodle", "shepherd",
    "pinscher", "schnauzer", "mastiff", "bulldog", "collie", "husky", "malamute",
    "chihuahua", "pug", "beagle", "rottweiler", "boxer", "dalmatian", "dachshund",
    "corgi", "shih-tzu", "pomeranian", "samoyed", "newfoundland", "great dane",
    "saint bernard", "akita", "basenji", "basset", "borzoi", "whippet", "greyhound",
    "afghan", "weimaraner", "vizsla", "keeshond", "papillon", "pekinese", "pekingese",
    "affenpinscher", "schipperke", "komondor", "kuvasz", "briard", "malinois",
    "groenendael", "bouvier", "leonberg", "brabancon", "appenzeller", "entlebucher",
    "redbone", "bloodhound", "foxhound", "wolfhound", "shetland sheepdog", "kelpie",
    "cairn", "airedale", "chow", "eskimo dog", "mexican hairless", "scotch terrier",
    "west highland", "soft-coated wheaten", "bedlington", "norfolk terrier", "norwich terrier",
)


def _clean_imagenet_label(label: str) -> str:
    label = re.sub(r"^n\d+\s+", "", str(label)).strip().lower()
    return label.replace("_", " ")


def _is_dog_label(label: str) -> bool:
    value = _clean_imagenet_label(label)
    if value in {"dog", "domestic dog"} or (value.endswith(" dog") and value != "hot dog"):
        return True
    return any(term in value for term in _DOG_TERMS)


def _imagenet_family(label: str) -> str | None:
    value = _clean_imagenet_label(label)
    if _is_dog_label(value) or any(x in value for x in ("tabby", "siamese cat", "persian cat", "egyptian cat", "tiger cat", "house cat")):
        return "Animals & Wildlife"
    animal_terms = (
        "cat", "horse", "cow", "ox", "sheep", "goat", "pig", "elephant", "tiger", "lion",
        "bear", "deer", "fox", "wolf", "zebra", "monkey", "ape", "bird", "eagle", "owl",
        "parrot", "hen", "rooster", "duck", "goose", "penguin", "fish", "turtle", "snake",
        "lizard", "frog", "rabbit", "hare", "hamster", "squirrel", "butterfly", "bee", "spider",
        "crab", "lobster", "whale", "dolphin", "shark", "octopus", "seal", "rhinoceros", "giraffe",
    )
    if any(term in value for term in animal_terms):
        return "Animals & Wildlife"
    if any(x in value for x in ("laptop", "desktop computer", "computer keyboard", "computer mouse", "monitor", "screen", "cellular telephone", "mobile phone", "telephone", "printer", "modem", "remote control", "joystick", "hard disc", "disk drive", "computer")):
        return "Electronics & Technology"
    if any(x in value for x in ("car", "bus", "truck", "airliner", "airplane", "aircraft", "bicycle", "motor scooter", "moped", "tractor", "ambulance", "fire engine", "canoe", "ship", "boat", "train")):
        return "Vehicles & Transport"
    if any(x in value for x in ("book", "notebook", "paper", "pencil", "pen", "ruler", "school bag", "backpack", "textbook")):
        return "Study & Office Supplies"
    if any(x in value for x in ("flower", "rose", "sunflower", "daisy", "tree", "oak", "pine", "palm", "cactus", "mushroom", "corn", "ear of corn", "acorn", "vegetable", "fruit")):
        return "Plants & Trees"
    if any(x in value for x in ("pizza", "cheeseburger", "hamburger", "hotdog", "hot dog", "burrito", "plate", "spaghetti", "ice cream", "cake", "espresso", "coffee", "orange", "lemon", "banana", "strawberry", "pineapple")):
        return "Food & Beverage"
    if any(x in value for x in ("building", "palace", "castle", "church", "mosque", "monastery", "tower", "bridge", "temple", "dam", "lighthouse")):
        return "Buildings & Landmarks"
    return None


@lru_cache(maxsize=1)
def _load_imagenet_classifier():
    """Load a general ImageNet-1k classifier as a baseline for common objects."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(IMAGENET_MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(IMAGENET_MODEL_NAME).to(device)
    model.eval()
    return processor, model, device


def _imagenet_rank(image, top_k: int = 5) -> dict[str, Any]:
    import torch
    processor, model, device = _load_imagenet_classifier()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    id2label = model.config.id2label
    ranked_indices = np.argsort(probabilities)[::-1][:top_k]
    top = []
    dog_score = 0.0
    family_scores: dict[str, float] = {}
    for index, score in enumerate(probabilities):
        raw_label = id2label.get(index, id2label.get(str(index), str(index)))
        clean_label = _clean_imagenet_label(raw_label)
        family = _imagenet_family(clean_label)
        if family:
            family_scores[family] = family_scores.get(family, 0.0) + float(score)
        if _is_dog_label(clean_label):
            dog_score += float(score)
    for index in ranked_indices:
        raw_label = id2label.get(int(index), id2label.get(str(int(index)), str(index)))
        top.append({
            "label": _clean_imagenet_label(raw_label),
            "score": round(float(probabilities[index]), 4),
        })
    return {
        "model": IMAGENET_MODEL_NAME,
        "top_predictions": top,
        "top_label": top[0]["label"] if top else "unknown",
        "top_score": top[0]["score"] if top else 0.0,
        "dog_score": round(dog_score, 4),
        "family_scores": {key: round(value, 4) for key, value in family_scores.items()},
    }


def _custom_model_prediction(embedding: np.ndarray) -> dict[str, Any] | None:
    model_path = Path("training/saved_models/image_classifier.joblib")
    if not model_path.exists():
        return None
    try:
        import joblib
        bundle = joblib.load(model_path)
        classifier = bundle.get("model") if isinstance(bundle, dict) else bundle
        if classifier is None or not hasattr(classifier, "predict_proba"):
            return None
        probabilities = classifier.predict_proba(embedding.reshape(1, -1))[0]
        classes = list(classifier.classes_)
        order = np.argsort(probabilities)[::-1]
        return {
            "label": str(classes[int(order[0])]),
            "score": float(probabilities[int(order[0])]),
            "second_score": float(probabilities[int(order[1])]) if len(order) > 1 else 0.0,
        }
    except Exception:
        return None


def analyze_image_bytes(
    image_bytes: bytes,
    filename: str = "image",
    *,
    generate_caption: bool = True,
    detect_objects: bool = False,
) -> dict[str, Any]:
    """Classify an image, optionally caption it and detect multiple visible objects."""
    try:
        image = _pil_image(image_bytes)
        group_prompts = [
            f"{info['prompt']}" for info in TAXONOMY.values()
        ]
        group_ranked, embedding = _clip_rank(image, group_prompts)
        group_index, group_score = group_ranked[0]
        group_name = list(TAXONOMY.keys())[group_index]
        group_info = TAXONOMY[group_name]

        labels = group_info["labels"]
        label_prompts = [f"a clear photograph of {label}" for label in labels]
        ranked, _ = _clip_rank(image, label_prompts)
        best_index, best_score = ranked[0]
        best_label = labels[best_index]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # Cross-check with an independent ImageNet classifier. It has a fixed
        # 1,000-class vocabulary and is especially useful for common objects
        # and animal breeds that CLIP may miss when prompts are too broad.
        imagenet_result = None
        try:
            imagenet_result = _imagenet_rank(image, top_k=5)
        except Exception as exc:
            notes_for_models = [f"ImageNet classifier unavailable: {exc}"]
        else:
            notes_for_models = []

        imagenet_top_label = imagenet_result.get("top_label", "") if imagenet_result else ""
        imagenet_top_score = float(imagenet_result.get("top_score", 0.0)) if imagenet_result else 0.0
        imagenet_family = _imagenet_family(imagenet_top_label) if imagenet_top_label else None
        dog_score = float(imagenet_result.get("dog_score", 0.0)) if imagenet_result else 0.0

        # Dog recognition gets a specific cross-model rule: aggregate the
        # probabilities of all ImageNet dog breeds and also consider the top
        # breed class. This avoids relying on CLIP's softmax over dozens of labels.
        imagenet_says_dog = bool(
            imagenet_result
            and ((_is_dog_label(imagenet_top_label) and imagenet_top_score >= 0.10) or dog_score >= 0.22)
        )

        caption = ""
        notes: list[str] = list(notes_for_models)
        if generate_caption:
            try:
                caption = _caption_image(image)
            except Exception as exc:
                notes.append(f"Caption model unavailable: {exc}")

        objects: list[dict[str, Any]] = []
        if detect_objects:
            try:
                objects = _detect_objects(image)
            except Exception as exc:
                notes.append(f"Object detector unavailable: {exc}")

        detector_says_dog = any(
            str(item.get("label", "")).strip().lower() == "dog" and float(item.get("score", 0.0)) >= 0.50
            for item in objects
        )
        imagenet_family_score = 0.0
        if imagenet_result and imagenet_family:
            imagenet_family_score = float(imagenet_result.get("family_scores", {}).get(imagenet_family, 0.0))

        if imagenet_says_dog or detector_says_dog:
            group_name = "Animals & Wildlife"
            group_info = TAXONOMY[group_name]
            # Keep the broad, reliable class in the main label; expose the
            # ImageNet breed guess separately so it is not overstated.
            best_label = "dog"
            best_score = max(best_score, dog_score, imagenet_top_score)
            group_score = max(group_score, dog_score, imagenet_top_score)
        elif imagenet_family and imagenet_top_score >= 0.45:
            group_name = imagenet_family
            if group_name in TAXONOMY:
                group_info = TAXONOMY[group_name]
                if group_name == "Animals & Wildlife":
                    best_label = imagenet_top_label
                best_score = max(best_score, imagenet_top_score)
                group_score = max(group_score, imagenet_family_score, imagenet_top_score)

        # Confidence is deliberately conservative when both general models
        # disagree. Scores are model ranking signals, not calibrated certainty.
        needs_review = (
            best_score < 0.12
            or (best_score - second_score) < 0.008
            or group_score < 0.18
        )
        if imagenet_says_dog or detector_says_dog:
            needs_review = False

        expression = None
        if group_name == "People & Activities":
            exp_ranked, _ = _clip_rank(image, [p for _, p in EXPRESSION_PROMPTS])
            exp_index, exp_score = exp_ranked[0]
            exp_label = EXPRESSION_PROMPTS[exp_index][0]
            if exp_score >= 0.24 and exp_label != "not clear":
                expression = {"visible_cue": exp_label, "relative_score": round(exp_score, 3)}
            else:
                expression = {"visible_cue": "not clear", "relative_score": round(exp_score, 3)}

        custom_prediction = _custom_model_prediction(embedding)
        final_label = best_label
        custom_used = False
        if custom_prediction and custom_prediction["score"] >= 0.65 and (
            custom_prediction["score"] - custom_prediction["second_score"] >= 0.12
        ):
            final_label = custom_prediction["label"]
            custom_used = True

        tags = [final_label, group_name]
        if caption:
            tags.extend(re.findall(r"[A-Za-z][A-Za-z'-]{2,}", caption.lower())[:12])
        tags.extend(item["label"].lower() for item in objects)
        tags = list(dict.fromkeys(t for t in tags if t))

        return {
            "success": True,
            "filename": filename,
            "domain": group_info["domain"] if not needs_review else "Unknown / Needs Review",
            "visual_group": group_name,
            "predicted_category": final_label,
            "subcategory": best_label,
            "relative_score": round(best_score, 4),
            "group_relative_score": round(group_score, 4),
            "needs_review": bool(needs_review),
            "top_predictions": [
                {"label": labels[i], "relative_score": round(score, 4)}
                for i, score in ranked[:5]
            ],
            "imagenet_top_label": imagenet_top_label,
            "imagenet_top_score": round(imagenet_top_score, 4),
            "imagenet_dog_score": round(dog_score, 4),
            "imagenet_predictions": imagenet_result.get("top_predictions", []) if imagenet_result else [],
            "imagenet_family_scores": imagenet_result.get("family_scores", {}) if imagenet_result else {},
            "caption": caption,
            "visible_expression": expression,
            "detected_objects": objects,
            "tags": tags,
            "visual_embedding": embedding.tolist(),
            "custom_model_prediction": custom_prediction,
            "custom_model_used": custom_used,
            "notes": notes + [
                "CLIP scores rank the candidate labels; they are not calibrated probabilities.",
                "The ImageNet-1k model provides an independent check for common objects and animal breeds; its scores are also model scores, not guaranteed probabilities.",
                "For people, this reports visible expression cues only. It does not infer emotion, gender identity, age, or nationality from appearance.",
            ],
        }
    except Exception as exc:
        return {
            "success": False,
            "filename": filename,
            "message": str(exc),
            "notes": ["Keep this file in Unknown / Needs Review until the model can analyse it."],
        }


def image_training_labels() -> list[str]:
    """Return supported labels that users can select when correcting predictions."""
    labels = []
    for info in TAXONOMY.values():
        labels.extend(info["labels"])
    labels.extend(["Other / Unclear"])
    return list(dict.fromkeys(labels))


def save_image_training_example(image_bytes: bytes, filename: str, label: str) -> str:
    """Save a user-confirmed image label as a future training example."""
    import csv
    import uuid
    from datetime import datetime, timezone

    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label)).strip("_") or "Other"
    ext = Path(filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        ext = ".jpg"
    root = Path("training")
    image_dir = root / "image_examples" / safe_label
    image_dir.mkdir(parents=True, exist_ok=True)
    saved = image_dir / f"{uuid.uuid4().hex}{ext}"
    saved.write_bytes(image_bytes)

    metadata_path = root / "image_labels.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not metadata_path.exists()
    with metadata_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_path", "label", "source_filename", "saved_at_utc"])
        if new_file:
            writer.writeheader()
        writer.writerow({
            "file_path": saved.as_posix(),
            "label": label,
            "source_filename": Path(filename).name,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        })
    return saved.as_posix()
