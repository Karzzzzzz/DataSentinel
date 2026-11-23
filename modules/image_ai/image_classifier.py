
from functools import lru_cache
from io import BytesIO

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"


# Extend these categories as the system grows.
# Each label has a simple display name and a hierarchy.

IMAGE_CATEGORIES = {
    "House": ("Real Estate", "Residential Property"),
    "Apartment": ("Real Estate", "Apartments"),
    "Office Building": ("Buildings", "Commercial Buildings"),
    "School Building": ("Education", "Schools"),
    "College Building": ("Education", "Colleges"),
    "Classroom": ("Education", "Classrooms"),
    "Student": ("Education", "Students"),
    "Teacher": ("Education", "Teachers"),

    "Individual Person": ("People", "Individual Photographs"),
    "Group of People": ("People", "Group Photographs"),
    "Family": ("People", "Family Photographs"),

    "Car": ("Transport", "Cars"),
    "Motorcycle": ("Transport", "Motorcycles"),
    "Truck": ("Transport", "Trucks"),
    "Bus": ("Transport", "Buses"),
    "Bicycle": ("Transport", "Bicycles"),
    "Aircraft": ("Transport", "Aircraft"),
    "Ship": ("Transport", "Ships"),

    "Invoice": ("Documents", "Invoices"),
    "Receipt": ("Documents", "Receipts"),
    "Certificate": ("Documents", "Certificates"),
    "Identity Card": ("Documents", "Identity Documents"),
    "Handwritten Notes": ("Documents", "Handwritten Notes"),
    "Printed Document": ("Documents", "Printed Documents"),
    "Book": ("Education", "Books"),
    "Newspaper": ("Media", "Newspapers"),

    "Food": ("Food", "Food Items"),
    "Vegetables": ("Food", "Vegetables"),
    "Fruits": ("Food", "Fruits"),
    "Groceries": ("Household", "Groceries"),

    "Dog": ("Animals", "Dogs"),
    "Cat": ("Animals", "Cats"),
    "Bird": ("Animals", "Birds"),
    "Wild Animal": ("Animals", "Wildlife"),

    "Office Interior": ("Office", "Workplaces"),
    "Computer": ("Technology", "Computers"),
    "Mobile Phone": ("Technology", "Mobile Devices"),
    "Furniture": ("Household", "Furniture"),
    "Clothing": ("Retail", "Clothing"),

    "Medical Equipment": ("Healthcare", "Medical Equipment"),
    "Medical Image": ("Healthcare", "Medical Images"),

    "Nature": ("Environment", "Nature"),
    "Landscape": ("Environment", "Landscapes"),
    "Diagram": ("Technical", "Diagrams"),
    "Chart or Graph": ("Business", "Charts and Graphs"),
    "Sports": ("Sports", "Sports Activities"),
    "Artwork": ("Arts", "Artwork"),
    "Other Object": ("Other", "Unclassified Objects")
}


# Descriptions help CLIP distinguish similar categories.

PROMPT_TEMPLATES = {
    "House": "a photograph of a residential house",
    "Apartment": "a photograph of an apartment building",
    "Office Building": "a photograph of a commercial office building",
    "School Building": "a photograph of a school building",
    "College Building": "a photograph of a college or university building",
    "Classroom": "a photograph of a classroom with desks and a board",
    "Student": "a photograph of a student",
    "Teacher": "a photograph of a teacher",

    "Individual Person": "a photograph of one individual person",
    "Group of People": "a photograph of a group of people",
    "Family": "a family photograph",

    "Car": "a photograph of a car",
    "Motorcycle": "a photograph of a motorcycle",
    "Truck": "a photograph of a truck",
    "Bus": "a photograph of a bus",
    "Bicycle": "a photograph of a bicycle",
    "Aircraft": "a photograph of an aircraft",
    "Ship": "a photograph of a ship",

    "Invoice": "a photograph or scan of a business invoice",
    "Receipt": "a photograph or scan of a payment receipt",
    "Certificate": "a photograph of an official certificate",
    "Identity Card": "a photograph of an identity card",
    "Handwritten Notes": "a photograph of handwritten notes",
    "Printed Document": "a photograph of a printed document",
    "Book": "a photograph of a book",
    "Newspaper": "a photograph of a newspaper",

    "Food": "a photograph of prepared food",
    "Vegetables": "a photograph of vegetables",
    "Fruits": "a photograph of fruits",
    "Groceries": "a photograph of grocery items",

    "Dog": "a photograph of a dog",
    "Cat": "a photograph of a cat",
    "Bird": "a photograph of a bird",
    "Wild Animal": "a photograph of a wild animal",

    "Office Interior": "a photograph of an office interior",
    "Computer": "a photograph of a computer or laptop",
    "Mobile Phone": "a photograph of a mobile phone",
    "Furniture": "a photograph of household furniture",
    "Clothing": "a photograph of clothes",

    "Medical Equipment": "a photograph of medical equipment",
    "Medical Image": "a medical scan or medical image",

    "Nature": "a photograph of nature",
    "Landscape": "a photograph of a natural landscape",
    "Diagram": "a technical diagram or schematic",
    "Chart or Graph": "a chart or graph showing data",
    "Sports": "a photograph of a sporting activity",
    "Artwork": "a photograph of a painting or artwork",
    "Other Object": "a photograph of an object"
}


@lru_cache(maxsize=1)
def load_clip_model():
    """
    Load CLIP once and reuse the model.
    Initial execution downloads pretrained model weights.
    """

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model.to("cpu")
    model.eval()

    return processor, model


def classify_image(
    file_bytes: bytes,
    filename: str = "uploaded_image",
    top_k: int = 5
) -> dict:
    """
    Recognise the most likely contents of an image.

    Returns:
    - Main category
    - Domain
    - Subcategory
    - Alternative predictions
    """

    try:

        if not file_bytes:
            raise ValueError("Image file is empty.")

        # Open and validate image
        with Image.open(BytesIO(file_bytes)) as original:

            image = ImageOps.exif_transpose(
                original
            ).convert("RGB")

            if image.width * image.height > 50_000_000:
                raise ValueError(
                    "Image exceeds the maximum pixel limit."
                )

            image = image.copy()

        # Load pretrained model
        processor, model = load_clip_model()

        labels = list(IMAGE_CATEGORIES.keys())

        descriptions = [
            PROMPT_TEMPLATES[label]
            for label in labels
        ]

        # Prepare image and text
        inputs = processor(
            text=descriptions,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Run AI recognition
        with torch.inference_mode():

            outputs = model(**inputs)

            scores = outputs.logits_per_image.softmax(
                dim=1
            )[0]

        count = min(
            max(1, top_k),
            len(labels)
        )

        top_results = torch.topk(
            scores,
            k=count
        )

        predictions = []

        for score, index in zip(
            top_results.values.tolist(),
            top_results.indices.tolist()
        ):

            label = labels[index]

            domain, subcategory = IMAGE_CATEGORIES[label]

            predictions.append({
                "label": label,
                "domain": domain,
                "subcategory": subcategory,
                "relative_score": round(
                    float(score), 4
                )
            })

        best = predictions[0]

        return {
            "success": True,
            "filename": filename,

            "predicted_category": best["label"],
            "display_name": best["label"],
            "domain": best["domain"],
            "subcategory": best["subcategory"],

            "relative_score": best["relative_score"],

            "alternatives": predictions,

            "message": "Image recognition completed."
        }

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        RuntimeError
    ) as error:

        return {
            "success": False,
            "filename": filename,
            "error": str(error)
        }
