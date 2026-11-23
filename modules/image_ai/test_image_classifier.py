
from pathlib import Path

from modules.image_ai.image_classifier import classify_image


print("\n===== DATASENTINEL IMAGE RECOGNITION =====\n")

image_path = input(
    "Enter the full path of an image: "
).strip().strip('"')

path = Path(image_path)

if not path.exists() or not path.is_file():

    print("Image file not found.")

else:

    print("\nAnalysing image...")

    result = classify_image(
        file_bytes=path.read_bytes(),
        filename=path.name,
        top_k=5
    )

    if result["success"]:

        print("\n===== RECOGNITION RESULT =====")

        print("Identified as:", result["display_name"])
        print("Category:", result["domain"])
        print("Subcategory:", result["subcategory"])

        print("\n===== COMPLETED =====")

    else:

        print("\nRecognition failed:")
        print(result["error"])
