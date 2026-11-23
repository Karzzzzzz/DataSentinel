
from io import BytesIO
from hashlib import sha256

import numpy as np
import cv2

from PIL import Image, ImageOps, UnidentifiedImageError


MAX_IMAGE_PIXELS = 50_000_000


def extract_image_data(filename: str, file_bytes: bytes) -> dict:
    """
    Extract image metadata and basic quality indicators.

    Supports formats handled by Pillow, including
    JPEG, PNG, WEBP, GIF, BMP and TIFF.
    """

    if not file_bytes:
        return {
            "success": False,
            "filename": filename,
            "error": "Image file is empty."
        }

    try:
        # Verify that the uploaded file is a valid image
        with Image.open(BytesIO(file_bytes)) as original:
            original.verify()

        # Reopen after verification
        with Image.open(BytesIO(file_bytes)) as original:

            image_format = original.format
            original_width, original_height = original.size
            original_mode = original.mode

            if original_width <= 0 or original_height <= 0:
                raise ValueError("Invalid image dimensions.")

            if original_width * original_height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    "Image exceeds the maximum pixel limit."
                )

            # Correct image orientation using EXIF metadata
            image = ImageOps.exif_transpose(original).convert("RGB")

            image = image.copy()

        width, height = image.size

        # Convert image into a NumPy array
        image_array = np.array(image)

        # Convert RGB to grayscale for quality analysis
        grayscale = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        # Resize large images for faster analysis
        max_dimension = 1000
        analysis_image = grayscale

        if max(grayscale.shape) > max_dimension:

            scale = max_dimension / max(grayscale.shape)

            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))

            analysis_image = cv2.resize(
                grayscale,
                (new_width, new_height)
            )

        # Estimate image blur
        blur_score = float(
            cv2.Laplacian(
                analysis_image,
                cv2.CV_64F
            ).var()
        )

        # Estimate brightness
        brightness = float(
            np.mean(analysis_image)
        )

        # Identify extremely dark or potentially blurry images
        possibly_blurry = blur_score < 100
        possibly_too_dark = brightness < 30

        # Generate SHA-256 content hash
        image_hash = sha256(file_bytes).hexdigest()

        file_size = len(file_bytes)

        return {
            "success": True,
            "filename": filename,

            "format": image_format,
            "original_mode": original_mode,

            "width": width,
            "height": height,
            "original_width": original_width,
            "original_height": original_height,

            "total_pixels": width * height,
            "aspect_ratio": round(width / height, 3),

            "file_size_bytes": file_size,
            "file_size_mb": round(
                file_size / (1024 * 1024), 3
            ),

            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),

            "possibly_blurry": possibly_blurry,
            "possibly_too_dark": possibly_too_dark,

            "image_hash": image_hash,

            "error": None
        }

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        cv2.error
    ) as error:

        return {
            "success": False,
            "filename": filename,
            "error": str(error)
        }
