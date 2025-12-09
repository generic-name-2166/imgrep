from pathlib import Path
import sys

from transformers import BlipProcessor, BlipForImageTextRetrieval
import torch
from PIL import Image


def has_text(processor, model, image_path: Path, search_text: str):
    image = Image.open(image_path).convert("RGB")  # Ensure RGB mode
    inputs = processor(text=search_text, images=image, return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        itm_output = model(**inputs)
        itm_logits = itm_output[0]  # Shape: (1, 2)
        probs = torch.softmax(itm_logits, dim=1)
        match_score = probs[0][1].item()
    return match_score


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python script.py <image_or_dir> <search_text>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    search_text = sys.argv[2]

    processor = BlipProcessor.from_pretrained("models/", use_fast=True)
    model = BlipForImageTextRetrieval.from_pretrained("models/")

    if input_path.is_file():
        # Single image
        score = has_text(processor, model, input_path, search_text)
        print(f"{input_path.name}: {score:.3f}")
    elif input_path.is_dir():
        # Directory of images (recursive, common formats)
        image_paths = (
            list(input_path.rglob("*.[jJ][pP][gG]"))
            + list(input_path.rglob("*.[pP][nN][gG]"))
            + list(input_path.rglob("*.[jJ][pP][eE][gG]"))
        )

        if not image_paths:
            print("No images found in directory.")
            return

        print(f"Processing {len(image_paths)} images for '{search_text}':")
        for img_path in image_paths:
            score = has_text(processor, model, img_path, search_text)
            print(f"{img_path.name}: {score:.3f}")
    else:
        print("Error: Path is neither a file nor directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
