from transformers import BlipProcessor, BlipForImageTextRetrieval
import torch
from PIL import Image
import sys


def describe_image(processor, model, image_path, prompt: str = "Listen and learn"):
    image = Image.open(image_path)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        itm_output = model(**inputs)
        itm_logits = itm_output[0]  # Shape: (1, 2)
        probs = torch.softmax(itm_logits, dim=1)
        match_score = probs[0][1].item()
    return match_score


def main() -> None:
    processor = BlipProcessor.from_pretrained("models/", use_fast=True)
    model = BlipForImageTextRetrieval.from_pretrained("models/")
    result = describe_image(processor, model, sys.argv[1])
    print(result)

if __name__ == "__main__":
    main()
