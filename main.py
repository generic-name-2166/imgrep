from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
import sys


def describe_image(processor, model, image_path, prompt: str = "Describe this image:") -> str:
    image = Image.open(image_path)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(generated[0], skip_special_tokens=True)


def main() -> None:
    processor = AutoProcessor.from_pretrained("models/")
    model = AutoModelForVision2Seq.from_pretrained("models/")
    result = describe_image(processor, model, sys.argv[1])
    print(result)

if __name__ == "__main__":
    main()
