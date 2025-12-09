from transformers import BlipProcessor, BlipForImageTextRetrieval


def main() -> None:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco", use_fast=True)
    print("processor")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to("cuda")
    print("model")

    processor.save_pretrained("models/")
    model.save_pretrained("models/")


if __name__ == "__main__":
    main()
