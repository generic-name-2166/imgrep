from transformers import AutoProcessor, AutoModelForVision2Seq


def main() -> None:
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-base")

    processor.save_pretrained("models/")
    model.save_pretrained("models/")


if __name__ == "__main__":
    main()
