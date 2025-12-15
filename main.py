from pathlib import Path
import sys
import argparse

from transformers import BlipProcessor, BlipForImageTextRetrieval
import torch
from PIL import Image
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn


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
    parser = argparse.ArgumentParser(
        description="Search for text in images using BLIP model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py image.jpg "cat"
  python main.py images/ "sunset" --threshold 0.5 --format table
        """
    )
    parser.add_argument("path", help="Path to image file or directory")
    parser.add_argument("text", help="Text to search for in images")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum score threshold to display results (default: 0.0)"
    )
    parser.add_argument(
        "--format",
        choices=["simple", "table"],
        default="simple",
        help="Output format (default: simple)"
    )

    args = parser.parse_args()

    input_path = Path(args.path)
    search_text = args.text
    threshold = args.threshold
    output_format = args.format

    console = Console()

    with console.status("[bold green]Loading model...[/bold green]"):
        processor = BlipProcessor.from_pretrained("models/", use_fast=True)
        model = BlipForImageTextRetrieval.from_pretrained("models/")

    results = []

    if input_path.is_file():
        # Single image
        score = has_text(processor, model, input_path, search_text)
        if score >= threshold:
            results.append((input_path.name, score))
    elif input_path.is_dir():
        # Directory of images (recursive, common formats)
        image_paths = (
            list(input_path.rglob("*.[jJ][pP][gG]"))
            + list(input_path.rglob("*.[pP][nN][gG]"))
            + list(input_path.rglob("*.[jJ][pP][eE][gG]"))
        )

        if not image_paths:
            console.print("[red]No images found in directory.[/red]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Processing {len(image_paths)} images for '{search_text}'...", total=len(image_paths))

            for img_path in image_paths:
                score = has_text(processor, model, img_path, search_text)
                if score >= threshold:
                    results.append((img_path.name, score))
                progress.advance(task)
    else:
        console.print("[red]Error: Path is neither a file nor directory.[/red]")
        sys.exit(1)

    if not results:
        console.print("[yellow]No matches found above threshold.[/yellow]")
        return

    if output_format == "table":
        table = Table(title=f"Image Text Search Results for '{search_text}'")
        table.add_column("Image", style="cyan", no_wrap=True)
        table.add_column("Score", style="magenta", justify="right")

        for name, score in results:
            table.add_row(name, f"{score:.3f}")

        console.print(table)
    else:
        for name, score in results:
            console.print(f"{name}: {score:.3f}")



if __name__ == "__main__":
    main()
