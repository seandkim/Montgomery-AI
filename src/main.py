# src/main.py

from models.sam.sam_inference import SAMWrapper

def main():
    # Initialize SAM
    sam = SAMWrapper(
        model_variant="hiera_small",
        checkpoint_path="path/to/hiera_small_checkpoint.pth",
        device="mps"  # or "cpu", "cuda" if applicable
    )
    
    # Load image
    sam.load_image("data/raw/image.jpg")
    
    # Define prompts (e.g., points, boxes)
    prompts = [
        {"type": "point", "coordinates": (100, 200)},
        {"type": "box", "coordinates": (50, 50, 150, 150)}
    ]
    
    # Perform prediction
    masks, scores = sam.predict(prompts)
    
    # Process masks as needed
    # ...

if __name__ == "__main__":
    main()