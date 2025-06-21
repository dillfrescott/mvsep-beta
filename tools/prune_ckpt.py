import argparse
import torch

def prune_checkpoint(input_path: str, output_path: str):
    # Load the full checkpoint
    ckpt = torch.load(input_path, map_location="cpu")
    
    # Extract only the model weights
    if "model_state_dict" not in ckpt:
        raise KeyError(f"No 'model_state_dict' key found in {input_path}")
    slim_ckpt = {"model_state_dict": ckpt["model_state_dict"]}
    
    # Save the pruned checkpoint
    torch.save(slim_ckpt, output_path)
    print(f"Pruned checkpoint saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Prune a training checkpoint for inference-only usage")
    parser.add_argument(
        "--input_checkpoint",
        "-i",
        type=str,
        required=True,
        help="Path to the original (full) checkpoint .pt file",
    )
    parser.add_argument(
        "--output_checkpoint",
        "-o",
        type=str,
        required=True,
        help="Path where the smaller inference-only checkpoint will be written",
    )
    args = parser.parse_args()

    prune_checkpoint(args.input_checkpoint, args.output_checkpoint)

if __name__ == "__main__":
    main()