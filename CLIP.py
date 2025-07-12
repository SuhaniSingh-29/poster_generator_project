import os
import torch
import clip
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Load CLIP model

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def rank_images_with_clip(image_folder, prompt, top_k=5, batch_size=256):
    print(f"Ranking images from '{image_folder}' for prompt: \"{prompt}\"")

    # Load image paths
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    all_scores = []
    all_indices = []

    text_input = clip.tokenize([prompt]).to(device)
    model.eval()

    # Batch processing
    for start in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        end = start + batch_size
        batch_paths = image_paths[start:end]
        images = []

        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(preprocess(image))
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                continue

        if not images:
            continue

        image_input = torch.stack(images).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarities = (100.0 * image_features @ text_features.T).squeeze()

        for i, sim in enumerate(similarities):
            all_scores.append(sim.item())
            all_indices.append(start + i)

    # Sort and get top-K indices
    top_dir = "top_clip_results"
    os.makedirs(top_dir, exist_ok=True)

    topk_indices = sorted(zip(all_scores, all_indices), reverse=True)[:top_k]

    for rank, (score, idx) in enumerate(topk_indices):
        try:
            img = Image.open(image_paths[idx]).convert("RGB")
            img.save(os.path.join(top_dir, f"rank{rank+1}_{score:.2f}.png"))
        except Exception as e:
            print(f"Failed to save top image {idx}: {e}")

    print(f"Top {top_k} results saved to '{top_dir}'")


# Example usage:
if __name__ == "__main__":
    rank_images_with_clip(
        image_folder="dataset/poster",
        prompt="A modern sci-fi movie poster with blue neon lights",
        top_k=5,
        batch_size=256
    )
