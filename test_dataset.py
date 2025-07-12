import os
from dataset_loader import PosterDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Get the root directory (D:\POSTER)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Corrected dataset path
data_dir = os.path.join(ROOT_DIR, "data", "posters", "all")
assert os.path.exists(data_dir), f"Path does not exist: {data_dir}"

# Load dataset
dataset = PosterDataset(data_dir, image_size=256)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Show a batch
for batch in dataloader:
    grid = vutils.make_grid(batch, nrow=4, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
    break
