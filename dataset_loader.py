# dataset_loader.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PosterDataset(Dataset):
    def __init__(self, image_folder, image_size=256):
        self.image_folder = image_folder
        self.image_paths = [
            os.path.join(image_folder, file)
            for file in os.listdir(image_folder)
            if file.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Convert to [0,1]
            transforms.Normalize([0.5], [0.5])  # Scale to [-1,1] for GANs
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)
