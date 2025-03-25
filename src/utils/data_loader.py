import os
import json
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import TRAIN_ANNOTATIONS, TRAIN_DIR, VALID_ANNOTATIONS, VALID_DIR
from mask_utils import polygons_to_mask  # for converting polygons to mask

# Custom dataset class to load image and full mask
class PetRadiologyDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load COCO JSON
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        self.images = coco_data["images"]
        self.annotations = coco_data["annotations"]

        # Group annotations by image_id
        self.ann_by_image = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.ann_by_image:
                self.ann_by_image[img_id] = []
            self.ann_by_image[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info["id"]
        filename = image_info["file_name"]

        # Load image
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Build empty mask
        full_mask = np.zeros((height, width), dtype=np.uint8)

        # Load and combine masks from all annotations for this image
        annotations = self.ann_by_image.get(image_id, [])
        for ann in annotations:
            segmentation = ann.get("segmentation", [])
            mask = polygons_to_mask(segmentation, (height, width))
            full_mask = np.maximum(full_mask, mask)  # combine masks

        # Convert to PIL Image for transforms
        mask_pil = Image.fromarray(full_mask)

        # Apply transforms (resize both image and mask the same way)
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask_pil)  # Convert mask to 1xHxW tensor

        return image, mask

# ✅ Define transform once
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def get_dataloader(image_dir, annotation_file, batch_size=8, shuffle=True):
    dataset = PetRadiologyDataset(image_dir, annotation_file, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.transforms.functional import to_pil_image

    dataset = PetRadiologyDataset(TRAIN_DIR, TRAIN_ANNOTATIONS, transform=transform)

    image, mask = dataset[0]  # mask will be [1, H, W]

    # Show image
    plt.imshow(to_pil_image(image))
    plt.title("Image")
    plt.axis("off")
    plt.show()

    # Show mask
    plt.imshow(mask.squeeze(0), cmap="gray")
    plt.title("Segmentation Mask")
    plt.axis("off")
    plt.show()
