import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import TRAIN_ANNOTATIONS, TRAIN_DIR, VALID_ANNOTATIONS, VALID_DIR
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib.pyplot as plt

# Custom dataset class to load images and their annotation info from COCO format
class PetRadiologyDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir                  # Path to image folder
        self.transform = transform                  # Optional image transformations

        # Load annotation JSON file (COCO format)
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        self.images = coco_data["images"]           # List of image metadata
        self.annotations = coco_data["annotations"] # List of annotation objects

        # Organize annotations by image_id for fast lookup
        self.ann_by_image = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.ann_by_image:
                self.ann_by_image[img_id] = []
            self.ann_by_image[img_id].append(ann)

    def __len__(self):
        return len(self.images)  # Total number of images

    def __getitem__(self, idx):
        # Get image metadata and ID
        image_info = self.images[idx]
        image_id = image_info["id"]
        filename = image_info["file_name"]

        # Build the image path and load it
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")

        # Load all annotations (segmentations) for this image
        segmentations = self.ann_by_image.get(image_id, [])

        # Apply any transformations to the image (resizing, normalization, etc.)
        if self.transform:
            image = self.transform(image)

        # For now, return image and raw segmentation data (polygons)
        return image, segmentations

# Define basic image transformation: resize + tensor + normalize
transform = transforms.Compose([
    transforms.Resize((256, 256)),         # Resize to a fixed size
    transforms.ToTensor(),                 # Convert image to tensor
    transforms.Normalize(                  # Normalize pixel values
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Wrapper function to create a DataLoader from a folder + annotation file
def get_dataloader(image_dir, annotation_file, batch_size=8, shuffle=True):
    dataset = PetRadiologyDataset(image_dir, annotation_file, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# test data loading

if __name__ == "__main__":
    from torchvision.transforms.functional import to_tensor
    import matplotlib.pyplot as plt

    # Load dataset with NO transform to see the raw image
    dataset = PetRadiologyDataset(TRAIN_DIR, TRAIN_ANNOTATIONS, transform=None)

    # Load first sample (image + segmentation polygons)
    raw_image, segmentations = dataset[0]

    # Show original image (PIL format)
    plt.imshow(raw_image)
    plt.title("Original PIL Image")
    plt.axis("off")
    plt.show()

    # Convert to tensor without normalization
    tensor_image = to_tensor(raw_image)  # shape: [C, H, W]
    plt.imshow(tensor_image.permute(1, 2, 0))  # reshape to [H, W, C] for plotting
    plt.title("Tensor Image (Unnormalized)")
    plt.axis("off")
    plt.show()

    # Print the segmentation polygon data
    print("Segmentation polygons for image 0:")
    print(segmentations)
