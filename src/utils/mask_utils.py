import numpy as np
from pycocotools import mask as maskUtils
import PIL.Image

def polygons_to_mask(segmentation, image_size):
    """
    Convert COCO polygon segmentation to a binary mask.

    Args:
        segmentation (list): COCO polygon list (list of lists of x,y points).
        image_size (tuple): (height, width) of the image.

    Returns:
        np.ndarray: Binary mask of shape (H, W) with 1s in the segmented area.
    """
    height, width = image_size

    # Handle the case where segmentation is empty
    if not segmentation:
        return np.zeros((height, width), dtype=np.uint8)

    # Format required by pycocotools
    rles = maskUtils.frPyObjects(segmentation, height, width) #    # Convert the polygon to RLE format
    rle = maskUtils.merge(rles) #merge all the polygons into one mask
    mask = maskUtils.decode(rle) #    # Decode RLE into a binary mask

    return mask.astype(np.uint8)
