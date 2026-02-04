# Visualization Functions

import matplotlib.pyplot as plt
import mask_functions as ms

def get_mask_for_image(rle_string, image_shape):
    """Convert RLE string to mask array"""
    height, width = image_shape
    mask = ms.rle2mask(rle_string, width, height).T
    return mask

def visualize_pneumothorax(image_array, mask, image_id=None):
    """Visualize X-ray with pneumothorax overlay"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_array, cmap='gray')
    axes[0].set_title('Original X-ray')
    axes[0].axis('off')
    
    # Mask only
    axes[1].imshow(mask, cmap='hot')
    axes[1].set_title('Pneumothorax Mask')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image_array, cmap='gray')
    axes[2].imshow(mask, cmap='Reds', alpha=0.3)
    axes[2].set_title('X-ray with Pneumothorax Overlay')
    axes[2].axis('off')
    
    if image_id:
        fig.suptitle(f'Image ID: {image_id}', fontsize=12)
    
    plt.tight_layout()
    plt.show()
