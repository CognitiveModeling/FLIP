import numpy as np
import json
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from PIL import Image
import cv2
from tqdm import tqdm
import os
import h5py
import argparse
import torch
import torch as th
from torchvision.transforms import ToTensor
import torch.nn.functional as F

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STR_DTYPE = h5py.string_dtype(encoding="utf-8")  # reusable UTF-8 vlen string

def compress_image(image, format='.jpg'):
    """Compress image using OpenCV encoding"""
    if len(image.shape) == 3:
        if format == '.jpg':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    is_success, buffer = cv2.imencode(format, image)
    if is_success:
        return np.array(buffer)
    else:
        raise Exception("Failed to compress image")

def compute_bbox_from_mask(mask):
    """Compute bounding box from binary mask - compatible with existing scripts"""
    assert len(mask.shape) == 2, f"Mask must be 2D, but got shape {mask.shape}"

    # Check if the mask has any valid pixels
    if not np.any(mask > 0):
        return None

    height, width = mask.shape
    x_range = np.linspace(0, width - 1, width)
    y_range = np.linspace(0, height - 1, height)
    x_coords, y_coords = np.meshgrid(x_range, y_range)

    mask = (mask > 0.75).astype(np.float32)
    x_masked = x_coords * mask
    y_masked = y_coords * mask

    x_min = np.min(np.where(x_masked > 0, x_masked, np.inf))
    y_min = np.min(np.where(y_masked > 0, y_masked, np.inf))
    x_max = np.max(np.where(x_masked > 0, x_masked, -np.inf))
    y_max = np.max(np.where(y_masked > 0, y_masked, -np.inf))

    bbox = np.array([[x_min, y_min, x_max, y_max]])
    return bbox

def compute_gaussian_params_from_mask(mask):
    """
    Compute Gaussian parameters (mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy) from binary mask.
    
    Args:
        mask: Binary mask tensor of shape (B, C, H, W)
    
    Returns:
        Tensor of shape (B, 5) containing [mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy]
    """
    B, C, height, width = mask.shape
    mask = (mask > 0.5).float()
    
    # Create a meshgrid normalized to [-1, 1]
    x_range = th.linspace(-1, 1, width, device=mask.device)
    y_range = th.linspace(-1, 1, height, device=mask.device)
    x_grid, y_grid = th.meshgrid(x_range, y_range, indexing='xy')
    y_grid = y_grid.expand(B, 1, height, width)
    x_grid = x_grid.expand(B, 1, height, width)

    # Filter the coordinates using element-wise multiplication with the binary mask
    x = x_grid * mask
    y = y_grid * mask

    # Compute mean
    sum_mask = th.clip(th.sum(mask, dim=[2, 3], keepdim=True), min=1e-16)
    mu_x = th.sum(x, dim=[2, 3], keepdim=True) / sum_mask
    mu_y = th.sum(y, dim=[2, 3], keepdim=True) / sum_mask

    mu_x2 = th.sum(x**2, dim=[2, 3], keepdim=True) / sum_mask
    mu_y2 = th.sum(y**2, dim=[2, 3], keepdim=True) / sum_mask
    mu_xy = th.sum(x*y, dim=[2, 3], keepdim=True) / sum_mask
    
    # Remove spatial dimensions
    mu_x = mu_x[:, 0, 0, 0]
    mu_y = mu_y[:, 0, 0, 0]
    mu_x2 = mu_x2[:, 0, 0, 0]
    mu_y2 = mu_y2[:, 0, 0, 0]
    mu_xy = mu_xy[:, 0, 0, 0]

    # Compute variance and covariance
    sigma_x2 = mu_x2 - mu_x**2
    sigma_y2 = mu_y2 - mu_y**2
    sigma_xy = mu_xy - mu_x * mu_y

    return th.stack((mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy), dim=1)

class HDF5Dataset:
    def __init__(self, root_path: str, dataset_name: str, type: str, license_table: list):
        
        instance_counter = 1
        hdf5_file_path = os.path.join(root_path, f'{dataset_name}-{type}-v1.hdf5')
        while os.path.exists(hdf5_file_path):
            instance_counter += 1
            hdf5_file_path = os.path.join(root_path, f'{dataset_name}-{type}-v{instance_counter}.hdf5')

        print(f"Creating HDF5 dataset at: {hdf5_file_path}")

        # Setup the hdf5 file
        hdf5_file = h5py.File(hdf5_file_path, "w")

        # Create datasets with same structure as existing scripts
        hdf5_file.create_dataset(
            "rgb_images",   
            (0, ),
            maxshape=(None, ),
            dtype=h5py.vlen_dtype(np.dtype('uint8')),
        )
        hdf5_file.create_dataset(
            "instance_masks", 
            (0, ),
            maxshape=(None, ),
            dtype=h5py.vlen_dtype(np.dtype('uint8')),
        )
        hdf5_file.create_dataset(
            "sequence_indices",
            (0, 2), # start index, number of images
            maxshape=(None, 2),
            dtype=np.int64,
            compression='gzip',
            compression_opts=5,
        )
        hdf5_file.create_dataset(
            "image_instance_indices",
            (0, 2), # start index, number of instances
            maxshape=(None, 2),
            dtype=np.int64,
            compression='gzip',
            compression_opts=5,
        )
        hdf5_file.create_dataset(
            "instance_masks_images", 
            (0, 1), 
            maxshape=(None, 1),
            compression='gzip',
            compression_opts=5,
            dtype=np.int64,
        )
        hdf5_file.create_dataset(
            "instance_mask_bboxes", 
            (0, 4), 
            maxshape=(None, 4), 
            compression='gzip',
            compression_opts=5,
            dtype=np.float32, 
        )
        hdf5_file.create_dataset(
            "coco_image_ids",
            (0,),                   # 1-D vector: one id per jpeg
            maxshape=(None,),
            dtype=np.int64,         # COCO ids fit in 64-bit
        )
        hdf5_file.create_dataset(
            "license_ids",
            (0,),                   # licence id 1-8 from the JSON
            maxshape=(None,),
            dtype=np.uint8,         # fits in one byte
        )
        hdf5_file.create_dataset(
            "positions", 
            (0, 5),                 # [mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy]
            maxshape=(None, 5),
            dtype=np.float32,
            compression='gzip',
            compression_opts=5,
        )
        hdf5_file.create_dataset(
            "file_names",
            (0,),
            maxshape=(None,),
            dtype=STR_DTYPE,
        )
        hdf5_file.create_dataset(
            "flickr_urls",
            (0,),
            maxshape=(None,),
            dtype=STR_DTYPE,
        )
        hdf5_file.create_dataset(
            "coco_urls",
            (0,),
            maxshape=(None,),
            dtype=STR_DTYPE,
        )
        # per-image “indicate changes” note – constant string is fine
        hdf5_file.create_dataset(
            "transform_notes",
            (0,),
            maxshape=(None,),
            dtype=STR_DTYPE,
        )

        # Create a metadata group and set the attributes
        metadata_grp = hdf5_file.create_group("metadata")
        metadata_grp.attrs["dataset_name"] = dataset_name
        metadata_grp.attrs["type"] = type
        metadata_grp.attrs["coco_licenses"] = json.dumps(license_table)

        self.hdf5_file = hdf5_file

    def close(self):
        self.hdf5_file.flush()
        self.hdf5_file.close()

    def __getitem__(self, index):
        return self.hdf5_file[index]

    def append_data(self, index, item):
        self[index].resize((self[index].shape[0] + item.shape[0], *item.shape[1:]))
        self[index][-item.shape[0]:] = item

    def append_image(self, index, item):
        self[index].resize((self[index].shape[0] + 1,))
        self[index][-1] = item

def polygon_to_mask(polygon_list, height, width):
    """Convert COCO polygon(s) to binary mask"""
    # Create RLE from polygon(s) - polygon_list is already a list
    rle = coco_mask.frPyObjects(polygon_list, height, width)
    # If multiple polygons, merge them into single mask
    if len(rle) > 1:
        rle = coco_mask.merge(rle)
    else:
        rle = rle[0]
    # Decode RLE to binary mask
    mask = coco_mask.decode(rle)
    return mask

def rle_to_mask(rle, height, width):
    """Convert COCO RLE to binary mask"""
    if isinstance(rle['counts'], list):
        # Uncompressed RLE
        rle = coco_mask.frPyObjects([rle], height, width)[0]
    # Decode RLE to binary mask
    mask = coco_mask.decode(rle)
    return mask

def process_coco_annotations(coco, img_info):
    """Process all annotations for a single COCO image"""
    image_id = img_info['id']
    height = img_info['height']
    width = img_info['width']
    
    # Get all annotations for this image
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    
    masks = []
    bboxes = []
    positions = []
    
    for ann in anns:
        # Skip crowd annotations
        if ann.get('iscrowd', 0):
            continue
            
        # Convert annotation to mask
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):
                # Polygon format - filter out empty polygons
                valid_polygons = [poly for poly in ann['segmentation'] if len(poly) >= 6]  # Need at least 3 points (6 coordinates)
                if len(valid_polygons) > 0:
                    try:
                        mask = polygon_to_mask(valid_polygons, height, width)
                    except Exception as e:
                        print(f"Error converting polygon to mask for annotation {ann['id']}: {e}")
                        continue
                else:
                    continue
            elif isinstance(ann['segmentation'], dict):
                # RLE format
                try:
                    mask = rle_to_mask(ann['segmentation'], height, width)
                except Exception as e:
                    print(f"Error converting RLE to mask for annotation {ann['id']}: {e}")
                    continue
            else:
                continue
                
            # Validate mask
            if mask is None or mask.size == 0 or not np.any(mask > 0):
                continue
                
            # Compute bounding box from mask
            bbox = compute_bbox_from_mask(mask)
            if bbox is not None:
                # Compute Gaussian parameters
                try:
                    # Convert to torch tensor for Gaussian computation
                    mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(DEVICE)
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                    
                    with torch.no_grad():
                        gaussian_params = compute_gaussian_params_from_mask(mask_tensor)
                        gaussian_params = gaussian_params.cpu().numpy()[0]  # Remove batch dimension
                    
                    masks.append(mask)
                    bboxes.append(bbox[0])  # Remove extra dimension
                    positions.append(gaussian_params)
                    
                except Exception as e:
                    print(f"Error computing Gaussian parameters for annotation {ann['id']}: {e}")
                    continue
    
    if len(masks) == 0:
        return None, None, None
        
    return np.array(masks), np.array(bboxes), np.array(positions)

def main():
    parser = argparse.ArgumentParser(description='Create HDF5 dataset from COCO annotations')
    parser.add_argument('--coco_root', required=True, help='Path to COCO dataset root directory')
    parser.add_argument('--annotation_file', required=True, help='Path to COCO annotation JSON file')
    parser.add_argument('--output_dir', required=True, help='Output directory for HDF5 file')
    parser.add_argument('--split', default='val2017', help='Dataset split name (e.g., val, train)')
    
    args = parser.parse_args()
    
    print(f"Using device: {DEVICE}")
    
    # Initialize COCO API
    print(f"Loading COCO annotations from {args.annotation_file}")
    coco = COCO(args.annotation_file)
    with open(args.annotation_file, 'r') as jf:
        annotation_blob = json.load(jf)          # keep a copy for the licence table

    license_table = annotation_blob["licenses"]  # list[dict] length = 8
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    print(f"Found {len(img_ids)} images")
    
    # Create HDF5 dataset
    dataset_name = f'coco-{args.split}'
    dataset = HDF5Dataset(
        root_path=args.output_dir,
        dataset_name=dataset_name,
        type=args.split,
        license_table=license_table  
    )
    
    processed_images = 0
    skipped_images = 0
    
    for img_id in tqdm(img_ids, desc="Processing COCO images"):
        try:
            # Get image info
            img_info = coco.loadImgs(img_id)[0]
            img_filename = img_info['file_name']
            img_path = os.path.join(args.coco_root, img_filename)
            
            # Check if image exists
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                skipped_images += 1
                continue
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            # Process annotations for this image
            masks, bboxes, positions = process_coco_annotations(coco, img_info)
            
            if masks is None:
                skipped_images += 1
                continue
            
            # Store image index for sequence tracking
            image_idx = dataset['rgb_images'].shape[0]
            mask_start_idx = dataset['instance_masks'].shape[0]
            num_instances = len(masks)

            image_id       = img_info['id']
            license_id     = img_info['license']     # integer 1–8
            image_idx      = dataset['rgb_images'].shape[0]
            
            # Add compressed image
            dataset.append_image('rgb_images', compress_image(image, '.jpg'))
            
            # add license and attribution data
            dataset.append_data('coco_image_ids',  np.array([image_id],                               dtype=np.int64))
            dataset.append_data('license_ids',     np.array([license_id],                             dtype=np.uint8))
            dataset.append_data('file_names',      np.array([img_info['file_name']],                  dtype=STR_DTYPE))
            dataset.append_data('flickr_urls',     np.array([img_info.get('flickr_url', '')],         dtype=STR_DTYPE))
            dataset.append_data('coco_urls',       np.array([img_info.get('coco_url', '')],           dtype=STR_DTYPE))
            dataset.append_data('transform_notes', np.array(["recompressed JPEG; no creative edits"], dtype=STR_DTYPE))
            
            # Add compressed masks and their metadata
            for i, (mask, bbox, position) in enumerate(zip(masks, bboxes, positions)):
                dataset.append_image('instance_masks', compress_image(mask.astype(np.uint8) * 255, '.png'))
                dataset.append_data('instance_mask_bboxes', bbox.reshape(1, 4))
                dataset.append_data('instance_masks_images', np.array([[image_idx]]))
                dataset.append_data('positions', position.reshape(1, 5))
            
            # Add image instance indices (start index and count for this image)
            dataset.append_data('image_instance_indices', 
                              np.array([[mask_start_idx, num_instances]]))
            
            # Add sequence indices (treat each image as single frame sequence)
            dataset.append_data('sequence_indices', 
                              np.array([[image_idx, 1]]))
            
            processed_images += 1
            
            # Print progress every 1000 images
            if processed_images % 1000 == 0:
                print(f"Processed {processed_images} images, skipped {skipped_images}")
                
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            skipped_images += 1
            continue
    
    print(f"Finished! Processed {processed_images} images, skipped {skipped_images}")
    dataset.close()

if __name__ == '__main__':
    main()
