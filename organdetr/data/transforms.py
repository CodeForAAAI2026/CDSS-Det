"""Transformations for different operations."""

import numpy as np
import random
import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Resized,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandRotated,
    RandZoomd,
    RandAffined,
    RandFlipd,
    ToTensord,
    Transform
)

# Function to calculate roi_size based on scale and aspect ratio
def calculate_roi_size(volume_shape, scale, aspect_ratio_range=None):
    D, H, W = volume_shape[-3:]
    #scale = random.uniform(*scale_range)
    new_D = int(D * scale)
    if aspect_ratio_range is None:
        new_H = int(H * scale)
        new_W = int(W * scale)
    else:
        aspect_ratio = random.uniform(*aspect_ratio_range)
        new_H = int(H * scale * aspect_ratio)
        new_W = int(W * scale / aspect_ratio)
    
    # Ensure the sizes are within the bounds of the original volume
    new_D = min(new_D, D)
    new_H = min(new_H, H)
    new_W = min(new_W, W)
    
    return (new_D, new_H, new_W)

def crop_air(x):
    # To not crop fat which is -120 to -90
    return x > -500

def transform_preprocessing(
    margin, crop_key, orientation, resize_shape
):
    transform_list = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=orientation),
        CropForegroundd(
            keys=["image", "label"], source_key=crop_key, 
            margin=margin, select_fn=crop_air
        ),
        Resized(
            keys=['image', 'label'], spatial_size=resize_shape,
            mode=['area', 'nearest']
        )
    ]

    return Compose(transform_list)

'''
class RandSpatialCropSamplesdWithCoords(Transform):
    def __init__(self, keys, roi_size, num_samples=1, random_center=False, random_size=False):
        self.transform = RandSpatialCropSamplesd(keys, roi_size, num_samples, random_center, random_size)
        self.keys = keys

    def __call__(self, data):
        results = self.transform(data)
        for result in results:
            for key in self.keys:
                result[f'{key}_coords'] = result['image_meta_dict']['affine'] @ np.array(
                    [[0, 0, 0, 1],
                     [result['image'].shape[-1], 0, 0, 1],
                     [0, result['image'].shape[-2], 0, 1],
                     [0, 0, result['image'].shape[-3], 1]]
                ).T[:3, :]
        return results
'''

class RandSpatialCropSamplesdWithCoords(Transform):
    def __init__(self, keys, roi_size, num_samples=1, random_center=False, random_size=False):
        self.transform = RandSpatialCropSamplesd(keys, roi_size, num_samples, random_center, random_size)
        self.keys = keys

    def __call__(self, data):
        results = self.transform(data)
        for result in results:
            for key in self.keys:
                shape = result[key].shape[-3:]  # Get the shape of the 3D image
                result[f'{key}_coords'] = torch.tensor([
                    [0, 0, 0],
                    [shape[0], 0, 0],
                    [0, shape[1], 0],
                    [0, 0, shape[2]]
                ])
        return results

class RandSpatialCropdWithCoords(RandSpatialCropd):
    def __init__(self, keys, roi_size, random_center=True, random_size=False):
        super().__init__(keys=keys, roi_size=roi_size, random_center=random_center, random_size=random_size)

    def __call__(self, data):
        d = dict(data)
        cropped_data = super().__call__(d)
        _, D, H, W = data['image'].shape
        s = self.cropper._slices
        # D, H, W
        # cropped_data['coords'] = (s[2].start, s[1].start, s[0].start, s[2].stop, s[1].stop, s[0].stop)
        # print(f"*******************************D={D}, H={H}, W={W}, ")
        # print(f"*******************************center_x={(s[2].stop - s[2].start) / 2}, center_y={(s[1].stop - s[1].start) / 2}, center_y={(s[0].stop - s[0].start) / 2}")
        cropped_data['coords'] = (s[2].start / W, s[1].start / H, s[0].start / D, s[2].stop / W, s[1].stop / H, s[0].stop / D)
        # print("coords: ", cropped_data['coords'])
        return cropped_data

def get_pixpro_transform(config, strong=True):
    if strong:
        transform = [
            # Scale and clip intensity values
            ScaleIntensityRanged(
                # keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
                keys=['image'], a_min=config['foreground_voxel_statistics']['percentile_00_5'], 
                a_max=config['foreground_voxel_statistics']['percentile_99_5'], b_min=0.0, b_max=1.0, clip=True
            ),

            # Intensity transformations
            RandGaussianNoised(
                keys=['image'], prob=config['augmentation']['p_gaussian_noise'], 
                mean=config['augmentation']['gaussian_noise_mean'], std=config['augmentation']['gaussian_noise_std']
            ),
            RandGaussianSmoothd(
                keys=['image'], prob=config['augmentation']['p_gaussian_smooth'],
                sigma_x=config['augmentation']['gaussian_smooth_sigma'], 
                sigma_y=config['augmentation']['gaussian_smooth_sigma'],
                sigma_z=config['augmentation']['gaussian_smooth_sigma'],
            ),
            RandScaleIntensityd(
                keys=['image'], prob=config['augmentation']['p_intensity_scale'],
                factors=config['augmentation']['intensity_scale_factors']
            ),
            RandShiftIntensityd(
                keys=['image'], prob=config['augmentation']['p_intensity_shift'],
                offsets=config['augmentation']['intensity_shift_offsets']
            ),
            RandAdjustContrastd(
                keys=['image'], prob=config['augmentation']['p_adjust_contrast'],
                gamma=config['augmentation']['adjust_contrast_gamma']
            )
        ]
    else:
        transform = [
            # Scale and clip intensity values
            ScaleIntensityRanged(
                # keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
                keys=['image'], a_min=config['foreground_voxel_statistics']['percentile_00_5'], 
                a_max=config['foreground_voxel_statistics']['percentile_99_5'], b_min=0.0, b_max=1.0, clip=True
            )
        ]

    # roi_size = calculate_roi_size(patch_size, config['augmentation']['crop_size']) 
    # transform.append(RandSpatialCropdWithCoords(keys=['image'], roi_size=roi_size, random_center=True, random_size=False))
    transform.append(RandSpatialCropdWithCoords(keys=['image'], roi_size=config['augmentation']['crop_size'], random_center=True, random_size=False))

    # Convert to torch.Tensor
    transform.append(
        ToTensord(
            keys=['image', "coords"]
        )
    )

    return Compose(transform)

def get_transforms(split, config):
    rotate_range = [i / 180 * np.pi for i in config['augmentation']['rotation']]
    translate_range = [(i * config['augmentation']['translate_precentage']) / 100 for i in config['shape_statistics']['median']]
    
    if config['augmentation']['patch_size'] is None:
        patch_size = config['shape_statistics']['median']
    else:
        patch_size = config['augmentation']['patch_size']

    if split == 'train' or split == 'labeled_train' or split == 'unlabeled_train':
        transform = [
            # Scale and clip intensity values
            ScaleIntensityRanged(
                # keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
                keys=['image'], a_min=config['foreground_voxel_statistics']['percentile_00_5'], 
                a_max=config['foreground_voxel_statistics']['percentile_99_5'], b_min=0.0, b_max=1.0, clip=True
            ),

            # Spatial transformations
            # Resized(        # Resize
            #     keys=['image', 'label'], spatial_size=config['shape_statistics']['median'],
            #     mode=['area', 'nearest']
            # ),
            RandRotated(    # Rotation    
                keys=['image', 'label'], prob=config['augmentation']['p_rotate'],
                range_x=rotate_range, range_y=rotate_range, range_z=rotate_range,
                mode=['bilinear', 'nearest'], padding_mode='zeros'
            ),
            RandZoomd(      # Zoom
                keys=['image', 'label'], prob=config['augmentation']['p_zoom'],
                min_zoom=config['augmentation']['min_zoom'],
                max_zoom=config['augmentation']['max_zoom'],
                mode=['area', 'nearest'], padding_mode='constant', constant_values=0
            ),
            RandAffined(    # Translation
                keys=['image', 'label'], prob=config['augmentation']['p_translate'],
                mode=['bilinear', 'nearest'],
                translate_range=translate_range, padding_mode='zeros'
            ), 
            RandAffined(    # Shear
                keys=['image', 'label'], prob=config['augmentation']['p_shear'],
                shear_range=config['augmentation']['shear_range'],
                mode=['bilinear', 'nearest'], padding_mode='zeros'
            ),
            RandFlipd(      # Flip axis 0
                keys=['image', 'label'], prob=config['augmentation']['p_flip'],
                spatial_axis=config['augmentation']['flip_axis'][0]
            ),
            RandFlipd(      # Flip axis 1
                keys=['image', 'label'], prob=config['augmentation']['p_flip'],
                spatial_axis=config['augmentation']['flip_axis'][1]
            ),
            RandFlipd(      # Flip axis 2
                keys=['image', 'label'], prob=config['augmentation']['p_flip'],
                spatial_axis=config['augmentation']['flip_axis'][2]
            ),
            RandSpatialCropd(
                keys=['image', 'label'],
                roi_size=patch_size,
                random_size=False, random_center=True
            ),

            # Intensity transformations
            RandGaussianNoised(
                keys=['image'], prob=config['augmentation']['p_gaussian_noise'], 
                mean=config['augmentation']['gaussian_noise_mean'], std=config['augmentation']['gaussian_noise_std']
            ),
            RandGaussianSmoothd(
                keys=['image'], prob=config['augmentation']['p_gaussian_smooth'],
                sigma_x=config['augmentation']['gaussian_smooth_sigma'], 
                sigma_y=config['augmentation']['gaussian_smooth_sigma'],
                sigma_z=config['augmentation']['gaussian_smooth_sigma'],
            ),
            RandScaleIntensityd(
                keys=['image'], prob=config['augmentation']['p_intensity_scale'],
                factors=config['augmentation']['intensity_scale_factors']
            ),
            RandShiftIntensityd(
                keys=['image'], prob=config['augmentation']['p_intensity_shift'],
                offsets=config['augmentation']['intensity_shift_offsets']
            ),
            RandAdjustContrastd(
                keys=['image'], prob=config['augmentation']['p_adjust_contrast'],
                gamma=config['augmentation']['adjust_contrast_gamma']
            ),

            # Convert to torch.Tensor
            ToTensord(
                keys=['image', 'label']
            )
        ]

    elif split == 'val':
        transform = [
            ScaleIntensityRanged(
                # keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
                keys=['image'], a_min=config['foreground_voxel_statistics']['percentile_00_5'], 
                a_max=config['foreground_voxel_statistics']['percentile_99_5'], b_min=0.0, b_max=1.0, clip=True
            ),
            # Resized(
            #     keys=['image', 'label'], spatial_size=config['shape_statistics']['median'],
            #     mode=['area', 'nearest']
            # ),
            RandSpatialCropd(
                keys=['image', 'label'], roi_size=patch_size,
                random_size=False, random_center=True
            ),
            ToTensord(
                keys=['image', 'label']
            )
        ]
    elif split == 'test':
        transform = [
            ScaleIntensityRanged(
                # keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
                keys=['image'], a_min=config['foreground_voxel_statistics']['percentile_00_5'], 
                a_max=config['foreground_voxel_statistics']['percentile_99_5'], b_min=0.0, b_max=1.0, clip=True
            ),
            # Resized(
            #     keys=['image', 'label'], spatial_size=config['shape_statistics']['median'],
            #     mode=['area', 'nearest']
            # ),
            RandSpatialCropd(
                keys=['image', 'label'], roi_size=patch_size,
                random_size=False, random_center=True
            ),
            ToTensord(
                keys=['image', 'label']
            )
        ]
    else:
        raise ValueError("Please use 'test', 'val', or 'train' as split arg.")
    return Compose(transform)

def get_weak_transforms(split, config, keys=['image']):
    rotate_range = [i / 180 * np.pi for i in config['augmentation']['rotation']]
    translate_range = [(i * config['augmentation']['translate_precentage']) / 100 for i in config['shape_statistics']['median']]

    if config['augmentation']['patch_size'] is None:
        patch_size = config['shape_statistics']['median']
    else:
        patch_size = config['augmentation']['patch_size']

    if split == 'train' or split == 'labeled_train' or split == 'unlabeled_train':
        transform = [
            # Scale and clip intensity values
            ScaleIntensityRanged(
                # keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
                keys=['image'], a_min=config['foreground_voxel_statistics']['percentile_00_5'], 
                a_max=config['foreground_voxel_statistics']['percentile_99_5'], b_min=0.0, b_max=1.0, clip=True
            ),
            # Spatial transformations
            # Resized(        # Resize
            #     keys=['image', 'label'], spatial_size=config['shape_statistics']['median'],
            #     mode=['area', 'nearest']
            # ),
            RandRotated(    # Rotation    
                keys=keys, prob=config['augmentation']['p_rotate'],
                range_x=rotate_range, range_y=rotate_range, range_z=rotate_range,
                mode=['bilinear', 'nearest'], padding_mode='zeros'
            ),
            RandZoomd(      # Zoom
                keys=keys, prob=config['augmentation']['p_zoom'],
                min_zoom=config['augmentation']['min_zoom'],
                max_zoom=config['augmentation']['max_zoom'],
                mode=['area', 'nearest'], padding_mode='constant', constant_values=0
            ),
            RandAffined(    # Translation
                keys=keys, prob=config['augmentation']['p_translate'],
                mode=['bilinear', 'nearest'],
                translate_range=translate_range, padding_mode='zeros'
            ), 
            RandAffined(    # Shear
                keys=keys, prob=config['augmentation']['p_shear'],
                shear_range=config['augmentation']['shear_range'],
                mode=['bilinear', 'nearest'], padding_mode='zeros'
            ),
            RandFlipd(      # Flip axis 0
                keys=keys, prob=config['augmentation']['p_flip'],
                spatial_axis=config['augmentation']['flip_axis'][0]
            ),
            RandFlipd(      # Flip axis 1
                keys=keys, prob=config['augmentation']['p_flip'],
                spatial_axis=config['augmentation']['flip_axis'][1]
            ),
            RandFlipd(      # Flip axis 2
                keys=keys, prob=config['augmentation']['p_flip'],
                spatial_axis=config['augmentation']['flip_axis'][2]
            ),
            RandSpatialCropd(
                keys=keys,
                roi_size=patch_size,
                random_size=False, random_center=True
            ),

            # Convert to torch.Tensor
            ToTensord(
                keys=keys
            )
        ]

    elif split == 'val':
        transform = [
            ScaleIntensityRanged(
                # keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
                keys=['image'], a_min=config['foreground_voxel_statistics']['percentile_00_5'], 
                a_max=config['foreground_voxel_statistics']['percentile_99_5'], b_min=0.0, b_max=1.0, clip=True
            ),
            # Resized(
            #     keys=['image', 'label'], spatial_size=config['shape_statistics']['median'],
            #     mode=['area', 'nearest']
            # ),
            RandSpatialCropd(
                keys=['image'], roi_size=patch_size,
                random_size=False, random_center=True
            ),
            ToTensord(
                keys=keys
            )
        ]
    elif split == 'test':
        transform = [
            ScaleIntensityRanged(
                # keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
                keys=['image'], a_min=config['foreground_voxel_statistics']['percentile_00_5'], 
                a_max=config['foreground_voxel_statistics']['percentile_99_5'], b_min=0.0, b_max=1.0, clip=True
            ),
            # Resized(
            #     keys=keys, spatial_size=config['shape_statistics']['median'],
            #     mode=['area', 'nearest']
            # ),
            RandSpatialCropd(
                keys=keys, roi_size=patch_size,
                random_size=False, random_center=True
            ),
            ToTensord(
                keys=keys
            )
        ]
    else:
        raise ValueError("Please use 'test', 'val', or 'train' as split arg.")
    return Compose(transform)