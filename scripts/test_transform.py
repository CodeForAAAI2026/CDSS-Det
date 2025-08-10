from monai.transforms import Compose, RandSpatialCropd
import numpy as np

# Create the composed transform
transform = Compose([
    RandSpatialCropd(keys=['image'], roi_size=(64, 64, 64), random_size=False, random_center=True),
    # Add other transforms here
])

# Apply the composed transform
image = np.random.rand(1, 224, 224, 160).astype(np.float32)  # Example 3D image
data = {'image': image}

transformed_data = transform(data)
print(f"transformed_data.shape={transformed_data['image'].shape}")

crop_coords = transform.transforms[0].cropper._slices
print("slices", crop_coords)
print("coords", (crop_coords[0].start, crop_coords[1].start, crop_coords[2].start), (crop_coords[0].stop, crop_coords[1].stop, crop_coords[2].stop))


transformed_data_k = transform(data)
print(f"transformed_data_k.shape={transformed_data_k['image'].shape}")

# Retrieve the coordinates
crop_coords_k = transform.transforms[0].cropper._slices
print("slices", crop_coords_k)