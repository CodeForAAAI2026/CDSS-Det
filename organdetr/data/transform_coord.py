from __future__ import division
import torch
import math
import random
from torchvision.transforms import functional as F
import warnings

from monai.transforms.transform import MapTransform
from monai.transforms.utils import convert_to_dst_type

class RandColorJitterd(MapTransform):
    def __init__(self, keys, prob=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        super().__init__(keys)
        self.prob = prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, data):
        d = dict(data)
        if random.random() < self.prob:
            for key in self.keys:
                img = d[key]
                # Apply brightness jitter
                img = img * (1 + random.uniform(-self.brightness, self.brightness))
                # Apply contrast jitter
                img_mean = img.mean()
                img = (img - img_mean) * (1 + random.uniform(-self.contrast, self.contrast)) + img_mean
                # Apply saturation jitter (not applicable in grayscale images)
                # Apply hue jitter (not applicable in grayscale images)
                img, *_ = convert_to_dst_type(img, img, dtype=img.dtype)
                d[key] = img
        return d

class RandGrayscaled(MapTransform):
    def __init__(self, keys, prob=0.2):
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        if random.random() < self.prob:
            for key in self.keys:
                img = d[key]
                img = img.mean(axis=0, keepdims=True)  # Convert to grayscale
                img, *_ = convert_to_dst_type(img, img, dtype=img.dtype)
                d[key] = img
        return d

class RandSolarized(MapTransform):
    def __init__(self, keys, prob=0.2, threshold=128):
        super().__init__(keys)
        self.prob = prob
        self.threshold = threshold

    def __call__(self, data):
        d = dict(data)
        if random.random() < self.prob:
            for key in self.keys:
                img = d[key]
                img = torch.where(img < self.threshold, img, 255 - img)
                img, *_ = convert_to_dst_type(img, img, dtype=img.dtype)
                d[key] = img
        return d

# Utility function to get the size of a 3D tensor (depth, height, width)
def _get_volume_size(vol):
    if isinstance(vol, torch.Tensor) and vol.dim() == 4:
        return vol.shape[1:]  # (depth, height, width)
    else:
        raise TypeError("Unexpected type {}".format(type(vol)))

class Compose3D(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose3D([
        >>>     transforms.RandomResizedCropCoord3D((10, 10, 10)),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, vol):
        coord = None
        for t in self.transforms:
            if 'RandomResizedCropCoord3D' in t.__class__.__name__:
                vol, coord = t(vol)
            elif 'FlipCoord3D' in t.__class__.__name__:
                vol, coord = t(vol, coord)
            else:
                vol = t(vol)
        return vol, coord

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomHorizontalFlipCoord3D(object):
    """Horizontally flip the given 3D volume randomly with a given probability.

    Args:
        p (float): probability of the volume being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vol, coord):
        """
        Args:
            vol (torch.Tensor): Volume to be flipped.

        Returns:
            torch.Tensor: Randomly flipped volume.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[0] = coord[2]
            coord_new[2] = coord[0]
            return vol.flip(3), coord_new  # Flip along width
        return vol, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlipCoord3D(object):
    """Vertically flip the given 3D volume randomly with a given probability.

    Args:
        p (float): probability of the volume being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vol, coord):
        """
        Args:
            vol (torch.Tensor): Volume to be flipped.

        Returns:
            torch.Tensor: Randomly flipped volume.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[1] = coord[3]
            coord_new[3] = coord[1]
            return vol.flip(2), coord_new  # Flip along height
        return vol, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomDepthFlipCoord3D(object):
    """Depth-wise flip the given 3D volume randomly with a given probability.

    Args:
        p (float): probability of the volume being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vol, coord):
        """
        Args:
            vol (torch.Tensor): Volume to be flipped.

        Returns:
            torch.Tensor: Randomly flipped volume.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[4] = coord[5]
            coord_new[5] = coord[4]
            return vol.flip(1), coord_new  # Flip along depth
        return vol, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCropCoord3D(object):
    """Crop the given 3D volume to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge (depth, height, width)
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(vol, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            vol (torch.Tensor): Volume to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (d, h, w, depth, height, width) to be passed to ``crop`` for a random
                sized crop.
        """
        depth, height, width = _get_volume_size(vol)
        area = depth * height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round((target_area * aspect_ratio) ** (1 / 3)))
            h = int(round((target_area / aspect_ratio) ** (1 / 3)))
            d = int(round((target_area / aspect_ratio) ** (1 / 3)))

            if 0 < w <= width and 0 < h <= height and 0 < d <= depth:
                d0 = random.randint(0, depth - d)
                h0 = random.randint(0, height - h)
                w0 = random.randint(0, width - w)
                return d0, h0, w0, d, h, w, depth, height, width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
            d = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
            d = int(round(h * max(ratio)))
        else:  # whole volume
            w = width
            h = height
            d = depth
        d0 = (depth - d) // 2
        h0 = (height - h) // 2
        w0 = (width - w) // 2
        return d0, h0, w0, d, h, w, depth, height, width

    def __call__(self, vol):
        """
        Args:
            vol (torch.Tensor): Volume to be cropped and resized.

        Returns:
            torch.Tensor: Randomly cropped and resized volume.
        """
        d0, h0, w0, d, h, w, depth, height, width = self.get_params(vol, self.scale, self.ratio)
        coord = torch.Tensor([
            float(w0) / (width - 1), float(h0) / (height - 1), float(d0) / (depth - 1),
            float(w0 + w - 1) / (width - 1), float(h0 + h - 1) / (height - 1), float(d0 + d - 1) / (depth - 1)
        ])
        cropped_vol = vol[:, d0:d0+d, h0:h0+h, w0:w0+w]
        return F.interpolate(cropped_vol.unsqueeze(0), size=self.size, mode='trilinear', align_corners=False).squeeze(0), coord

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        return format_string
