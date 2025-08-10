"""Module containing dataloader related functionality."""

import torch
from torch.utils.data import DataLoader

from organdetr.data.dataset import OrgandetrDataset
from organdetr.utils.bboxes import segmentation2bbox

def get_loader(config, split, batch_size=None, root=None, replay_samples=None, pixpro=False):
    if not batch_size:
        batch_size = config['batch_size']

    # Init collator
    collator = OrgandetrCollator(config, split, pixpro)
    # Don't shuffle replay samples
    shuffle = False if split in ['test', 'val'] else config['shuffle']

    dataset = OrgandetrDataset(config, split, root, replay_samples, pixpro)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=config['num_workers'], collate_fn=collator
    )

    return dataloader


# def init_fn(worker_id):
#     """
#     https://github.com/pytorch/pytorch/issues/7068
#     https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
#     """
#     torch_seed = torch.initial_seed()
#     if torch_seed >= 2**30:
#         torch_seed = torch_seed % 2**30
#     seed = torch_seed + worker_id

#     random.seed(seed)   
#     np.random.seed(seed)
#     monai.utils.set_determinism(seed=seed)
#     torch.manual_seed(seed)


class OrgandetrCollator:
    def __init__(self, config, split, pixpro=False):
        self._bbox_padding = config['bbox_padding']
        self._split = split
        self._config = config
        self._pixpro = pixpro

    def __call__(self, batch):
        # if self._split == 'unlabeled_train':
        if self._pixpro:
            batch_images_q = []
            batch_images_k = []
            batch_crop_coords_q = []
            batch_crop_coords_k = []
            batch_paths = []
            for image_q, image_k, crop_coords_q, crop_coords_k, path in batch:
                batch_images_q.append(image_q)
                batch_images_k.append(image_k)
                batch_crop_coords_q.append(crop_coords_q)
                batch_crop_coords_k.append(crop_coords_k)
                batch_paths.append(path)

            return torch.stack(batch_images_q), torch.stack(batch_images_k), torch.stack(batch_crop_coords_q), torch.stack(batch_crop_coords_k), batch_paths

        elif self._split == 'unlabeled_train':
            batch_images = []
            batch_weak_images = []
            batch_labels = []
            batch_weak_labels = []
            batch_masks = []
            batch_paths = []
            for image, weak_image, label, weak_label, path in batch:
                batch_images.append(image)
                batch_weak_images.append(image)
                batch_labels.append(label)
                batch_weak_labels.append(weak_label)
                batch_masks.append(torch.zeros_like(image))
                batch_paths.append(path)

            if batch_labels[0] is None:
                return torch.stack(batch_images), torch.stack(batch_weak_images), torch.stack(batch_masks), None, None, None, None, batch_paths
            else:
                batch_bboxes, batch_classes = segmentation2bbox(torch.stack(batch_labels), self._bbox_padding)
                batch_weak_bboxes, batch_weak_classes = segmentation2bbox(torch.stack(batch_weak_labels), self._bbox_padding)
                return torch.stack(batch_images), torch.stack(batch_weak_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), list(zip(batch_weak_bboxes, batch_weak_classes)), torch.stack(batch_labels), torch.stack(batch_weak_labels), batch_paths
        else:
            batch_images = []
            batch_weak_images = []
            batch_labels = []
            batch_weak_labels = []
            batch_masks = []
            batch_paths = []
            for image, weak_image, label, weak_label, path in batch:
                batch_images.append(image)
                batch_weak_images.append(weak_image)
                batch_labels.append(label)
                batch_weak_labels.append(weak_label)
                batch_masks.append(torch.zeros_like(image))
                batch_paths.append(path)

            #print(f"================dataloader, batch_labels length={len(batch_labels)}")
            # Generate bboxes and corresponding class labels
            batch_bboxes, batch_classes = segmentation2bbox(torch.stack(batch_labels), self._bbox_padding)
            batch_weak_bboxes, batch_weak_classes = segmentation2bbox(torch.stack(batch_weak_labels), self._bbox_padding)

            return torch.stack(batch_images), torch.stack(batch_weak_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), list(zip(batch_weak_bboxes, batch_weak_classes)), torch.stack(batch_labels), torch.stack(batch_weak_labels), batch_paths
