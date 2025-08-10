"""Module containing the dataset related functionality."""

from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from organdetr.data.transforms import get_transforms
from organdetr.data.transforms import get_weak_transforms
from organdetr.data.transforms import get_pixpro_transform

#data_base_dir = "/mnt/data/organdetr_prep/dataset/"  #"datasets/"

class OrgandetrDataset(Dataset):
    """Dataset class of the organdetr project."""
    def __init__(self, config, split, root=None, replay_samples=None, pixpro=False):
        assert split in ['train', 'val', 'test', 'labeled_train', 'unlabeled_train']
        self._config = config
        self._pixpro = pixpro

        # Read from env
        if root == None:
            data_dir = Path(os.getenv("ORGANDETR_DATA")).resolve()
            self._path_to_split = data_dir / self._config['dataset'] / split
        # Read from config file
        else:
            data_dir = Path(root).resolve()
            self._path_to_split = data_dir /  split

        self._split = split
        self._data = []
        if replay_samples == None:
            self._replay = False
            self._data = [data_path.name for data_path in self._path_to_split.iterdir()]
        else:
            self._replay = True
            source_data_dir = Path(config['source_dataset_dir']).resolve()
            self._path_to_source_split = source_data_dir / config['source_split']

            self._data = [sample.parts[-1] for sample in replay_samples]
            '''
            count = 0
            for data_path in self._path_to_split.iterdir():
                # target
                self._data.append(data_path.name)
                # source
                self._data.append(replay_samples[count].parts[-1])
                count += 1
                # repeat source
                if count == len(replay_samples):
                    count = 0
            '''

        print(f"=============spilt={split}, len(self._data)={len(self._data)}")
        if replay_samples is not None:
            print(f"=============replay, source spilt={config['source_split']}, len(replay_samples)={len(replay_samples)}")

        self._augmentation = get_transforms(split, config)
        # key label is for temporary, label is unavailabel for unlabeled data.
        self._weak_image_augmentation = get_weak_transforms(split, config, keys=['image', 'label'])
        self._weak_image_label_augmentation = get_weak_transforms(split, config, keys=['image', 'label'])

        # if split == 'unlabeled_train':
        if self._pixpro: 
            self._pixpro_weak_augmentation = get_pixpro_transform(config, False)
            self._pixpro_strong_augmentation = get_pixpro_transform(config, True)
    
    # reset dynamically during training
    def reset_transforms(self, config):
        self._augmentation = get_transforms(self._split, config)
        # key label is for temporary, label is unavailabel for unlabeled data.
        self._weak_image_augmentation = get_weak_transforms(self._split, config, keys=['image', 'label'])
        self._weak_image_label_augmentation = get_weak_transforms(self._split, config, keys=['image', 'label'])

        # if split == 'unlabeled_train':
        if self._pixpro: 
            self._pixpro_weak_augmentation = get_pixpro_transform(config, False)
            self._pixpro_strong_augmentation = get_pixpro_transform(config, True)


    def __len__(self):
        return len(self._data)

    def get_pixpro_train_item(self, idx):
        if self._config['overfit']:
            idx = 0

        case = self._data[idx]
        path_to_case = self._path_to_split / case
        # data_path, label_path = sorted(list(path_to_case.iterdir()), key=lambda x: len(str(x)))
        # data_path = list(path_to_case.iterdir())[0]
        data_path = path_to_case / 'data.npy'
        # Load npy files
        data = np.load(data_path)
        data_dict = {
            'image': data,
            'coords': None
        }

        # Apply data augmentation
        self._pixpro_strong_augmentation.set_random_state(torch.initial_seed() + idx)
        self._pixpro_weak_augmentation.set_random_state(torch.initial_seed() + idx)

        data_q_transformed = self._pixpro_strong_augmentation(data_dict)
        data_q = data_q_transformed['image']
        crop_coords_q = data_q_transformed['coords']

        if 'weak_strong_pixpro' in self._config['augmentation'] and self._config['augmentation']['weak_strong_pixpro']:
            print(f"=======================weak")
            data_k_transformed = self._pixpro_weak_augmentation(data_dict)
        else:
            data_k_transformed = self._pixpro_strong_augmentation(data_dict)

        data_k= data_k_transformed['image']
        crop_coords_k = data_k_transformed['coords']
        return data_q, data_k, crop_coords_q, crop_coords_k, path_to_case 

    def get_unlabeled_train_item(self, idx):
        if self._config['overfit']:
            idx = 0

        case = self._data[idx]
        path_to_case = self._path_to_split / case
        data_path = path_to_case / 'data.npy' 
        # We might also have label for unlabeled data
        # These labels are only used to check the quality of pseudo labels, not for training
        label_path = path_to_case / 'label.npy' 
        if os.path.isfile(label_path):
            label = np.load(label_path)
            num_classes = self._config['backbone']['num_organs']
            label[(label > num_classes)] = 0

            '''
            # Load npy files
            data, label = np.load(data_path), np.load(label_path)

            # Extract the valid label numbers and sort them in ascending order
            valid_labels = sorted(int(k) for k in self._config['labels'].keys())  # Ensure keys are integers

            # Create a mapping from original labels to a new monotonically increasing range (1 to num_classes)
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_labels, start=1)}

            # Apply the mapping to the label array
            label = np.vectorize(lambda x: label_mapping.get(x, 0))(label)

            print("Updated labels:", np.unique(label))  # Check unique labels after remapping
            '''
        else:
            label = None

        # Load npy files
        data= np.load(data_path)

        if self._config['augmentation']['use_augmentation']:
            if label is None:
                data_dict = {
                    'image': data,
                }
                self._weak_image_augmentation.set_random_state(torch.initial_seed() + idx)
                self._augmentation.set_random_state(torch.initial_seed() + idx)

                data_transformed, data_weak_transformed = self._augmentation(data_dict), self._weak_image_augmentation(data_dict)
                data, weak_data = data_transformed['image'], data_weak_transformed['image'] 
                label, weak_label = None, None
            else:
                data_dict = {
                    'image': data,
                    'label': label,
                }

                self._weak_image_label_augmentation.set_random_state(torch.initial_seed() + idx)
                self._augmentation.set_random_state(torch.initial_seed() + idx)
                data_transformed, data_weak_transformed = self._augmentation(data_dict), self._weak_image_label_augmentation(data_dict)
                data, label = data_transformed['image'], data_transformed['label']
                weak_data, weak_label = data_weak_transformed['image'], data_weak_transformed['label']

            # Apply data augmentation
        else:
            data, weak_data = torch.tensor(data), torch.tensor(data)
            if label is None:
                label, weak_label = None
            else:
                label, weak_label = torch.tensor(label), torch.tensor(label)

        return data, weak_data, label, weak_label, path_to_case


    def __getitem__(self, idx):
        if self._config['overfit']:
            idx = 0

        # if self._split == 'unlabeled_train':
        if self._pixpro:
            return self.get_pixpro_train_item(idx)
        elif self._split == 'unlabeled_train':
            return self.get_unlabeled_train_item(idx)
            
        case = self._data[idx]
        if self._replay:
            '''
            if idx % 2 == 0:
                path_to_case = self._path_to_split / case
            else:
                path_to_case = self._path_to_source_split / case
            '''
            path_to_case = self._path_to_source_split / case
        else:
            path_to_case = self._path_to_split / case


        data_path, label_path = sorted(list(path_to_case.iterdir()), key=lambda x: len(str(x)))

        # Load npy files
        data, label = np.load(data_path), np.load(label_path)
        num_classes = self._config['backbone']['num_organs']
        label[(label > num_classes)] = 0

        '''
        # Load npy files
        data, label = np.load(data_path), np.load(label_path)

        # Extract and sort valid label numbers (fixed across all files)
        valid_labels = sorted(int(k) for k in self._config['labels'].keys())  # Ensure keys are integers

        # Create a fixed mapping (always maps 1 → 1, 2 → 2, ..., 16 → 8)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_labels, start=1)}

        # Apply the mapping to the label array (ignoring missing labels)
        label = np.vectorize(lambda x: label_mapping.get(x, 0))(label)
        '''

        # print("Updated labels:", np.unique(label))  # Check unique labels after remapping

        if self._config['augmentation']['use_augmentation']:
            data_dict = {
                'image': data,
                'label': label
            }

            #print(f"==============before transform, label.shape={label.shape}")

            # Apply data augmentation
            self._augmentation.set_random_state(torch.initial_seed() + idx)
            self._weak_image_label_augmentation.set_random_state(torch.initial_seed() + idx)
            data_transformed, data_weak_transformed  = self._augmentation(data_dict), self._weak_image_label_augmentation(data_dict)
            data, label = data_transformed['image'], data_transformed['label']
            weak_data, weak_label = data_weak_transformed['image'], data_weak_transformed['label']
            #print(f"==============after transform, label.shape={label.shape}")
        else:
            data, label = torch.tensor(data), torch.tensor(label)
            weak_data, weak_label = torch.tensor(data), torch.tensor(label)

        if self._split == 'test':
            # no aug, weak_data == data
            return data, data, label, label, path_to_case # path is used for visualization of predictions on source data
        else:
            return data, weak_data, label, weak_label, path_to_case
