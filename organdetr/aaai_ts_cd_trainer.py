"""Module containing the trainer of the organdetr project."""

import copy
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import PIL.Image
from tqdm import tqdm
import numpy as np
from organdetr.evaluator import DetectionEvaluator
from organdetr.inference import inference
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import io
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.ops as ops
from scipy.optimize import linear_sum_assignment
from organdetr.models.discriminator import FCDiscriminator_img_3D
from organdetr.models.discriminator import grad_reverse

# helper function: generate box_plot of grads in tensorboard
def gen_box_plot(grads_list, num_epoch_list, name=None):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.boxplot(grads_list)
    plt.title(f'{name} of from epoch {num_epoch_list[0]} to epoch {num_epoch_list[-1]}')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    plt.xticks(np.arange(len(grads_list))+1, num_epoch_list)
    buf.seek(0)
    return buf

class TSTrainer:

    def __init__(
        self, source_train_loader, labeled_train_loader, pixpro_train_loader, pseudo_train_loader, val_loader, model, teacher_model, ensemble_model, pixpro_model, discriminator, criterion, optimizer, scheduler,
        device, config, path_to_run, epoch, metric_start_val, dense_hybrid_criterion, pseudo_cls_coef=None
    ):
        self._source_train_loader = source_train_loader
        self._labeled_train_loader = labeled_train_loader
        self._pixpro_train_loader = pixpro_train_loader
        self._pseudo_train_loader = pseudo_train_loader
        self._val_loader = val_loader
        self._model = model
        self._teacher_model = teacher_model
        self._ensemble_model = ensemble_model
        self._pixpro_model = pixpro_model
        self._discriminator = discriminator
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device
        self._path_to_run = path_to_run
        self._epoch_to_start = epoch
        self._config = config
        self.log_grad = config.get('log_grad', False)
        self.log_grad_every_epoch = config.get('log_grad_every_epoch', 10)
        self._hybrid = config.get('hybrid_matching', False)
        self._hybrid_K = config.get('hybrid_K', 0)
        self._dense_hybrid_criterion = dense_hybrid_criterion

        #self.dtype = torch.bfloat16

        if pseudo_cls_coef:
            self._pseudo_cls_coef = pseudo_cls_coef 
        else:
            self._pseudo_cls_coef = config['loss_coefs']['pseudo']

        self._avg_cls = 0

        # domain loss
        if 'srcdom' in config['loss_coefs'] and 'tgtdom' in config['loss_coefs']:
            self._srcdom_loss_coef = config['loss_coefs']['srcdom']
            self._tgtdom_loss_coef = config['loss_coefs']['tgtdom']
        else:
            self._srcdom_loss_coef = None
            self._tgtdom_loss_coef = None
        
        if self.log_grad:
            self.log_grads_list_pos = []
            self.log_grads_list_neg = []
            self.log_epoch_list = []

        self._writer = SummaryWriter(log_dir=path_to_run)
        self._scaler = GradScaler()

        '''
        # Ensure labels are sorted based on their numerical order
        sorted_labels = sorted((int(k), v) for k, v in config['labels'].items())  # List of (key, value) tuples

        # Create a fixed mapping: {original_label → new_index}
        label_mapping = {old_label: new_label for new_label, (old_label, _) in enumerate(sorted_labels, start=1)}

        # Extract sorted class names in fixed order
        sorted_class_names = [v for _, v in sorted_labels]  # Only take the names, preserving order

        # Function to remap labels in classes_small, classes_mid, classes_large
        def remap_classes(original_dict):
            return {str(label_mapping[int(k)]): v for k, v in original_dict.items() if int(k) in label_mapping}

        # Apply remapping to small, mid, and large label dictionaries
        classes_small = remap_classes(config['labels_small'])
        classes_mid = remap_classes(config['labels_mid'])
        classes_large = remap_classes(config['labels_large'])

        # Pass to the evaluator
        self._evaluator = DetectionEvaluator(
            classes=sorted_class_names,
            classes_small=classes_small,
            classes_mid=classes_mid,
            classes_large=classes_large,
            iou_range_nndet=(0.1, 0.5, 0.05),
            iou_range_coco=(0.5, 0.95, 0.05),
            sparse_results=True
        )

        # Debugging: Print the remapped dictionaries to verify
        print("Remapped classes_small:", classes_small)
        print("Remapped classes_mid:", classes_mid)
        print("Remapped classes_large:", classes_large)
        '''


        self._evaluator = DetectionEvaluator(
            classes=list(config['labels'].values()),
            classes_small=config['labels_small'],
            classes_mid=config['labels_mid'],
            classes_large=config['labels_large'],
            iou_range_nndet=(0.1, 0.5, 0.05),
            iou_range_coco=(0.5, 0.95, 0.05),
            sparse_results=True
        )

        # Init main metric for checkpoint
        self._main_metric_key = 'mAP_coco'
        self._main_metric_max_val = metric_start_val

        self._main_metric_key_student = 'mAP_coco'
        self._main_metric_max_val_student = metric_start_val

    def binary_cross_entropy_with_logits_eps(self, logits, targets, epsilon=1e-3):
        # print(f"============logits.max={torch.max(logits).item()}, logits.min={torch.min(logits).item()}")
        logits = torch.clamp(logits, min=-10, max=10)
        # print(f"============clamped logits.max={torch.max(logits).item()}, logits.min={torch.min(logits).item()}")
        # Step 1: Convert logits to probabilities
        probabilities = torch.sigmoid(logits)
        # print(f"============prob.max={torch.max(probabilities).item()}, prob.min={torch.min(probabilities).item()}")
        
        # Step 2: Clamp probabilities to avoid log(0) or log(1) issues
        probabilities = torch.clamp(probabilities, min=epsilon, max=1-epsilon)
        # print(f"============clamped prob.max={torch.max(probabilities).item()}, clamped prob.min={torch.min(probabilities).item()}")
        
        # Step 3: Compute the binary cross-entropy manually
        bce_loss = - (targets * torch.log(probabilities) + (1 - targets) * torch.log(1 - probabilities))
        bce_loss = bce_loss.mean()
        # print(f"============bec_loss={(bce_loss).item()}") 
        #assert not torch.isnan(bce_loss)
        
        # Return the mean loss
        return bce_loss 


    def get_domain_loss(self, feature, domain='source'):
        assert domain == 'source' or domain == 'target'
        if domain == 'source':
            label = 0
        else:
            label = 1
        
        '''
            # Hook to capture gradient before grad_reverse
        def print_grad_before(grad):
            print(f"************Gradient before grad_reverse: {torch.norm(grad)}")

        # Register the hook before grad_reverse
        feature.register_hook(print_grad_before)
        '''

        feature = grad_reverse(feature)

        '''
            # Hook to capture gradient after grad_reverse
        def print_grad_after(grad):
            print(f"************Gradient after grad_reverse: {torch.norm(grad)}")

        # Register the hook after grad_reverse
        feature.register_hook(print_grad_after)
        '''

        logits = self._discriminator(feature)
        target = torch.full_like(logits, fill_value=label, device=self._device)

        loss = self.binary_cross_entropy_with_logits_eps(logits, target)
        return loss

    def calculate_gradient_norm(self):
        total_norm = 0.0
        for p in self._model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def xyzwhd_to_minmax(self, boxes):
        """
        Convert boxes from [x, y, z, w, h, d] (center format) to [x_min, y_min, z_min, x_max, y_max, z_max] format.
        
        Args:
        - boxes (Tensor): Bounding boxes in the format [x, y, z, w, h, d], shape [N, 6].
        
        Returns:
        - Tensor: Bounding boxes in the format [x_min, y_min, z_min, x_max, y_max, z_max], shape [N, 6].
        """
        # Extract center coordinates and dimensions
        x, y, z, w, h, d = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5]

        # Calculate min/max coordinates
        x_min = x - w / 2
        y_min = y - h / 2
        z_min = z - d / 2
        x_max = x + w / 2
        y_max = y + h / 2
        z_max = z + d / 2

        # Return boxes in min-max format
        return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=1)

    def box3d_iou(self, box1, box2):
        """
        Compute the Intersection-over-Union (IoU) between two 3D bounding boxes.
        Each box is represented as [x_min, y_min, z_min, x_max, y_max, z_max].
        """
        # Compute intersection
        inter_min = torch.max(box1[:, :3], box2[:, :3])  # Take max of min coords
        inter_max = torch.min(box1[:, 3:], box2[:, 3:])  # Take min of max coords
        inter_dims = (inter_max - inter_min).clamp(min=0)  # Get intersection dimensions
        inter_volume = inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2]  # Volume of intersection

        # Compute volumes of both boxes
        box1_volume = (box1[:, 3] - box1[:, 0]) * (box1[:, 4] - box1[:, 1]) * (box1[:, 5] - box1[:, 2])
        box2_volume = (box2[:, 3] - box2[:, 0]) * (box2[:, 4] - box2[:, 1]) * (box2[:, 5] - box2[:, 2])

        # Compute IoU
        union_volume = box1_volume + box2_volume - inter_volume
        iou = inter_volume / union_volume
        return iou

    def nms_3d(self, boxes, scores, iou_threshold=0.5):
        """
        Perform Non-Maximum Suppression (NMS) on 3D bounding boxes.
        
        Args:
        - boxes (Tensor): 3D bounding boxes [N, 6] in the format [x, y, z, w, h, d].
        - scores (Tensor): Confidence scores for each box [N].
        - iou_threshold (float): IoU threshold for suppressing overlapping boxes.
        
        Returns:
        - keep (Tensor): Indices of the boxes to keep.
        """
        keep = []

        # Convert boxes from xyzwhd format to x_min, y_min, z_min, x_max, y_max, z_max
        boxes_minmax = self.xyzwhd_to_minmax(boxes)

        # Sort boxes by scores in descending order
        _, idxs = scores.sort(descending=True)

        while idxs.numel() > 0:
            # Select the box with the highest score
            current_idx = idxs[0]
            keep.append(current_idx)

            if idxs.numel() == 1:
                break

            # Compute IoU between the current box and the rest
            current_box = boxes_minmax[current_idx].unsqueeze(0)
            other_boxes = boxes_minmax[idxs[1:]]
            iou = self.box3d_iou(current_box, other_boxes)

            # Keep boxes with IoU less than the threshold
            idxs = idxs[1:][iou < iou_threshold]

        return torch.tensor(keep)

    def get_pseudo_labels_with_nms_3d(self, outputs):
        prob = F.softmax(outputs['pred_logits'], dim=-1)  # Shape: [batch_size, num_queries, num_classes]
        pred_classes = prob.argmax(dim=-1)  # Shape: [batch_size, num_queries]
        print(f"==================================pred_classes={pred_classes[0, :30]}")

        # Get the confidence values (max probabilities) for each prediction
        confidence_values, _ = prob.max(dim=-1)  # Shape: [batch_size, num_queries]
        print(f"==================================confidence_values={confidence_values[0, :30]}")

        # Apply a confidence threshold
        confidence_threshold = self._config['pseudo_label_threshold']
        high_confidence_mask = confidence_values > confidence_threshold  # Shape: [batch_size, num_queries]

        # Initialize lists to store the filtered predictions
        high_confidence_pred_classes = []
        high_confidence_pred_boxes = []
        high_confidence_pred_scores = []

        for i in range(prob.size(0)):  # Iterate over batch
            # Create a combined mask to remove low-confidence predictions and class 0
            combined_mask = (high_confidence_mask[i]) & (pred_classes[i] != 0)

            # Filter out predictions based on the mask
            filtered_pred_classes = pred_classes[i][combined_mask]  # Filtered classes
            filtered_pred_boxes = outputs['pred_boxes'][i][combined_mask]  # Corresponding boxes
            filtered_pred_scores = confidence_values[i][combined_mask]  # Corresponding scores

            # Append to the lists
            high_confidence_pred_classes.append(filtered_pred_classes)
            high_confidence_pred_boxes.append(filtered_pred_boxes)
            high_confidence_pred_scores.append(filtered_pred_scores)

        # Apply 3D Non-Maximum Suppression (NMS) for each batch element
        iou_threshold = 0.5 
        final_pred_classes = []
        final_pred_boxes = []
        
        for i in range(len(high_confidence_pred_classes)):
            if len(high_confidence_pred_boxes[i]) == 0:  # Skip if no boxes
                final_pred_classes.append(torch.tensor([]).to(outputs['pred_logits'].device))
                final_pred_boxes.append(torch.tensor([]).to(outputs['pred_logits'].device))
                continue

            # Use the NMS to filter out redundant boxes
            keep_indices = self.nms_3d(high_confidence_pred_boxes[i], high_confidence_pred_scores[i], iou_threshold)
            
            # Store the final classes and boxes after applying NMS
            final_pred_classes.append(high_confidence_pred_classes[i][keep_indices])
            final_pred_boxes.append(high_confidence_pred_boxes[i][keep_indices])

        # Return the pseudo labels (filtered classes and boxes) after NMS
        targets = [{"labels": final_pred_classes[i], "boxes": final_pred_boxes[i]} for i in range(prob.size(0))]
        return targets

    # Define IoU function for 3D bounding boxes
    def compute_iou_3d(self, box1, box2):
        # Calculate the intersection coordinates
        min_coords = torch.max(box1[:, :3], box2[:, :3])  # Max of the mins (top-left)
        max_coords = torch.min(box1[:, 3:], box2[:, 3:])  # Min of the maxes (bottom-right)
        
        # Calculate intersection volume
        intersection_dims = torch.clamp(max_coords - min_coords, min=0)
        intersection_volume = intersection_dims[:, 0] * intersection_dims[:, 1] * intersection_dims[:, 2]
        
        # Calculate volumes of both boxes
        box1_volume = torch.prod(box1[:, 3:] - box1[:, :3], dim=1)
        box2_volume = torch.prod(box2[:, 3:] - box2[:, :3], dim=1)
        
        # Calculate union volume
        union_volume = box1_volume + box2_volume - intersection_volume
        
        # Calculate IoU
        iou = intersection_volume / union_volume
        return iou

    def pseudo_accuracy(self, pseudo_boxes, gt_boxes, pseudo_labels, gt_labels):
        # Compute IoU matrix
        num_pseudo = len(pseudo_boxes)
        num_gt = len(gt_boxes)
        iou_matrix = torch.zeros((num_pseudo, num_gt))

        pseudo_boxes_minmax = self.xyzwhd_to_minmax(pseudo_boxes)
        gt_boxes_minmax = self.xyzwhd_to_minmax(gt_boxes)
        
        for i in range(num_pseudo):
            for j in range(num_gt):
                iou_matrix[i, j] = self.compute_iou_3d(pseudo_boxes_minmax[i:i+1], gt_boxes_minmax[j:j+1])
        
        #print(f"IOU Matrix:\n{iou_matrix}")

        # Perform Hungarian matching
        cost_matrix = -iou_matrix.numpy()
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        #print(f"Cost Matrix:\n{cost_matrix}")
        #print(f"Row indices: {row_indices}")
        #print(f"Column indices: {col_indices}")

        # Evaluate matches
        iou_threshold = 0.5  # You might want to experiment with different thresholds
        correct_classifications = 0
        matched_gt_indices = set()

        for i, j in zip(row_indices, col_indices):
            if i < num_pseudo and j < num_gt:
                #print(f"Matching pseudo box {i} with ground truth box {j}")
                #print(f"Pseudo Box {i} Label: {pseudo_labels[i]}, GT Box {j} Label: {gt_labels[j]}")
                #print(f"IoU: {iou_matrix[i, j]}")
                if iou_matrix[i, j] >= iou_threshold:
                    if pseudo_labels[i] == gt_labels[j]:
                        correct_classifications += 1
                    matched_gt_indices.add(j)

        classification_accuracy = correct_classifications / len(gt_labels) if len(gt_labels) > 0 else 0

        matched_ious = iou_matrix[row_indices, col_indices]
        avg_iou = matched_ious.mean().item() if matched_ious.numel() > 0 else 0

        print(f"Average IoU: {avg_iou}")
        #print(f"Correct Classifications: {correct_classifications}")
        #print(f"Total Ground Truth Labels: {len(gt_labels)}")
        print(f"Classification Accuracy: {classification_accuracy}")


    def calculate_iou_3d(self, box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two 3D crops with coordinates (x1, y1, z1, x2, y2, z2).

        :param box1: List or tuple of coordinates (x1, y1, z1, x2, y2, z2) for the first crop.
        :param box2: List or tuple of coordinates (x1, y1, z1, x2, y2, z2) for the second crop.
        :return: IoU value between 0 and 1.
        """
        # Extract the coordinates for the two boxes
        x1_1, y1_1, z1_1, x2_1, y2_1, z2_1 = box1
        x1_2, y1_2, z1_2, x2_2, y2_2, z2_2 = box2

        # Calculate the coordinates of the intersection box
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        z1_int = max(z1_1, z1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        z2_int = min(z2_1, z2_2)

        # Calculate the volume of the intersection box
        int_width = max(0, x2_int - x1_int)
        int_height = max(0, y2_int - y1_int)
        int_depth = max(0, z2_int - z1_int)
        intersection_volume = int_width * int_height * int_depth

        # Calculate the volume of both boxes
        volume_box1 = (x2_1 - x1_1) * (y2_1 - y1_1) * (z2_1 - z1_1)
        volume_box2 = (x2_2 - x1_2) * (y2_2 - y1_2) * (z2_2 - z1_2)

        # Calculate the union volume
        union_volume = volume_box1 + volume_box2 - intersection_volume

        # Calculate the IoU
        if union_volume == 0:
            return 0.0  # Avoid division by zero
        iou = intersection_volume / union_volume
        return iou

    '''
    def get_ins_feature(self, feature, targets, large_organ_labels, device):
        feat_w, feat_h, feat_d = feature.shape[2], feature.shape[3], feature.shape[4]
        ins_set = torch.tensor([]).to(device)
        # for ins in targets_ins:
        for i, img in enumerate(targets):
            img_idx = i
            for j in range(img['boxes'].shape[0]):
                if img['labels'][j] in large_organ_labels:
                    x,y,z,w,h,d = img['boxes'][j]
                    x1, x2 = max(int(feat_w*(x-0.5*w)), 0), max(int(feat_w*(x+0.5*w)/2), int(feat_w*(x-0.5*w))+1)
                    y1, y2 = max(int(feat_h*(y-0.5*h)), 0), max(int(feat_h*(y+0.5*h)/2), int(feat_h*(y-0.5*h))+1)
                    z1, z2 = max(int(feat_d*(z-0.5*d)), 0), max(int(feat_d*(z+0.5*d)/2), int(feat_d*(z-0.5*d))+1)
                    # o_ins = feature[img_idx, :, x1:x2, y1:y2, z1:z2].mean(3).mean(2).mean(1).unsqueeze(0)
                    o_ins = feature[img_idx, :, x1:x2, y1:y2, z1:z2]
                    ins_set=torch.cat((ins_set, o_ins))

        return ins_set
    '''

    '''
    def get_ins_feature(self, feature, targets, large_organ_labels, device):
        feat_w, feat_h, feat_d = feature.shape[2], feature.shape[3], feature.shape[4]
        ins_set = torch.tensor([]).to(device)

        for i, img in enumerate(targets):
            img_idx = i
            for j in range(img['boxes'].shape[0]):
                if img['labels'][j] in large_organ_labels:
                    # Scale and ensure integer coordinates match the feature map
                    x, y, z, w, h, d = img['boxes'][j]
                    x1 = max(int(feat_w * (x - 0.5 * w)), 0)
                    x2 = min(int(feat_w * (x + 0.5 * w)), feat_w)
                    y1 = max(int(feat_h * (y - 0.5 * h)), 0)
                    y2 = min(int(feat_h * (y + 0.5 * h)), feat_h)
                    z1 = max(int(feat_d * (z - 0.5 * d)), 0)
                    z2 = min(int(feat_d * (z + 0.5 * d)), feat_d)

                    # Extract the feature patch for the organ
                    o_ins = feature[img_idx, :, x1:x2, y1:y2, z1:z2]
                    ins_set = torch.cat((ins_set, o_ins), dim=0)

        return ins_set
    '''

    '''
    def get_ins_feature(self, feature, targets, device, number=100):
        """
        Extract instance-level features for MMD alignment.

        Args:
            feature: Tensor [B, C, W, H, D] - backbone feature map.
            targets: list of dicts, each with:
                    targets[i]['boxes']: Tensor [N, 6] in (x, y, z, w, h, d) normalized [0,1]
            device: torch device
            number: max number of instances to sample.

        Returns:
            ins_set: Tensor [N, C] - averaged instance features.
        """
        feat_w, feat_h, feat_d = feature.shape[2], feature.shape[3], feature.shape[4]
        ins_set = []

        # Randomly sample (optional) or just use all
        for img_idx, img in enumerate(targets[:number]):
            for j in range(img['boxes'].shape[0]):
                x, y, z, w, h, d = img['boxes'][j]

                # Convert to feature map indices (ensure within bounds)
                x1 = max(int((x - 0.5 * w) * feat_w), 0)
                x2 = min(int((x + 0.5 * w) * feat_w), feat_w)
                y1 = max(int((y - 0.5 * h) * feat_h), 0)
                y2 = min(int((y + 0.5 * h) * feat_h), feat_h)
                z1 = max(int((z - 0.5 * d) * feat_d), 0)
                z2 = min(int((z + 0.5 * d) * feat_d), feat_d)

                if x2 <= x1 or y2 <= y1 or z2 <= z1:
                    continue  # skip invalid boxes

                # Average pool instance feature: shape [C]
                o_ins = feature[img_idx, :, x1:x2, y1:y2, z1:z2].mean(dim=(1, 2, 3))
                ins_set.append(o_ins)

        if len(ins_set) == 0:
            return torch.zeros((0, feature.shape[1])).to(device)

        return torch.stack(ins_set).to(device)  # [N, C]

    '''
    def gaussian_kernel(self, x, y, sigma=1.0):
        x_size, y_size, dim = x.size(0), y.size(0), x.size(1)
        xx = x.unsqueeze(1).expand(x_size, y_size, dim)
        yy = y.unsqueeze(0).expand(x_size, y_size, dim)
        return torch.exp(-((xx - yy) ** 2).mean(2) / (2 * sigma ** 2))

    def mmd_loss(self, source_features, target_features, sigma=1.0):
        if source_features.size(0) == 0 or target_features.size(0) == 0:
            return torch.tensor(0.0, device=source_features.device)
        K_ss = self.gaussian_kernel(source_features, source_features, sigma)
        K_tt = self.gaussian_kernel(target_features, target_features, sigma)
        K_st = self.gaussian_kernel(source_features, target_features, sigma)
        return K_ss.mean() + K_tt.mean() - 2 * K_st.mean()

    def get_ins_feature(self, feature, targets, device, number=100):
        """
        Extract instance-level features for MMD alignment (class-wise).

        Args:
            feature: Tensor [B, C, W, H, D] - backbone feature map.
            targets: list of dicts, each with:
                    targets[i]['boxes']: Tensor [N, 6] (x, y, z, w, h, d) normalized [0,1]
                    targets[i]['labels']: Tensor [N] (class labels)
            device: torch device
            number: max number of images to process (not instances).

        Returns:
            ins_features: Tensor [N, C] - averaged instance features.
            ins_labels: Tensor [N] - corresponding class labels.
        """
        feat_w, feat_h, feat_d = feature.shape[2], feature.shape[3], feature.shape[4]
        ins_features = []
        ins_labels = []

        for img_idx, img in enumerate(targets[:number]):
            boxes = img['boxes']
            labels = img['labels']
            for j in range(boxes.shape[0]):
                x, y, z, w, h, d = boxes[j]
                cls = int(labels[j].item())

                # Convert to feature map indices (ensure within bounds)
                x1 = max(int((x - 0.5 * w) * feat_w), 0)
                x2 = min(int((x + 0.5 * w) * feat_w), feat_w)
                y1 = max(int((y - 0.5 * h) * feat_h), 0)
                y2 = min(int((y + 0.5 * h) * feat_h), feat_h)
                z1 = max(int((z - 0.5 * d) * feat_d), 0)
                z2 = min(int((z + 0.5 * d) * feat_d), feat_d)

                if x2 <= x1 or y2 <= y1 or z2 <= z1:
                    continue  # skip invalid boxes

                # Average pooled instance feature: shape [C]
                o_ins = feature[img_idx, :, x1:x2, y1:y2, z1:z2].mean(dim=(1, 2, 3))
                ins_features.append(o_ins)
                ins_labels.append(cls)

        if len(ins_features) == 0:
            return (
                torch.zeros((0, feature.shape[1]), device=device),
                torch.zeros((0,), dtype=torch.long, device=device)
            )

        return (
            torch.stack(ins_features).to(device),  # [N, C]
            torch.tensor(ins_labels, dtype=torch.long).to(device)  # [N]
        )


    def class_wise_feature_loss(self, source_features, source_labels,
                            target_features, target_labels,
                            num_classes):
        """
        Direct feature distance matching for 1-instance-per-class scenario.
        """
        total_loss = torch.tensor(0.0, device=source_features.device)
        valid_classes = 0

        for cls in range(1, num_classes + 1):  # 1-based labels
            src_feat = source_features[source_labels == cls]
            tgt_feat = target_features[target_labels == cls]

            if src_feat.size(0) == 1 and tgt_feat.size(0) == 1:
                # simple L2 distance
                total_loss += torch.nn.functional.mse_loss(src_feat, tgt_feat)
                valid_classes += 1

        if valid_classes > 0:
            total_loss /= valid_classes

        return total_loss


    '''
    def class_wise_mmd_loss(self, source_features, source_labels,
                        target_features, target_labels,
                        num_classes, sigma=1.0):
        """
        Compute class-wise MMD loss.

        Args:
            source_features: [Ns, C]
            source_labels: [Ns]
            target_features: [Nt, C]
            target_labels: [Nt]
            num_classes: total number of classes
            sigma: Gaussian kernel sigma

        Returns:
            total_mmd: scalar tensor
        """
        total_mmd = torch.tensor(0.0, device=source_features.device)
        valid_classes = 0

        for cls in range(num_classes):
            src_cls_feat = source_features[source_labels == cls]
            tgt_cls_feat = target_features[target_labels == cls]

            print(f"*******************cls={cls}, src_cls_feat.size()={src_cls_feat.size()}")
            if src_cls_feat.size(0) < 2 or tgt_cls_feat.size(0) < 2:
                continue  # skip if not enough samples for stability

            total_mmd += self.mmd_loss(src_cls_feat, tgt_cls_feat, sigma)
            valid_classes += 1

        if valid_classes > 0:
            total_mmd /= valid_classes  # average over valid classes

        return total_mmd
    '''
    def tensor_equal_to_list_ignore_order(self, tensor, num_organs):
        target_list = list(range(1, num_organs + 1))
        tensor_list = tensor.cpu().view(-1).tolist()
        # Quick length check
        if len(tensor_list) != len(target_list):
            return False

        return sorted(tensor_list) == sorted(target_list)

    def _train_one_epoch(self, num_epoch):
        self._model.train()
        self._teacher_model.train()
        # self._criterion.train()

        total_loss_agg = 0

        loss_bbox_agg = 0
        loss_giou_agg = 0
        loss_cls_agg = 0
        loss_seg_ce_agg = 0
        loss_seg_dice_agg = 0

        source_loss_bbox_agg = 0
        source_loss_giou_agg = 0
        source_loss_cls_agg = 0
        source_loss_seg_ce_agg = 0
        source_loss_seg_dice_agg = 0

        loss_pseudo_cls_agg = 0
        loss_pseudo_bbox_agg = 0
        loss_pseudo_giou_agg = 0

        loss_pixpro_agg = 0

        loss_source_domain_agg = 0
        loss_target_domain_agg = 0

        loss_mmd_agg = 0

        loss_dn_agg = {}
        # Aux loss
        loss_aux_agg = {}
        loss_aux_agg_dn = {}
        # Two stage
        loss_enc_bbox_agg = 0
        loss_enc_giou_agg = 0
        loss_enc_cls_agg = 0
        # log Hausdorff
        hd95_agg = 0
        # contrastive loss
        loss_contrast_agg = 0
        # hybrid matching
        loss_bbox_one2many_agg = 0
        loss_giou_one2many_agg = 0
        loss_cls_one2many_agg = 0
        loss_seg_ce_one2many_agg = 0
        loss_seg_dice_one2many_agg = 0
        
        pos_query_grads_list = torch.Tensor([])
        neg_query_grads_list = torch.Tensor([])

        print(f"==============len(self._source_train_loader)={len(self._source_train_loader)}, len(self._labeled_train_loader)={len(self._labeled_train_loader)}")
        print(f"==============len(self._pixpro_train_loader)={len(self._pixpro_train_loader)}, len(self._pseudo_train_loader)={len(self._pseudo_train_loader)}, len(self._val_loader)={len(self._val_loader)}")

        use_pretrain_pipeline = 'use_pretrain_pipeline' in self._config and self._config['use_pretrain_pipeline']
        # update teacher and student
        if use_pretrain_pipeline:
            if num_epoch == self._config['label_pretrain_stop_epoch']:
                self._update_teacher_model(keep_rate=0.0)

            if num_epoch == self._config['pseudo_pretrain_stop_epoch']:
                self._update_student_model(keep_rate=0.0)

        '''
        progress_bar = tqdm(zip(self._source_train_loader, self._labeled_train_loader, self._pseudo_train_loader))
        for idx, ((source_data, _, _, source_bboxes, _, source_seg_targets, _, source_path), (data, _, _, bboxes, _, seg_targets, _, path),
            (pseudo_data, pseudo_weak_data, _, gt_pseudo_bboxes, gt_pseudo_weak_bboxes, _, _, pseudo_path)) in enumerate(progress_bar):
        '''
        zero_tensor = torch.tensor(0.0, device=self._device)
        progress_bar = tqdm(zip(self._source_train_loader, self._labeled_train_loader, self._pixpro_train_loader, self._pseudo_train_loader))
        for idx, ((source_data, _, _, source_bboxes, _, source_seg_targets, _, source_path), (data, _, _, bboxes, _, seg_targets, _, path), (data_q, data_k, crop_coords_q, crop_coords_k, pixpro_path),
            (pseudo_data, pseudo_weak_data, _, gt_pseudo_bboxes, gt_pseudo_weak_bboxes, _, _, pseudo_path)) in enumerate(progress_bar):
            #print(f"==========source_path={source_path}")
            #print(f"==========path={path}")
            #print(f"==========pixpro_path={pixpro_path}")
            #print(f"==========pseudo_path={pseudo_path}")
            # Put data to gpu

            # print(f"=======================pixpro iou={self.calculate_iou_3d(crop_coords_q[0], crop_coords_k[0])}")

            source_data, source_seg_targets = source_data.to(device=self._device), source_seg_targets.to(device=self._device)
            data, seg_targets = data.to(device=self._device), seg_targets.to(device=self._device)
            pseudo_data = pseudo_data.to(device=self._device)
            pseudo_weak_data = pseudo_weak_data.to(device=self._device)
            data_q, data_k = data_q.to(device=self._device), data_k.to(device=self._device)
            crop_coords_q, crop_coords_k = crop_coords_q.to(device=self._device), crop_coords_k.to(device=self._device)
        
            # source
            source_det_targets = []
            for item in source_bboxes:
                target = {
                    'boxes': item[0].to(dtype=torch.float, device=self._device),
                    'labels': item[1].to(device=self._device)
                }
                source_det_targets.append(target)

            # target label
            det_targets = []
            for item in bboxes:
                target = {
                    'boxes': item[0].to(dtype=torch.float, device=self._device),
                    'labels': item[1].to(device=self._device)
                }
                det_targets.append(target)

            # target unlabel, only for debug
            gt_pseudo_weak_det_targets = []
            for item in gt_pseudo_weak_bboxes:
                target = {
                    'boxes': item[0].to(dtype=torch.float, device=self._device),
                    'labels': item[1].to(device=self._device)
                }
                gt_pseudo_weak_det_targets.append(target)

            gt_pseudo_det_targets = []
            for item in gt_pseudo_bboxes:
                gt_pseudo_target = {
                    'boxes': item[0].to(dtype=torch.float, device=self._device),
                    'labels': item[1].to(device=self._device)
                }
                gt_pseudo_det_targets.append(gt_pseudo_target)

            # conditions: target_labeled_train, pseudo_label, pixpro, domain loss
            if use_pretrain_pipeline:
                use_target_label = 'use_target_label' in self._config and self._config['use_target_label'] and (num_epoch < self._config['label_pretrain_stop_epoch'] or num_epoch >= self._config['pseudo_pretrain_stop_epoch'])
                use_pseudo_label = 'pseudo_label' in self._config and self._config['pseudo_label'] and num_epoch >= self._config['label_pretrain_stop_epoch'] and num_epoch < self._config['pseudo_pretrain_stop_epoch']
                use_pixpro = 'use_pixpro' in self._config['pixpro'] and self._config['pixpro']['use_pixpro']
                use_domain_loss = 'domain_loss' in self._config and self._config['domain_loss']
            else:
                use_target_label = 'use_target_label' in self._config and self._config['use_target_label']
                use_pseudo_label = 'pseudo_label' in self._config and self._config['pseudo_label']
                use_pixpro = 'use_pixpro' in self._config['pixpro'] and self._config['pixpro']['use_pixpro']
                use_domain_loss = 'domain_loss' in self._config and self._config['domain_loss']

            # print(f"===============use_target_label={use_target_label}, use_pseudo_label={use_pseudo_label}, use_pixpro={use_pixpro}, use_domain_loss={use_domain_loss}")

            # update teacher model every iteration
            # even no pseudo labels, teacher model can still be more stable
            if num_epoch < self._config['pseudo_start_ema_epoch']:
                # update all
                self._update_teacher_model(keep_rate=0.00)
            else:
                # update ema
                self._update_teacher_model(keep_rate=self._config['pseudo_ema_keep_rate'])

            '''
            # source + target
            if use_target_label:
                labeled_data = torch.cat((source_data, data), 0)
                labeled_det_targets = source_det_targets + det_targets
                labeled_seg_targets = torch.cat((source_seg_targets, seg_targets), 0)
            # only source
            else:
                labeled_data = source_data
                labeled_det_targets = source_det_targets
                labeled_seg_targets = source_seg_targets
            '''

            # Make prediction
            with autocast(): 
                # 1. source labeled data
                source_out, source_contrast_losses, source_dn_meta, source_det_srcs, = self._model(source_data, source_det_targets, num_epoch)
                source_loss_dict, source_pos_indices = self._criterion(source_out, source_det_targets, source_seg_targets, source_dn_meta, num_epoch=num_epoch)

                # 2. target labeled data
                out, contrast_losses, dn_meta, det_srcs, = self._model(data, det_targets, num_epoch)
                loss_dict, pos_indices = self._criterion(out, det_targets, seg_targets, dn_meta, num_epoch=num_epoch)

                # 3. generate pseudo labels
                if use_pseudo_label:
                    # use teacher model to get pseudo labels, weak augmentation
                    if 'pseudo_teacher_strong' in self._config and self._config['pseudo_teacher_strong']:
                        teacher_pseudo_data = pseudo_data
                    else:
                        teacher_pseudo_data = pseudo_weak_data

                    teacher_pseudo_out, _, _, _ = self._teacher_model(teacher_pseudo_data, targets=None, num_epoch=num_epoch)
                    pseudo_weak_det_targets = self.get_pseudo_labels_with_nms_3d(teacher_pseudo_out)
                    for i in range(len(pseudo_weak_det_targets)):
                        #print(f"==================pseudo boxes={pseudo_weak_det_targets[i]['boxes']}")
                        print(f"==================pseudo labels={pseudo_weak_det_targets[i]['labels']}")
                        print(f"==================pseudo path={pseudo_path}")
                        #print(f"==================pseudo gt boxes={gt_pseudo_det_targets[i]['boxes']}")
                        #print(f"==================pseudo gt labels={gt_pseudo_det_targets[i]['labels']}")
                        #print(f"==================pseudo weak gt boxes={gt_pseudo_weak_det_targets[i]['boxes']}")
                        #print(f"==================pseudo weak gt labels={gt_pseudo_weak_det_targets[i]['labels']}")
                        if pseudo_weak_det_targets[i]['boxes'].shape[0] > 0:
                            self.pseudo_accuracy(pseudo_weak_det_targets[i]['boxes'], gt_pseudo_weak_det_targets[i]['boxes'], pseudo_weak_det_targets[i]['labels'], gt_pseudo_weak_det_targets[i]['labels'])

                    # 4. use pseudo labels to train student, strong augmentation, labels are the same.
                    pseudo_out, _, _, pseudo_det_srcs = self._model(pseudo_data, pseudo_weak_det_targets, num_epoch)
                    # print(f"*****************************len(pseudo_det_srcs)={len(pseudo_det_srcs)}")
                    # print(f"*****************************pseudo_det_srcs[0].shape={pseudo_det_srcs[0].shape}")

                    # only keep perfect labels
                    if self._config.get('pseudo_perfect_label', False):
                        if self.tensor_equal_to_list_ignore_order(pseudo_weak_det_targets[0]['labels'], self._config['backbone']['num_organs']):
                            # print(f"*****************perfect label")
                            self._criterion._seg_msa = False
                            pseudo_loss_dict, _ = self._criterion(pseudo_out, pseudo_weak_det_targets, None, dn_meta, num_epoch=num_epoch)
                            self._criterion._seg_msa = True
                        else:
                            # print(f"*****************not perfect label")
                            with torch.no_grad():
                                pseudo_loss_dict = {'cls': zero_tensor, 'bbox': zero_tensor, 'giou': zero_tensor}
                    else:
                        self._criterion._seg_msa = False
                        pseudo_loss_dict, _ = self._criterion(pseudo_out, pseudo_weak_det_targets, None, dn_meta, num_epoch=num_epoch)
                        self._criterion._seg_msa = True

                    # avoid OOM
                    del pseudo_out
                else:
                    _, _, _, pseudo_det_srcs = self._model(pseudo_data, None, num_epoch)
                    pseudo_loss_dict = {'cls': zero_tensor, 'bbox': zero_tensor, 'giou': zero_tensor}

                # 5. pixpro
                if use_pixpro:
                    pixpro_loss, _ = self._pixpro_model(data_q, data_k, crop_coords_q, crop_coords_k)
                else:
                    pixpro_loss = torch.tensor(0)

                # 6. domain loss
                srcdom_loss = torch.tensor(0).to(dtype=torch.float, device=self._device)
                tgtdom_loss = torch.tensor(0).to(dtype=torch.float, device=self._device)

                source_count = 0
                target_count = 0
                if use_domain_loss:
                    source_feat = source_det_srcs[0][0].unsqueeze(0)
                    target_feat = pseudo_det_srcs[0][0].unsqueeze(0)
                    srcdom_loss += self.get_domain_loss(source_feat, domain='source')
                    tgtdom_loss += self.get_domain_loss(target_feat, domain='target')
                    source_count += 1
                    target_count += 1
                    if source_count > 0:
                        srcdom_loss = srcdom_loss / source_count
                    if target_count > 0:
                        tgtdom_loss = tgtdom_loss / target_count

                    '''
                    if use_target_label:
                        # only use P4, batch_size=1 for both source and target
                        # use strong augmentation now, todo: change to weak augmentation
                        source_feat = source_det_srcs[0][0].unsqueeze(0)
                        target_feat = det_srcs[0][0].unsqueeze(0)

                        if self._config['use_large_organ_labels']:
                            source_ins_feat = self.get_ins_feature(source_feat, source_det_targets, self._config['source_large_organ_labels'], self._device)
                            target_ins_feat = self.get_ins_feature(target_feat, det_targets, self._config['target_large_organ_labels'], self._device)
                            #print(f"=======================source_feat.shape={source_feat.shape}, source_ins_feat.shape={source_ins_feat.shape}")

                            srcdom_loss += self.get_domain_loss(source_ins_feat, domain='source')
                            tgtdom_loss += self.get_domain_loss(target_ins_feat, domain='target')
                        else:
                            srcdom_loss += self.get_domain_loss(source_feat, domain='source')
                            tgtdom_loss += self.get_domain_loss(target_feat, domain='target')

                        source_count += 1
                        target_count += 1
                    else:
                        source_feat = source_det_srcs[0][0].unsqueeze(0)
                        if self._config['use_large_organ_labels']:
                            source_ins_feat = self.get_ins_feature(source_feat, source_det_targets, self._config['source_large_organ_labels'], self._device)
                            srcdom_loss += self.get_domain_loss(source_ins_feat, domain='source')
                        else:
                            srcdom_loss += self.get_domain_loss(source_feat, domain='source')

                        source_count += 1

                    if not self._config['use_large_organ_labels'] and use_pseudo_label:
                        pseudo_feat = pseudo_det_srcs[0][0].unsqueeze(0)
                        tgtdom_loss += self.get_domain_loss(pseudo_feat, domain='target')
                        target_count += 1

                    # mean, count should never be 0
                    # print(f"==============source_count={source_count}, target_count={target_count}")
                    if source_count > 0:
                        srcdom_loss = srcdom_loss / source_count
                    if target_count > 0:
                        tgtdom_loss = tgtdom_loss / target_count
                    '''

                # print(f"**********************len(source_det_targets)={len(source_det_targets), source_det_targets[0]['labels']}")
                # 7. MMD loss
                if 'mmd_loss' in self._config and self._config['mmd_loss']:
                    # print(f"****************source_det_targets[0]['labels']={source_det_targets[0]['labels']}")
                    source_ins_0, source_labels_0 = self.get_ins_feature(source_det_srcs[0], source_det_targets, self._device)
                    target_ins_0, target_labels_0 = self.get_ins_feature(det_srcs[0], det_targets, self._device)
                    source_ins_1, source_labels_1 = self.get_ins_feature(source_det_srcs[1], source_det_targets, self._device)
                    target_ins_1, target_labels_1 = self.get_ins_feature(det_srcs[1], det_targets, self._device)
                    source_ins_2, source_labels_2 = self.get_ins_feature(source_det_srcs[2], source_det_targets, self._device)
                    target_ins_2, target_labels_2 = self.get_ins_feature(det_srcs[2], det_targets, self._device)

                    num_organs = self._config['backbone']['num_organs']
                    mmd_loss = self.class_wise_feature_loss(source_ins_0, source_labels_0, target_ins_0, target_labels_0, num_organs)
                    mmd_loss += self.class_wise_feature_loss(source_ins_1, source_labels_1, target_ins_1, target_labels_1, num_organs)
                    mmd_loss += self.class_wise_feature_loss(source_ins_2, source_labels_2, target_ins_2, target_labels_2, num_organs)
                    mmd_loss /= 3

                    if 'pseudo_mmd_loss' in self._config and self._config['pseudo_mmd_loss']:
                        pseudo_target_ins_0, pseudo_target_labels_0 = self.get_ins_feature(pseudo_det_srcs[0], pseudo_weak_det_targets, self._device)
                        pseudo_target_ins_1, pseudo_target_labels_1 = self.get_ins_feature(pseudo_det_srcs[1], pseudo_weak_det_targets, self._device)
                        pseudo_target_ins_2, pseudo_target_labels_2 = self.get_ins_feature(pseudo_det_srcs[2], pseudo_weak_det_targets, self._device)
                        pseudo_mmd_loss = self.class_wise_feature_loss(source_ins_0, source_labels_0, pseudo_target_ins_0, pseudo_target_labels_0, num_organs)
                        pseudo_mmd_loss += self.class_wise_feature_loss(source_ins_1, source_labels_1, pseudo_target_ins_1, pseudo_target_labels_1, num_organs)
                        pseudo_mmd_loss += self.class_wise_feature_loss(source_ins_2, source_labels_2, pseudo_target_ins_2, pseudo_target_labels_2, num_organs)
                        pseudo_mmd_loss /= 3
                        # print(f"**********************pseudo_mmd_loss={pseudo_mmd_loss}")
                        # pseudo ratio
                        pseudo_ratio = self._pseudo_cls_coef / self._config['loss_coefs']['cls']
                        pseudo_mmd_loss *= pseudo_ratio
                        # print(f"**********************after ration, pseudo_mmd_loss={pseudo_mmd_loss}")
                        mmd_loss += pseudo_mmd_loss
                else:
                    mmd_loss = torch.tensor(0)

                if self._criterion._seg_proxy: # log Hausdorff
                    hd95 = loss_dict['hd95'].item()
                del loss_dict['hd95'] # remove HD from loss, so it does not influence loss_abs
                del source_loss_dict['hd95'] # remove HD from loss, so it does not influence loss_abs

                if self._hybrid: # hybrid matching
                    outputs_one2many = dict()
                    outputs_one2many["pred_logits"] = out["pred_logits_one2many"]
                    outputs_one2many["pred_boxes"] = out["pred_boxes_one2many"]
                    outputs_one2many["aux_outputs"] = out["aux_outputs_one2many"]
                    outputs_one2many["seg_one2many"] = True
                    if self._dense_hybrid_criterion: # DM in additional branch
                        loss_dict_one2many, _ = self._dense_hybrid_criterion(outputs_one2many, det_targets, seg_targets) # det_targets replaces det_many_targets
                    else:  # regular one-to-many branch
                        det_many_targets = copy.deepcopy(det_targets)
                        # repeat the targets
                        for target in det_many_targets:
                            target["boxes"] = target["boxes"].repeat(self._hybrid_K, 1)
                            target["labels"] = target["labels"].repeat(self._hybrid_K)

                        loss_dict_one2many, _ = self._criterion(outputs_one2many, det_many_targets, seg_targets)
                    del loss_dict_one2many['hd95']
                    for key, value in loss_dict_one2many.items():
                        if key + "_one2many" in loss_dict.keys():
                            loss_dict[key + "_one2many"] += value * self._config['hybrid_loss_weight_one2many']
                        else:
                            loss_dict[key + "_one2many"] = value * self._config['hybrid_loss_weight_one2many']

                # 1. source supervised loss
                source_sup_loss = 0
                for loss_key, loss_val in source_loss_dict.items():
                    source_sup_loss += loss_val * self._config['loss_coefs'][loss_key.split('_')[0]]

                # 2. target supervised loss
                sup_loss = 0
                for loss_key, loss_val in loss_dict.items():
                    sup_loss += loss_val * self._config['loss_coefs'][loss_key.split('_')[0]]

                # weighted supervised loss
                total_sup_loss = (source_sup_loss * self._config['source_loss_coef'] + sup_loss) / (1 + self._config['source_loss_coef'])
                # print(f"=================source_sup_loss={source_sup_loss.item()}, sup_loss={sup_loss.item()}, total_sup_loss={total_sup_loss.item()}")

                source_loss_abs = 0
                loss_abs = 0

                # 3. source domain loss
                source_loss_abs += srcdom_loss * self._srcdom_loss_coef

                # 4. target domain loss
                loss_abs += tgtdom_loss * self._tgtdom_loss_coef

                # mmd loss
                if 'mmd_loss' in self._config and self._config['mmd_loss']:
                    loss_abs += mmd_loss * self._config['mmd_loss_coef']

                # 5. pixpro loss, put it into target loss 
                loss_abs += pixpro_loss * self._config['loss_coefs']['pixpro']

                # 6. pseudo loss, put it into target loss
                loss_abs += pseudo_loss_dict['cls'] * self._pseudo_cls_coef
                if 'pseudo_bbox' in self._config and self._config['pseudo_bbox']:
                    loss_abs += pseudo_loss_dict['bbox'] * self._pseudo_cls_coef * self._config['pseudo_bbox_cls_ratio']
                    loss_abs += pseudo_loss_dict['giou'] * self._pseudo_cls_coef * self._config['pseudo_giou_cls_ratio']

                for loss_key, loss_val in contrast_losses.items():
                    loss_abs += loss_val # already multiplied coefficient in organdetrnet.py
                    loss_contrast_agg += loss_val 

            total_loss_agg += source_loss_abs.item()
            total_loss_agg += loss_abs.item()
            total_loss_agg += total_sup_loss.item()

            total_loss_abs = source_loss_abs + loss_abs + total_sup_loss

            if self._config['gradient_accu']:
                accum_iter = self._config["gradient_accu_iter"]
                total_loss_abs = total_loss_abs / accum_iter
                # self._optimizer.zero_grad()
                self._scaler.scale(total_loss_abs).backward()
                
                # Clip grads to counter exploding grads
                max_norm = self._config['clip_max_norm']
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm)

                if ((idx + 1) % accum_iter == 0) or (idx + 1 == len(self._labeled_train_loader)):
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                    self._optimizer.zero_grad()
            else:
                self._optimizer.zero_grad()
                self._scaler.scale(total_loss_abs).backward()

                # Clip grads to counter exploding grads
                max_norm = self._config['clip_max_norm']
                if max_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm)
                    # print(f"=============idx={idx}, grad_norm={grad_norm}, source_domain_loss={loss_dict['srcdom'].item()}, target_domain_loss={loss_dict['tgtdom'].item()}")

                self._scaler.step(self._optimizer)
                self._scaler.update()

            # log gradients of positive & negative queries
            if self.log_grad:
                for name, param in self._model.named_parameters():
                    if name == 'query_embed.weight':
                        _, tgt_param_grad = torch.split(param.grad, param.size(1)//2, dim=1) # only fetch grads of tgt
                        #tgt_param_grad = param.grad
                        neg_indices = param.new_ones(param.size(0)).bool()

                        neg_indices[pos_indices[0][0]] = False
                        pos_query_grads = tgt_param_grad[pos_indices[0][0]]
                        neg_query_grads = tgt_param_grad[neg_indices]
                        # remove nan
                        pos_query_grads[torch.isnan(pos_query_grads)] = torch.tensor(0.0)
                        neg_query_grads[torch.isnan(neg_query_grads)] = torch.tensor(0.0)
                        pos_query_grads_list = torch.cat((pos_query_grads_list.cuda(), pos_query_grads), dim=0)
                        neg_query_grads_list = torch.cat((neg_query_grads_list.cuda(), neg_query_grads), dim=0)

            loss_bbox_agg += loss_dict['bbox'].item()
            loss_giou_agg += loss_dict['giou'].item()
            loss_cls_agg += loss_dict['cls'].item()
            loss_seg_ce_agg += loss_dict['segce'].item()
            loss_seg_dice_agg += loss_dict['segdice'].item()

            source_loss_bbox_agg += source_loss_dict['bbox'].item()
            source_loss_giou_agg += source_loss_dict['giou'].item()
            source_loss_cls_agg += source_loss_dict['cls'].item()
            source_loss_seg_ce_agg += source_loss_dict['segce'].item()
            source_loss_seg_dice_agg += source_loss_dict['segdice'].item()

            loss_pseudo_cls_agg += pseudo_loss_dict['cls'].item()
            loss_pseudo_bbox_agg += pseudo_loss_dict['bbox'].item()
            loss_pseudo_giou_agg += pseudo_loss_dict['giou'].item()

            loss_pixpro_agg += pixpro_loss.item()

            loss_source_domain_agg += srcdom_loss.item() 
            loss_target_domain_agg += tgtdom_loss.item()

            loss_mmd_agg += mmd_loss.item()

            if self._hybrid: # hybrid matching
                loss_bbox_one2many_agg += loss_dict['bbox_one2many'].item()
                loss_giou_one2many_agg += loss_dict['giou_one2many'].item()
                loss_cls_one2many_agg += loss_dict['cls_one2many'].item()
                loss_seg_ce_one2many_agg += loss_dict['segce_one2many'].item()
                loss_seg_dice_one2many_agg += loss_dict['segdice_one2many'].item()

            if self._criterion._seg_proxy: # log Hausdorff
                hd95_agg += hd95
            
            if dn_meta is not None:
                if len(loss_dn_agg) == 0: # initialize loss entries
                    loss_dn_agg['bbox_dn'] = 0
                    loss_dn_agg['giou_dn'] = 0
                    loss_dn_agg['cls_dn'] = 0
                else:
                    loss_dn_agg['bbox_dn'] += loss_dict[f'bbox_dn'].item()
                    loss_dn_agg['giou_dn'] += loss_dict[f'giou_dn'].item()
                    loss_dn_agg['cls_dn'] += loss_dict[f'cls_dn'].item()
            if "aux_outputs" in out:
                if len(loss_aux_agg) == 0: # initialize loss entries
                    for i in range(len(out["aux_outputs"])):
                        loss_aux_agg[f'bbox_{i}'] = 0
                        loss_aux_agg[f'giou_{i}'] = 0
                        loss_aux_agg[f'cls_{i}'] = 0
                for i in range(len(out["aux_outputs"])):
                    loss_aux_agg[f'bbox_{i}'] += loss_dict[f'bbox_{i}'].item()
                    loss_aux_agg[f'giou_{i}'] += loss_dict[f'giou_{i}'].item()
                    loss_aux_agg[f'cls_{i}'] += loss_dict[f'cls_{i}'].item()
                
                if dn_meta is not None:
                    if len(loss_aux_agg_dn) == 0: # initialize loss entries
                        for i in range(len(out["aux_outputs"])):
                            loss_aux_agg_dn[f'bbox_{i}_dn'] = 0
                            loss_aux_agg_dn[f'giou_{i}_dn'] = 0
                            loss_aux_agg_dn[f'cls_{i}_dn'] = 0
                    for i in range(len(out["aux_outputs"])):
                        loss_aux_agg_dn[f'bbox_{i}_dn'] += loss_dict[f'bbox_{i}_dn'].item()
                        loss_aux_agg_dn[f'giou_{i}_dn'] += loss_dict[f'giou_{i}_dn'].item()
                        loss_aux_agg_dn[f'cls_{i}_dn'] += loss_dict[f'cls_{i}_dn'].item()
            if "enc_outputs" in out:
                loss_enc_bbox_agg += loss_dict['bbox_enc'].item()
                loss_enc_giou_agg += loss_dict['giou_enc'].item()
                loss_enc_cls_agg += loss_dict['cls_enc'].item()
            memory_allocated, memory_cached = get_gpu_memory(self._device)
            progress_bar.set_postfix({'cached': "{:.2f}GB".format(memory_cached/(1024**3))})
            

        # source_loss = source_loss_agg / len(self._labeled_train_loader)
        # loss = loss_agg / len(self._labeled_train_loader)
        loss = total_loss_agg / len(self._labeled_train_loader)
        #print(f'total train loss for epoch {num_epoch}: '+str(loss))
        loss_bbox = loss_bbox_agg / len(self._labeled_train_loader)
        loss_giou = loss_giou_agg / len(self._labeled_train_loader)
        loss_cls = loss_cls_agg / len(self._labeled_train_loader)
        loss_seg_ce = loss_seg_ce_agg / len(self._labeled_train_loader)
        loss_seg_dice = loss_seg_dice_agg / len(self._labeled_train_loader)

        source_loss_bbox = source_loss_bbox_agg / len(self._labeled_train_loader)
        source_loss_giou = source_loss_giou_agg / len(self._labeled_train_loader)
        source_loss_cls = source_loss_cls_agg / len(self._labeled_train_loader)
        source_loss_seg_ce = source_loss_seg_ce_agg / len(self._labeled_train_loader)
        source_loss_seg_dice = source_loss_seg_dice_agg / len(self._labeled_train_loader)

        loss_source_sup = source_loss_bbox + source_loss_giou + source_loss_cls + source_loss_seg_ce + source_loss_seg_dice
        loss_sup = loss_bbox + loss_giou + loss_cls + loss_seg_ce + loss_seg_dice

        loss_pseudo_cls = loss_pseudo_cls_agg / len(self._labeled_train_loader)
        loss_pseudo_bbox = loss_pseudo_bbox_agg / len(self._labeled_train_loader)
        loss_pseudo_giou = loss_pseudo_giou_agg / len(self._labeled_train_loader)

        loss_pixpro = loss_pixpro_agg / len(self._labeled_train_loader)
        loss_source_domain = loss_source_domain_agg / len(self._labeled_train_loader)
        loss_target_domain = loss_target_domain_agg / len(self._labeled_train_loader)

        loss_mmd = loss_mmd_agg / len(self._labeled_train_loader)

        if self._hybrid: # hybrid matching
            loss_bbox_one2many = loss_bbox_one2many_agg / len(self._labeled_train_loader)
            loss_giou_one2many = loss_giou_one2many_agg / len(self._labeled_train_loader)
            loss_cls_one2many = loss_cls_one2many_agg / len(self._labeled_train_loader)
            loss_seg_ce_one2many = loss_seg_ce_one2many_agg / len(self._labeled_train_loader)
            loss_seg_dice_one2many = loss_seg_dice_one2many_agg / len(self._labeled_train_loader)
        
        if self._criterion._seg_proxy:  # log Hausdorff
            seg_hd95 = hd95_agg / len(self._labeled_train_loader)
        else:
            seg_hd95 = 0

        loss_contrast = loss_contrast_agg / len(self._labeled_train_loader)
        
        if len(loss_dn_agg) != 0:
            for key in loss_dn_agg:
                value = loss_dn_agg[key] / len(self._labeled_train_loader)
                self._writer.add_scalar("dn/"+key, value, num_epoch)
        if len(loss_aux_agg) != 0:
            for key in loss_aux_agg:
                value = loss_aux_agg[key] / len(self._labeled_train_loader)
                self._writer.add_scalar("train_aux/"+key, value, num_epoch)
        if len(loss_aux_agg_dn) != 0:
            for key in loss_aux_agg_dn:
                value = loss_aux_agg_dn[key] / len(self._labeled_train_loader)
                self._writer.add_scalar("dn/"+key, value, num_epoch)
        if loss_enc_bbox_agg or loss_enc_giou_agg or loss_enc_cls_agg:
            self._writer.add_scalar("train_enc/bbox_enc", loss_enc_bbox_agg/len(self._labeled_train_loader), num_epoch)
            self._writer.add_scalar("train_enc/giou_enc", loss_enc_giou_agg/len(self._labeled_train_loader), num_epoch)
            self._writer.add_scalar("train_enc/cls_enc", loss_enc_cls_agg/len(self._labeled_train_loader), num_epoch)

        # remove completely, not the same as dynamic domain loss
        if 'remove_domain_loss_in_the_middle' in self._config and self._config['remove_domain_loss_in_the_middle'] and self._config['remove_domain_loss_at_epoch'] == num_epoch:
            self._srcdom_loss_coef = 0
            self._tgtdom_loss_coef = 0
            dynamic_domain_loss = False
        else:
            dynamic_domain_loss = 'dynamic_domain_loss' in self._config and self._config['dynamic_domain_loss']

        # dynamic pseudo
        dynamic_pseudo = self._config['dynamic_pseudo'] 

        # dynamic augmentation
        dynamic_augmentation = self._config['augmentation']['dynamic']
        
        # for pseudo_cls_coef
        self._avg_cls += loss_cls

        avg_epoch = self._config['pseudo_avg_epoch']
        if 'pseudo_cls_step' in self._config:
            pseudo_cls_step = self._config['pseudo_cls_step']
        else:
            pseudo_cls_step = 0.1

        # update pseudo_cls and augmentations
        if num_epoch % avg_epoch == 0:
            # mono increase
            if 'mono_dynamic_pseudo' in self._config and self._config['mono_dynamic_pseudo']:
                self._pseudo_cls_coef = min(self._pseudo_cls_coef + pseudo_cls_step, self._config['pseudo_max_coef'])
            else:
                self._avg_cls /= avg_epoch
                if self._avg_cls < self._config['pseudo_cls_threshold']:
                    if dynamic_pseudo:
                        self._pseudo_cls_coef = min(self._pseudo_cls_coef + pseudo_cls_step, self._config['pseudo_max_coef'])

                    # domain loss, opposite to pseudo loss
                    if dynamic_domain_loss:
                        self._srcdom_loss_coef = max(self._srcdom_loss_coef - 0.1, self._config['domain_loss_min'])
                        self._tgtdom_loss_coef = max(self._tgtdom_loss_coef - 0.1, self._config['domain_loss_min'])

                    if dynamic_augmentation:
                        self._config['augmentation']['p_gaussian_noise'] = min(self._config['augmentation']['p_gaussian_noise'] + 0.05, self._config['augmentation']['p_gaussian_noise_max'])
                        self._config['augmentation']['p_gaussian_smooth'] = min(self._config['augmentation']['p_gaussian_smooth'] + 0.05, self._config['augmentation']['p_gaussian_smooth_max'])
                        # self._config['augmentation']['p_intensity_scale'] = min(self._config['augmentation']['p_intensity_scale'] + 0.05, self._config['augmentation']['p_intensity_scale_max'])
                        # self._config['augmentation']['p_intensity_shift'] = min(self._config['augmentation']['p_intensity_shift'] + 0.05, self._config['augmentation']['p_intensity_shift_max'])
                        self._config['augmentation']['p_adjust_contrast'] = min(self._config['augmentation']['p_adjust_contrast'] + 0.05, self._config['augmentation']['p_adjust_contrast_max'])
                else:
                    if dynamic_pseudo:
                        self._pseudo_cls_coef = max(self._pseudo_cls_coef - pseudo_cls_step, self._config['pseudo_min_coef'])

                    # domain loss, opposite to pseudo loss
                    if dynamic_domain_loss:
                        self._srcdom_loss_coef = min(self._srcdom_loss_coef + 0.1, self._config['domain_loss_max'])
                        self._tgtdom_loss_coef = min(self._tgtdom_loss_coef + 0.1, self._config['domain_loss_max'])

                    if dynamic_augmentation:
                        self._config['augmentation']['p_gaussian_noise'] = max(self._config['augmentation']['p_gaussian_noise'] - 0.05, self._config['augmentation']['p_gaussian_noise_min'])
                        self._config['augmentation']['p_gaussian_smooth'] = max(self._config['augmentation']['p_gaussian_smooth'] - 0.05, self._config['augmentation']['p_gaussian_smooth_min'])
                        # self._config['augmentation']['p_intensity_scale'] = max(self._config['augmentation']['p_intensity_scale'] - 0.05, self._config['augmentation']['p_intensity_scale_min'])
                        # self._config['augmentation']['p_intensity_shift'] = max(self._config['augmentation']['p_intensity_shift'] - 0.05, self._config['augmentation']['p_intensity_shift_min'])
                        self._config['augmentation']['p_adjust_contrast'] = max(self._config['augmentation']['p_adjust_contrast'] - 0.05, self._config['augmentation']['p_adjust_contrast_min'])

                if dynamic_augmentation:
                    self._source_train_loader.dataset.reset_transforms(self._config)
                    self._labeled_train_loader.dataset.reset_transforms(self._config)
                    self._pseudo_train_loader.dataset.reset_transforms(self._config)
                    self._pixpro_train_loader.dataset.reset_transforms(self._config)

                self._avg_cls = 0

        if self._hybrid: # log many2one just if hybrid matching is activated
            self._write_to_logger(
                    num_epoch, 'train', 
                    # source_total_loss=source_loss,
                    total_loss=loss,
                    bbox_loss=loss_bbox,
                    giou_loss=loss_giou,
                    cls_loss=loss_cls,
                    seg_ce_loss=loss_seg_ce,
                    seg_dice_loss=loss_seg_dice,
                    source_bbox_loss=source_loss_bbox,
                    source_giou_loss=source_loss_giou,
                    source_cls_loss=source_loss_cls,
                    source_seg_ce_loss=source_loss_seg_ce,
                    source_seg_dice_loss=source_loss_seg_dice,
                    pseudo_cls_loss=loss_pseudo_cls,
                    pixpro_loss=loss_pixpro,
                    domain_source_loss=loss_source_domain,
                    domain_target_loss=loss_target_domain,
                    seg_hd95=seg_hd95, # log Hausdorff
                    bbox_loss_one2many = loss_bbox_one2many,
                    giou_loss_one2many = loss_giou_one2many,
                    cls_loss_one2many = loss_cls_one2many,
                    seg_ce_loss_one2many = loss_seg_ce_one2many,
                    seg_dice_loss_one2many = loss_seg_dice_one2many,
            )
        else:
            self._write_to_logger(
                num_epoch, 'train', 
                # source_total_loss=source_loss,
                total_loss=loss,
                bbox_loss=loss_bbox,
                giou_loss=loss_giou,
                cls_loss=loss_cls,
                seg_ce_loss=loss_seg_ce,
                seg_dice_loss=loss_seg_dice,
                source_bbox_loss=source_loss_bbox,
                source_giou_loss=source_loss_giou,
                source_cls_loss=source_loss_cls,
                source_seg_ce_loss=source_loss_seg_ce,
                source_seg_dice_loss=source_loss_seg_dice,
                sup_loss = loss_sup,
                source_sup_loss = loss_source_sup,
                pseudo_cls_loss=loss_pseudo_cls,
                pseudo_bbox_loss=loss_pseudo_bbox,
                pseudo_giou_loss=loss_pseudo_giou,
                pixpro_loss=loss_pixpro,
                domain_source_loss=loss_source_domain,
                domain_target_loss=loss_target_domain,
                mmd_loss=loss_mmd,
                seg_hd95=seg_hd95, # log Hausdorff
                contrast_loss=loss_contrast,
                pseudo_cls_coef=self._pseudo_cls_coef,
            )

            print(f"==================dynamic_pseudo={dynamic_pseudo}, dynamic_augmentation={dynamic_augmentation}, dynamic_domain_loss={dynamic_domain_loss}")
            print(f"==================epoch={num_epoch},total_loss={loss},source_sup_loss={loss_source_sup},sup_loss={loss_sup},pixpro_loss={loss_pixpro},domain_source_loss={loss_source_domain},domain_target_loss={loss_target_domain},pseudo_cls_loss={loss_pseudo_cls},pseudo_bbox_loss={loss_pseudo_bbox},pseudo_giou_loss={loss_pseudo_giou},mmd_loss={loss_mmd}")
            print(f"==================avg_cls={self._avg_cls},pseudo_cls_coef={self._pseudo_cls_coef},domain_loss={self._srcdom_loss_coef},noise={self._config['augmentation']['p_gaussian_noise']},smooth={self._config['augmentation']['p_gaussian_smooth']},scale={self._config['augmentation']['p_intensity_scale']},shift={self._config['augmentation']['p_intensity_shift']},contrast={self._config['augmentation']['p_adjust_contrast']},")

            
            if self.log_grad and num_epoch % self.log_grad_every_epoch == 0:
                self.log_grads_list_pos.append(torch.flatten(pos_query_grads_list, 0).cpu())
                self.log_grads_list_neg.append(torch.flatten(neg_query_grads_list, 0).cpu())
                self.log_epoch_list.append(num_epoch)
                
                avg_pos_queries_grad = torch.abs(pos_query_grads_list).mean()
                avg_neg_queries_grad = torch.abs(neg_query_grads_list).mean()
                self._writer.add_scalar("grads/avg_pos_queries_grad", avg_pos_queries_grad, num_epoch)
                self._writer.add_scalar("grads/avg_neg_queries_grad", avg_neg_queries_grad, num_epoch)
                
                
                for name, param in self._model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self._writer.add_histogram('grads/' + name, param.grad, int(num_epoch))
                
    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        student_model_dict = self._model.state_dict()

        new_teacher_dict = OrderedDict() 
        for key, value in self._teacher_model.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self._teacher_model.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _update_student_model(self, keep_rate=0.9996):
        teacher_model_dict = self._teacher_model.state_dict()

        new_dict = OrderedDict() 
        for key, value in self._model.state_dict().items():
            if key in teacher_model_dict.keys():
                new_dict[key] = (
                    teacher_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self._model.load_state_dict(new_dict)

    @torch.no_grad()
    def _validate(self, num_epoch):
        
        # self._model.eval()
        # use teacher model to validate
        self._teacher_model.eval()
        # self._criterion.eval()

        loss_agg = 0
        loss_bbox_agg = 0
        loss_giou_agg = 0
        loss_cls_agg = 0
        loss_seg_ce_agg = 0
        loss_seg_dice_agg = 0
        # log Hausdorff
        hd95_agg = 0
        # Aux loss
        loss_aux_agg = {}
        # Two stage
        loss_enc_bbox_agg = 0
        loss_enc_giou_agg = 0
        loss_enc_cls_agg = 0
        progress_bar = tqdm(self._val_loader)
        for data, _, _, bboxes, _, seg_targets, _, _ in progress_bar:
            # Put data to gpu
            data, seg_targets = data.to(device=self._device), seg_targets.to(device=self._device)
        
            det_targets = []
            for item in bboxes:
                target = {
                    'boxes': item[0].to(dtype=torch.float, device=self._device),
                    'labels': item[1].to(device=self._device)
                }
                det_targets.append(target)

            # Make prediction
            with autocast():
                out = self._teacher_model(data)
                loss_dict, _ = self._criterion(out, det_targets, seg_targets)

                if self._criterion._seg_proxy: # log Hausdorff
                    hd95 = loss_dict['hd95'].item()
                del loss_dict['hd95'] # remove HD from loss, so it does not influence loss_abs

                # Create absolute loss and mult with loss coefficient
                loss_abs = 0
                for loss_key, loss_val in loss_dict.items():
                    loss_abs += loss_val * self._config['loss_coefs'][loss_key.split('_')[0]]

            # Evaluate validation predictions based on metric
            pred_boxes, pred_classes, pred_scores = inference(out)
            self._evaluator.add(
                pred_boxes=pred_boxes,
                pred_classes=pred_classes,
                pred_scores=pred_scores,
                gt_boxes=[target['boxes'].detach().cpu().numpy() for target in det_targets],
                gt_classes=[target['labels'].detach().cpu().numpy() for target in det_targets]
            )

            loss_agg += loss_abs.item()
            loss_bbox_agg += loss_dict['bbox'].item()
            loss_giou_agg += loss_dict['giou'].item()
            loss_cls_agg += loss_dict['cls'].item()
            loss_seg_ce_agg += loss_dict['segce'].item()
            loss_seg_dice_agg += loss_dict['segdice'].item()
            if self._criterion._seg_proxy: # log Hausdorff
                hd95_agg += hd95

            
            if "aux_outputs" in out:
                if len(loss_aux_agg) == 0: # initialize loss entries
                    for i in range(len(out["aux_outputs"])):
                        loss_aux_agg[f'bbox_{i}'] = 0
                        loss_aux_agg[f'giou_{i}'] = 0
                        loss_aux_agg[f'cls_{i}'] = 0
                for i in range(len(out["aux_outputs"])):
                    loss_aux_agg[f'bbox_{i}'] += loss_dict[f'bbox_{i}'].item()
                    loss_aux_agg[f'giou_{i}'] += loss_dict[f'giou_{i}'].item()
                    loss_aux_agg[f'cls_{i}'] += loss_dict[f'cls_{i}'].item()
                
            if "enc_outputs" in out:
                loss_enc_bbox_agg += loss_dict['bbox_enc'].item()
                loss_enc_giou_agg += loss_dict['giou_enc'].item()
                loss_enc_cls_agg += loss_dict['cls_enc'].item()
            memory_allocated, memory_cached = get_gpu_memory(self._device)
            progress_bar.set_postfix({'cached': "{:.2f}GB".format(memory_cached/(1024**3))})
                
        loss = loss_agg / len(self._val_loader)
        loss_bbox = loss_bbox_agg / len(self._val_loader)
        loss_giou = loss_giou_agg / len(self._val_loader)
        loss_cls = loss_cls_agg / len(self._val_loader)
        loss_seg_ce = loss_seg_ce_agg / len(self._val_loader)
        loss_seg_dice = loss_seg_dice_agg / len(self._val_loader)
        if self._criterion._seg_proxy:  # log Hausdorff
            seg_hd95 = hd95_agg / len(self._val_loader)
        else:
            seg_hd95 = 0

        metric_scores = self._evaluator.eval()
        self._evaluator.reset()

        # Check if new best checkpoint
        if metric_scores[self._main_metric_key] >= self._main_metric_max_val \
            and not self._config['debug_mode']:
            self._main_metric_max_val = metric_scores[self._main_metric_key]
            self._save_checkpoint(
                num_epoch,
                f'model_best_{metric_scores[self._main_metric_key]:.3f}_in_ep{num_epoch}.pt'
            )
            
        if len(loss_aux_agg) != 0:
            for key in loss_aux_agg:
                value = loss_aux_agg[key] / len(self._val_loader)
                self._writer.add_scalar("val_aux/"+key, value, num_epoch)
        if loss_enc_bbox_agg or loss_enc_giou_agg or loss_enc_cls_agg:
            self._writer.add_scalar("val_enc/bbox_enc", loss_enc_bbox_agg/len(self._val_loader), num_epoch)
            self._writer.add_scalar("val_enc/giou_enc", loss_enc_giou_agg/len(self._val_loader), num_epoch)
            self._writer.add_scalar("val_enc/cls_enc", loss_enc_cls_agg/len(self._val_loader), num_epoch)

        # Write to logger
        self._write_to_logger(
            num_epoch, 'val', 
            total_loss=loss,
            bbox_loss=loss_bbox,
            giou_loss=loss_giou,
            cls_loss=loss_cls,
            seg_ce_loss=loss_seg_ce,
            seg_dice_loss=loss_seg_dice
        )

        self._write_to_logger(
            num_epoch, 'val_metric',
            mAPcoco=metric_scores['mAP_coco'],
            mAPcocoS=metric_scores['mAP_coco_s'],
            mAPcocoM=metric_scores['mAP_coco_m'],
            mAPcocoL=metric_scores['mAP_coco_l'],
            mAPnndet=metric_scores['mAP_nndet'],
            AP10=metric_scores['AP_IoU_0.10'],
            AP50=metric_scores['AP_IoU_0.50'],
            AP75=metric_scores['AP_IoU_0.75'],
            seg_hd95=seg_hd95 # log Hausdorff
        )

        print(f"===============================epoch={num_epoch}, mAP_coco={metric_scores['mAP_coco']}") 
        
    @torch.no_grad()
    def _validate_student(self, num_epoch):
        
        # self._model.eval()
        # use teacher model to validate
        self._model.eval()
        # self._criterion.eval()

        loss_agg = 0
        loss_bbox_agg = 0
        loss_giou_agg = 0
        loss_cls_agg = 0
        loss_seg_ce_agg = 0
        loss_seg_dice_agg = 0
        # log Hausdorff
        hd95_agg = 0
        # Aux loss
        loss_aux_agg = {}
        # Two stage
        loss_enc_bbox_agg = 0
        loss_enc_giou_agg = 0
        loss_enc_cls_agg = 0
        progress_bar = tqdm(self._val_loader)
        for data, _, _, bboxes, _, seg_targets, _, _ in progress_bar:
            # Put data to gpu
            data, seg_targets = data.to(device=self._device), seg_targets.to(device=self._device)
        
            det_targets = []
            for item in bboxes:
                target = {
                    'boxes': item[0].to(dtype=torch.float, device=self._device),
                    'labels': item[1].to(device=self._device)
                }
                det_targets.append(target)

            # Make prediction
            with autocast():
                out = self._model(data)
                loss_dict, _ = self._criterion(out, det_targets, seg_targets)

                if self._criterion._seg_proxy: # log Hausdorff
                    hd95 = loss_dict['hd95'].item()
                del loss_dict['hd95'] # remove HD from loss, so it does not influence loss_abs

                # Create absolute loss and mult with loss coefficient
                loss_abs = 0
                for loss_key, loss_val in loss_dict.items():
                    loss_abs += loss_val * self._config['loss_coefs'][loss_key.split('_')[0]]

            # Evaluate validation predictions based on metric
            pred_boxes, pred_classes, pred_scores = inference(out)
            self._evaluator.add(
                pred_boxes=pred_boxes,
                pred_classes=pred_classes,
                pred_scores=pred_scores,
                gt_boxes=[target['boxes'].detach().cpu().numpy() for target in det_targets],
                gt_classes=[target['labels'].detach().cpu().numpy() for target in det_targets]
            )

            loss_agg += loss_abs.item()
            loss_bbox_agg += loss_dict['bbox'].item()
            loss_giou_agg += loss_dict['giou'].item()
            loss_cls_agg += loss_dict['cls'].item()
            loss_seg_ce_agg += loss_dict['segce'].item()
            loss_seg_dice_agg += loss_dict['segdice'].item()
            if self._criterion._seg_proxy: # log Hausdorff
                hd95_agg += hd95

            
            if "aux_outputs" in out:
                if len(loss_aux_agg) == 0: # initialize loss entries
                    for i in range(len(out["aux_outputs"])):
                        loss_aux_agg[f'bbox_{i}'] = 0
                        loss_aux_agg[f'giou_{i}'] = 0
                        loss_aux_agg[f'cls_{i}'] = 0
                for i in range(len(out["aux_outputs"])):
                    loss_aux_agg[f'bbox_{i}'] += loss_dict[f'bbox_{i}'].item()
                    loss_aux_agg[f'giou_{i}'] += loss_dict[f'giou_{i}'].item()
                    loss_aux_agg[f'cls_{i}'] += loss_dict[f'cls_{i}'].item()
                
            if "enc_outputs" in out:
                loss_enc_bbox_agg += loss_dict['bbox_enc'].item()
                loss_enc_giou_agg += loss_dict['giou_enc'].item()
                loss_enc_cls_agg += loss_dict['cls_enc'].item()
            memory_allocated, memory_cached = get_gpu_memory(self._device)
            progress_bar.set_postfix({'cached': "{:.2f}GB".format(memory_cached/(1024**3))})
                
        loss = loss_agg / len(self._val_loader)
        loss_bbox = loss_bbox_agg / len(self._val_loader)
        loss_giou = loss_giou_agg / len(self._val_loader)
        loss_cls = loss_cls_agg / len(self._val_loader)
        loss_seg_ce = loss_seg_ce_agg / len(self._val_loader)
        loss_seg_dice = loss_seg_dice_agg / len(self._val_loader)
        if self._criterion._seg_proxy:  # log Hausdorff
            seg_hd95 = hd95_agg / len(self._val_loader)
        else:
            seg_hd95 = 0

        metric_scores = self._evaluator.eval()
        self._evaluator.reset()

        # Check if new best checkpoint
        if metric_scores[self._main_metric_key_student] >= self._main_metric_max_val_student \
            and not self._config['debug_mode']:
            self._main_metric_max_val_student = metric_scores[self._main_metric_key_student]
            self._save_checkpoint(
                num_epoch,
                f'model_bst_student{metric_scores[self._main_metric_key_student]:.3f}_in_ep{num_epoch}.pt'
            )
            
        if len(loss_aux_agg) != 0:
            for key in loss_aux_agg:
                value = loss_aux_agg[key] / len(self._val_loader)
                self._writer.add_scalar("val_aux/"+key, value, num_epoch)
        if loss_enc_bbox_agg or loss_enc_giou_agg or loss_enc_cls_agg:
            self._writer.add_scalar("val_enc/bbox_enc", loss_enc_bbox_agg/len(self._val_loader), num_epoch)
            self._writer.add_scalar("val_enc/giou_enc", loss_enc_giou_agg/len(self._val_loader), num_epoch)
            self._writer.add_scalar("val_enc/cls_enc", loss_enc_cls_agg/len(self._val_loader), num_epoch)

        # Write to logger
        self._write_to_logger(
            num_epoch, 'student_val', 
            total_loss=loss,
            bbox_loss=loss_bbox,
            giou_loss=loss_giou,
            cls_loss=loss_cls,
            seg_ce_loss=loss_seg_ce,
            seg_dice_loss=loss_seg_dice
        )

        self._write_to_logger(
            num_epoch, 'student_val_metric',
            mAPcoco=metric_scores['mAP_coco'],
            mAPcocoS=metric_scores['mAP_coco_s'],
            mAPcocoM=metric_scores['mAP_coco_m'],
            mAPcocoL=metric_scores['mAP_coco_l'],
            mAPnndet=metric_scores['mAP_nndet'],
            AP10=metric_scores['AP_IoU_0.10'],
            AP50=metric_scores['AP_IoU_0.50'],
            AP75=metric_scores['AP_IoU_0.75'],
            seg_hd95=seg_hd95 # log Hausdorff
        )

        print(f"===============================student, epoch={num_epoch}, mAP_coco={metric_scores['mAP_coco']}")

    def run(self):
        '''
        if self._epoch_to_start == 0:   # For initial performance estimation
            self._validate(0)
        '''
        

        for epoch in range(self._epoch_to_start + 1, self._config['epochs'] + 1):
            print("starting epoch ", epoch)
            '''
            if epoch > 5:
                torch.autograd.set_detect_anomaly(True)
            '''

            self._train_one_epoch(epoch)

            # Log learning rates
            self._write_to_logger(
                epoch, 'lr',
                backbone=self._optimizer.param_groups[0]['lr'],
                neck=self._optimizer.param_groups[1]['lr']
            )
            
            if epoch % 50 == 0 and self.log_grad:
                print('logging grads boxplot...')
                plot_buf_pos = gen_box_plot(self.log_grads_list_pos, self.log_epoch_list, 'pos queries')
                plot_buf_neg = gen_box_plot(self.log_grads_list_neg, self.log_epoch_list, 'neg queries')
                image_pos, image_neg = PIL.Image.open(plot_buf_pos), PIL.Image.open(plot_buf_neg)
                image_pos, image_neg = ToTensor()(image_pos), ToTensor()(image_neg)
                self._writer.add_image('boxplot of pos queries grads', image_pos, epoch)
                self._writer.add_image('boxplot of neg queries grads', image_neg, epoch)

            if epoch % self._config['val_interval'] == 0:
                self._validate(epoch)
                self._validate_student(epoch)

            # Only schedule in the final fine-tuning process
            use_pretrain_pipeline = 'use_pretrain_pipeline' in self._config and self._config['use_pretrain_pipeline']
            if use_pretrain_pipeline and epoch > self._config['pseudo_pretrain_stop_epoch']:
                self._scheduler.step()
            else:
                self._scheduler.step()

            if not self._config['debug_mode']:
                self._save_checkpoint(epoch, 'model_last.pt')
                # self._save_checkpoint(epoch, 'student_model_last.pt')
            # fixed checkpoints at each 200 epochs:
            '''
            if (epoch % 500) == 0:
                self._save_checkpoint(epoch, f'model_epoch_{epoch}.pt')
            '''

    def _write_to_logger(self, num_epoch, category, **kwargs):
        for key, value in kwargs.items():
            name = category + '/' + key
            self._writer.add_scalar(name, value, num_epoch)

    def _save_checkpoint(self, num_epoch, name):
        # Delete prior best checkpoint
        if 'best' in name:
            [path.unlink() for path in self._path_to_run.iterdir() if 'best' in str(path)]

        if 'bst_student' in name:
            [path.unlink() for path in self._path_to_run.iterdir() if 'bst_student' in str(path)]

        torch.save({
            'epoch': num_epoch,
            'metric_max_val': self._main_metric_max_val,
            'model_state_dict': self._ensemble_model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._scheduler.state_dict(),
            'pseudo_cls_coef': self._pseudo_cls_coef,
        }, self._path_to_run / name)

def get_gpu_memory(device):
    #torch.cuda.empty_cache()
    memory_allocated = torch.cuda.memory_allocated(device)
    memory_cached = torch.cuda.memory_cached(device)
    return memory_allocated, memory_cached
