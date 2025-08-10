"""Script for training the organdetr project."""

import argparse
import os,sys
import random
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="TypedStorage")
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("append to path & chdir:", base_dir)
os.chdir(base_dir)
sys.path.append(base_dir)
import numpy as np
import torch
import monai, re
from tqdm import tqdm
import itertools

import pickle

from organdetr.pixpro_cd_trainer import PixProTrainer
from organdetr.data.dataloader import get_loader
from organdetr.utils.io import get_config, write_json, get_meta_data
from organdetr.models.organdetrnet import OrgandetrNet
from organdetr.models.pixpro_organdetrnet import PixProOrgandetrNet
from organdetr.models.build import build_criterion
from organdetr.utils.io import write_json, load_json
from organdetr.evaluator import DetectionEvaluator
from organdetr.inference import inference

def get_last_ckpt(filepath):
    """
    # check the best one
    keyword = 'model_best_'
    for f in os.listdir(filepath):
        if re.match(keyword, f):
            ckpt_file = f"{filepath}/{f}"
            return ckpt_file

    # check the other ckpts if avail
    keyword = 'model_epoch_'

    steps = []
    for f in os.listdir(filepath):
        if re.match(keyword, f):
            stepid = int(f.split(keyword)[-1].split('.pt')[0])
            steps.append(stepid)

    if steps:
        ckpt_file = f"{filepath}/{keyword}{max(steps)}.pt"
        return ckpt_file
    """

    # check if last checkpoint avail
    keyword = 'model_last.pt'
    ckpt_file = f"{filepath}/{keyword}"
    return ckpt_file

def match(n, keywords):
    out = False
    for b in keywords:
        if b in n:
            out = True
            break
    return out

@torch.no_grad()
def get_replay_samples(config, device, source_train_loader):
    dataset_dir = Path(config['source_dataset_dir']).resolve()
    data_config = load_json(dataset_dir / "data_info.json")

    evaluator_replay = DetectionEvaluator(
        classes=list(data_config['labels'].values()),
        classes_small=data_config['labels_small'],
        classes_mid=data_config['labels_mid'],
        classes_large=data_config['labels_large'],
        iou_range_nndet=(0.1, 0.5, 0.05),
        iou_range_coco=(0.5, 0.95, 0.05),
        sparse_results=False
    )
    replay_scores = {}

    # Load old model from config["CL_models"]["old_model_path"]
    old_model_samples_rep = OrgandetrNet(config).to(device=device)
    checkpoint_old_model = torch.load(config["old_model_path"])

    old_model_samples_rep.load_state_dict(checkpoint_old_model['model_state_dict'])
    old_model_samples_rep.eval()
    for param in old_model_samples_rep.parameters():
        param.requires_grad = False

    for data, _, _, bboxes, _, _, _, path in tqdm(source_train_loader):

        data = data.to(device=device)

        targets = {
            'boxes': bboxes[0][0].to(dtype=torch.float, device=device),
            'labels': bboxes[0][1].to(device=device)
        }

        out = old_model_samples_rep(data)

        pred_boxes, pred_classes, pred_scores = inference(out)
        
        gt_boxes = [targets['boxes'].detach().cpu().numpy()]
        gt_classes = [targets['labels'].detach().cpu().numpy()] 

        # Add pred to evaluator
        result = evaluator_replay.replay_evaluator(
            pred_boxes=pred_boxes,
            pred_classes=pred_classes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes
        )
        metric_scores = evaluator_replay.eval_replay(result)
        replay_scores[path[0]] = metric_scores['mAP_coco']

    evaluator_replay.reset() # Reset evaluator

    # Sort the replay scores in ascending order
    replay_scores = sorted(replay_scores.items(), key=lambda item: item[1])
    top_k_scores = replay_scores[:config['top_k_source']]
    top_k_paths = [item[0] for item in top_k_scores]
    print("top_k_paths:")
    print(top_k_paths)

    with open("/dss/dsshome1/06/ge42vol2/OrganDETR/word_replay_samples.pkl", "wb") as f:
        pickle.dump(top_k_paths, f)

    
    with open("/dss/dsshome1/06/ge42vol2/OrganDETR/word_replay_samples.pkl", "rb") as f:
        top_k_paths = pickle.load(f)

    print("top_k_paths:")
    print(top_k_paths)
    # target_labeled_train_loader = get_loader(config, 'labeled_train', batch_size=config['target_labeled_batch_size'], root=config['target_dataset_dir'], data_list=top_k_paths)
    # return target_labeled_train_loader


def train(config, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device'][-1]
    device = 'cuda'
    # Build necessary components
    # labeled_train_loader = get_loader(config, 'train', batch_size=config['batch_size'])
    # unlabeled_train_loader = get_loader(config, 'unlabeled_train', batch_size=config['pixpro']['batch_size'])
    source_train_loader = get_loader(config, 'train', batch_size=1, root=config['source_dataset_dir'])
    get_replay_samples(config, device, source_train_loader) 

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add minimal amount of args (most args should be set in config files)
    parser.add_argument("--config", type=str, required=True, help="Config to use for training located in /config.")
    parser.add_argument("--resume", action='store_true', help="Auto-loads model_last.pt.")
    parser.add_argument("--medicalnet", action='store_true', help="Load pretrained ResNet")
    parser.add_argument("--pretrained_fpn", action='store_true', help="Load pretrained FPN")
    args = parser.parse_args()

    # Get relevant configs
    config = get_config(args.config)

    # To get reproducable results
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    monai.utils.set_determinism(seed=config['seed'])
    random.seed(config['seed'])

    torch.backends.cudnn.benchmark = False  # performance vs. reproducibility
    torch.backends.cudnn.deterministic = True

    train(config, args)
