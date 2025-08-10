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
def get_labeled_target_loader(config, device, source_train_loader):
    if 'replay_samples_file' in config:
        with open(config['replay_samples_file'], "rb") as f:
            top_k_paths = pickle.load(f)
    else:
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

        for data, _, bboxes, _, path in tqdm(source_train_loader):

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

    target_labeled_train_loader = get_loader(config, 'labeled_train', batch_size=config['target_labeled_batch_size'], root=config['target_dataset_dir'], replay_samples=top_k_paths, weak_aug=False, pixpro=False)
    return target_labeled_train_loader


def train(config, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device'][-1]
    device = 'cuda'
    # Build necessary components
    # labeled_train_loader = get_loader(config, 'train', batch_size=config['batch_size'])
    # unlabeled_train_loader = get_loader(config, 'unlabeled_train', batch_size=config['pixpro']['batch_size'])
    source_train_loader = get_loader(config, 'train', batch_size=1, root=config['source_dataset_dir'], replay_samples=None, weak_aug=False, pixpro=False)
    if config['replay']:
        labeled_train_loader = get_labeled_target_loader(config, device, source_train_loader) 
    else:
        labeled_train_loader = get_loader(config, 'labeled_train', batch_size=config['target_labeled_batch_size'], root=config['target_dataset_dir'], replay_samples=None, weak_aug=False, pixpro=False)

    pixpro_train_loader = get_loader(config, 'unlabeled_train', batch_size=config['pixpro']['batch_size'], root=config['target_dataset_dir'], replay_samples=None, weak_aug=False, pixpro=True)
    pseudo_train_loader = get_loader(config, 'unlabeled_train', batch_size=config['pseudo_batch_size'], root=config['target_dataset_dir'], replay_samples=None, weak_aug=True, pixpro=False)

    if config['overfit']:
        val_loader = get_loader(config, 'train', root=config['target_dataset_dir'])
    else:
        val_loader = get_loader(config, 'val', batch_size=6, root=config['target_dataset_dir'])

    # Set up pixpro config
    config['pixpro']['num_instances'] = len(pixpro_train_loader)
    config['pixpro']['start_epoch'] = 1
    config['pixpro']['epochs'] = config['epochs']

    model = PixProOrgandetrNet(config).to(device=device)

    if args.medicalnet:  # Download pretrained model from https://github.com/Tencent/MedicalNet
        assert config['backbone']['name'] == 'resnet', 'Loading MedicalNet is only possible if ResNet backbone is configured!'
        ckpt = torch.load('resnet_50.pth')
        fixed_ckpt = {}
        fixed_ckpt = {k.replace('module.',''): v for k, v in ckpt['state_dict'].items()}# checkpoint contains additional "module." prefix
        try:
            model._backbone.load_state_dict(fixed_ckpt, strict=True)
        except Exception as e:
            print("These layers are not loaded from the pretrained checkpoint: ", e)
            print("Loading pretrained model with strict=False ...")
            model._backbone.load_state_dict(fixed_ckpt, strict=False)
    elif args.pretrained_fpn:
        ckpt = torch.load('p224_bs3final_model.pth')
        fixed_ckpt = {}
        fixed_ckpt = {k.replace('attnFPN.',''): v for k, v in ckpt.items()}
        try:
            model._backbone.load_state_dict(fixed_ckpt, strict=True)
        except Exception as e:
            print("These layers are not loaded from the pretrained checkpoint: ", e)
            print("Loading pretrained model with strict=False ...")
            model._backbone.load_state_dict(fixed_ckpt, strict=False)

    if config.get('hybrid_dense_matching', False):
        criterion, dense_hybrid_criterion = build_criterion(config)
        criterion = criterion.to(device=device)
        dense_hybrid_criterion = dense_hybrid_criterion.to(device=device)
    else:
        criterion = build_criterion(config).to(device=device)
        dense_hybrid_criterion = None

    # Analysis of model parameter distribution
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_backbone_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and match(n, ['backbone', 'input_proj', 'skip']))
    num_neck_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and match(n, ['neck', 'query']))
    num_head_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and match(n, ['head']))

    param_dicts = [
        {
            'params': [
                p for n, p in model.named_parameters() if not match(n, ['backbone', 'reference_points', 'sampling_offsets']) and p.requires_grad
            ],
            'lr': float(config['lr'])
        },
        {
            'params': [p for n, p in model.named_parameters() if match(n, ['backbone']) and p.requires_grad],
            'lr': float(config['lr_backbone'])
        } 
    ]

    # Append additional param dict for def detr
    if sum([match(n, ['reference_points', 'sampling_offsets']) for n, _ in model.named_parameters()]) > 0:
        param_dicts.append(
            {
                "params": [
                    p for n, p in model.named_parameters() if match(n, ['reference_points', 'sampling_offsets']) and p.requires_grad
                ],
                'lr': float(config['lr']) * config['lr_linear_proj_mult']
            }
        )

    '''
    print("===============================================================")
    print("model:")
    print(model)
    print("===============================================================")
    '''

    optim = torch.optim.AdamW(
        param_dicts, lr=float(config['lr_backbone']), weight_decay=float(config['weight_decay'])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optim, config['lr_drop'])


    # Init logging
    # path_to_run = Path(os.getcwd()) / 'runs' / config['experiment_name']
    path_to_run = Path(config['log_path']) / 'runs' / config['experiment_name']
    path_to_run.mkdir(exist_ok=True)


    # Load checkpoint if applicable
    if config.get('resume', False) or args.resume:
        ckpt_file = get_last_ckpt(path_to_run)
        print(f'[+] loading ckpt {ckpt_file} ...')
        checkpoint = torch.load(Path(ckpt_file))

        checkpoint['scheduler_state_dict']['step_size'] = config['lr_drop']

        # Unpack and load content
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        metric_start_val = checkpoint['metric_max_val']
    else:
        epoch = 0
        metric_start_val = 0
        if config['use_pre_trained_model']:
            # Load pre-trained model
            print(f"===================old model={config['old_model_path']}")
            old_model_checkpoint = torch.load(config['old_model_path'])
            old_model_state_dict = old_model_checkpoint['model_state_dict']
            model_state_dict = model.state_dict()
            new_model_state_dict = {}
            for key in model_state_dict:
                if key in old_model_state_dict:
                    new_model_state_dict[key] = old_model_state_dict[key]
                else:
                    new_model_state_dict[key] = model_state_dict[key]

            # pixpro encoder
            if config['pixpro']['use_pixpro']:
                for param_q, param_k, param_backbone in zip(model._pixpro.encoder.parameters(), model._pixpro.encoder_k.parameters(), model._backbone.parameters()):
                    param_q.data.copy_(param_backbone.data)
                    param_k.data.copy_(param_backbone.data)
                    # stop gradient
                    param_k.requires_grad = False


    # log num_params
    num_params_dict ={'num_params': num_params,
                      'num_backbone_params': num_backbone_params,
                      'num_neck_params': num_neck_params,
                      'num_head_params': num_head_params
                      }
    config.update(num_params_dict)
    # Get meta data and write config to run
    try:
        config.update(get_meta_data())
    except:
        pass

    write_json(config, path_to_run / 'config.json')

    # Build trainer and start training
    trainer = PixProTrainer(
        labeled_train_loader, pixpro_train_loader, pseudo_train_loader, val_loader, model, criterion, optim, scheduler, device, config, 
        path_to_run, epoch, metric_start_val, dense_hybrid_criterion
    )
    trainer.run()
        

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
