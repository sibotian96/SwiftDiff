import argparse
import os
import sys

import torch

from config import Config, update_config
from data_loader.dataset_amass import DatasetAMASS
from utils import create_logger, seed_set
from utils.demo_visualize import demo_visualize
from utils.evaluation import compute_stats
from utils.script import (create_model, create_solver, dataset_split,
                          display_exp_setting, get_multimodal_gt_full)

sys.path.append(os.getcwd())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', default='h36m', help='h36m or amass')
    parser.add_argument('--mode', default='eval', choices=['eval', 'pred'],
                        help='eval: compute metrics; pred: render visualization gifs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--ckpt', type=str, default='./checkpoints/swiftdiff_h36m.pt')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='number of MLP mixer blocks in the SwiftDiff student model')
    parser.add_argument('--latent_dims', type=int, default=768,
                        help='latent/channel dimension of the SwiftDiff student model')
    parser.add_argument('--vis_col', type=int, default=10)
    parser.add_argument('--vis_row', type=int, default=3)
    args = parser.parse_args()

    """setup"""
    seed_set(args.seed)

    cfg = Config(f'{args.cfg}', test=True)
    cfg = update_config(cfg, vars(args))

    if cfg.dataset == 'amass':
        dataset = {'test': DatasetAMASS('test')}
        dataset_multi_test = None
    else:
        dataset, dataset_multi_test = dataset_split(cfg)

    """logger"""
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)

    """model"""
    logger.info(f'layer number: {cfg.num_layers}')
    logger.info(f'latent dims: {cfg.latent_dims}')

    logger.info(">>> SwiftDiff model loading...")
    model = create_model(cfg)
    logger.info(">>> Total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    logger.info(">>> DDIM solver creating...")
    solver = create_solver(cfg)

    ckpt = torch.load(args.ckpt, map_location=cfg.device)
    model.load_state_dict(ckpt)
    model.eval()

    if args.mode == 'eval':
        if cfg.dataset == 'amass':
            multimodal_dict = get_multimodal_gt_full(logger, dataset['test'], args, cfg)
        else:
            multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        compute_stats(solver, multimodal_dict, model, logger, cfg)

    else:  # 'pred'
        demo_visualize(args.mode, cfg, model, solver, dataset)
