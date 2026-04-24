import os

import yaml

from utils import generate_pad, torch, util


def get_log_dir_index(out_dir):
    dirs = [x[0] for x in os.listdir(out_dir)]
    if '.' in dirs:  # minor change for .ipynb
        dirs.remove('.')
    log_dir_index = '_' + str(len(dirs))

    return log_dir_index


def update_config(cfg, args_dict):
    """
    update some configuration related to args
        - merge args to cfg
        - dct, idct matrix
        - save path dir
    """
    for k, v in args_dict.items():
        setattr(cfg, k, v)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    cfg.dtype = dtype

    cfg.dct_m, cfg.idct_m = util.get_dct_matrix(cfg.t_pred + cfg.t_his)
    cfg.dct_m_all = cfg.dct_m.float().to(cfg.device)
    cfg.idct_m_all = cfg.idct_m.float().to(cfg.device)

    index = get_log_dir_index(cfg.base_dir)
    cfg.cfg_dir = '%s/%s' % (cfg.base_dir, args_dict['mode'] + index)
    os.makedirs(cfg.cfg_dir, exist_ok=True)
    cfg.result_dir = '%s/results' % cfg.cfg_dir
    cfg.log_dir = '%s/log' % cfg.cfg_dir
    cfg.gif_dir = '%s/out' % cfg.cfg_dir
    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.gif_dir, exist_ok=True)

    return cfg


class Config:

    def __init__(self, cfg_id, test=True):
        self.id = cfg_id
        cfg_name = './cfg/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        # base dir for inference artifacts
        self.base_dir = 'inference'
        os.makedirs(self.base_dir, exist_ok=True)

        # common
        self.dataset = cfg.get('dataset', 'h36m')
        self.t_his = cfg['t_his']
        self.t_pred = cfg['t_pred']

        self.n_pre = cfg['n_pre']
        self.multimodal_path = cfg['multimodal_path']
        self.data_candi_path = cfg['data_candi_path']

        self.padding = cfg['padding']
        self.Complete = cfg['Complete']
        self.noise_steps = cfg['noise_steps']
        self.ddim_timesteps = cfg['ddim_timesteps']
        self.scheduler = cfg['scheduler']

        self.dropout = cfg['dropout']
        self.num_heads = cfg['num_heads']

        self.mod_test = cfg['mod_test']

        # indirect variable
        if self.dataset == 'h36m':
            self.joint_num = 16
        elif self.dataset == 'amass':
            self.joint_num = 21
        else:
            self.joint_num = 14
        self.idx_pad, self.zero_index = generate_pad(self.padding, self.t_his, self.t_pred)
