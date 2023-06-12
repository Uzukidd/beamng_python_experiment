import argparse
from pcdet.config import cfg, cfg_from_yaml_file

def parse_config(cfgs_path:str):
    cfg_from_yaml_file(cfgs_path, cfg)
    return cfg