import numpy as np
import torch
import logging

from .dataset_tools import carla_point_cloud_dataset, beamng_point_cloud_dataset, file_point_cloud_dataset
from .config_init import cfg, cfg_from_yaml_file
# from .visualize_utils import draw_scenes_maya, mayavi_animate_visualizer
from .open3d_vis_utils import draw_scenes
try :
    from .cpp_ext import iou3d_nms_cuda
except :
    from pcdet.ops.iou3d_nms import iou3d_nms_cuda
    print("try to use pcdet.ops.iou3d_nms.iou3d_nms_cuda")
# from .mods import post_processing

def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    # keep = torch.IntTensor(boxes.size(0))
    keep = torch.zeros(boxes.size(0)).int()

    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)

    return order[keep[:num_out].long().cuda()].contiguous(), None

def class_agnostic_nms(box_scores, box_preds, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(4096, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = nms_gpu(
                boxes_for_nms[:, 0:7], box_scores_nms, 0.01, 4096
        )
        selected = indices[keep_idx[:500]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            batch_dict[key] = torch.from_numpy(np.array([val])).cuda()
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            batch_dict[key] = torch.from_numpy(val).cuda()
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()
            
            
def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger