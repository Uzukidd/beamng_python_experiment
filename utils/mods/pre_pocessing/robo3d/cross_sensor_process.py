from functools import partial
from collections import defaultdict

import numpy as np

def cross_sensor_process(data_dict=None, config=None):
    """
        This implementation is based on https://github.com/ldkong1205/Robo3D/tree/main
    """
    if data_dict is None:
        return partial(cross_sensor_process, config=config)
    
    def _get_kitti_ringID(scan, normal_vector=np.array([0, 0, 1]), threshold=0.00022):
        normal_vector = np.array([0, 0, 1])
        angle_radians = np.arccos(np.dot(scan[:, :3], normal_vector) / (np.linalg.norm(scan[:, :3], axis=1) * np.linalg.norm(normal_vector)))
        new_idx = np.argsort(angle_radians)
        
        angle_radians = angle_radians[new_idx]
        diff = angle_radians[1:] - angle_radians[:-1]
        new_raw = np.nonzero(diff >= threshold)[0] + 1
        
        proj_y = np.zeros_like(angle_radians)
        proj_y[new_raw] = 1
        ringID = np.cumsum(proj_y)
        ringID = np.clip(ringID, 0, 63)
        ringID[new_idx] = ringID
        return ringID, new_idx
    
    # get beam id
    scan = data_dict["points"]
    beam_id = _get_kitti_ringID(scan)
    beam_id = beam_id.astype(np.int64)

    if config.NUM_BEAM_TO_DROP == 16:
        to_drop = np.arange(1, 64, 4)
        assert len(to_drop) == 16
    
    elif config.NUM_BEAM_TO_DROP == 32:
        to_drop = np.arange(1, 64, 2)
        assert len(to_drop) == 32

    elif config.NUM_BEAM_TO_DROP == 48:
        to_drop = np.arange(1, 64, 1.33)
        to_drop = to_drop.astype(int)
        assert len(to_drop) == 48

    to_keep = [i for i in np.arange(0, 64, 1) if i not in to_drop]
    assert len(to_drop) + len(to_keep) == 64


    for id in to_drop:
        points_to_drop = beam_id == id
        scan = np.delete(scan, points_to_drop, axis=0)

        beam_id = np.delete(beam_id, points_to_drop, axis=0)

    scan = scan[::2, :]
    data_dict["points"] = scan

    return data_dict