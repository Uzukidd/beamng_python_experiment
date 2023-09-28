from functools import partial
from collections import defaultdict

import numpy as np

def motion_blur(data_dict=None, config=None):
    """
        This implementation is based on https://github.com/ldkong1205/Robo3D/tree/main
    """
    if data_dict is None:
        return partial(motion_blur, config=config)
    
    points = data_dict["points"]
    assert points is not None

    noise_translate = np.array([
        np.random.normal(0, config.TRANS_STD[0], 1),
        np.random.normal(0, config.TRANS_STD[1], 1),
        np.random.normal(0, config.TRANS_STD[2], 1),
        ]).T

    points[:, 0:3] += noise_translate
    num_points = points.shape[0]
    jitters_x = np.clip(np.random.normal(loc=0.0, scale=config.TRANS_STD[0]*0.1, size=num_points), -3 * config.TRANS_STD[0], 3 * config.TRANS_STD[0])
    jitters_y = np.clip(np.random.normal(loc=0.0, scale=config.TRANS_STD[1]*0.1, size=num_points), -3 * config.TRANS_STD[1], 3 * config.TRANS_STD[1])
    jitters_z = np.clip(np.random.normal(loc=0.0, scale=config.TRANS_STD[2]*0.05, size=num_points), -3 * config.TRANS_STD[2], 3 * config.TRANS_STD[2])

    points[:, 0] += jitters_x
    points[:, 1] += jitters_y
    points[:, 2] += jitters_z
    
    data_dict["points"] = points
    
    return data_dict