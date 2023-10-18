from functools import partial
from collections import defaultdict

import os
import copy
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torch_data

tv = None
try:
    import cumm.tensorview as tv
except:
    pass

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points

class point_cloud_dataset_base(torch_data.Dataset):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None, lidar=None) -> None:
        super().__init__()
        self.lidar = lidar
        self.idx = 0
        self.class_names = class_names
        self.dataset_cfg = dataset_cfg
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.training = training
        self.num_point_features = len(self.dataset_cfg.POINT_FEATURE_ENCODING.used_feature_list)
        self.root_path = root_path
        self.mode = 'train' if training else 'test'
        
        self.preview_channel = self.dataset_cfg.PREVIEW_CHANNEL
        self.data_processor_queue = {}
        
        self.grid_size = self.voxel_size = None
        self.voxel_generator = None
        
        self.logger = logger
        
        self.init_data_processor()
        
    def init_data_processor(self):
        if self.dataset_cfg.DATA_PROCESSOR is not None:
            for channel_name, channel in self.dataset_cfg.DATA_PROCESSOR.items():
                self.data_processor_queue[channel_name] = []
                for process in channel:
                    if process["NAME"] is None:
                        continue
                    
                    module_call = self
                    method_name = process["NAME"]
                    if process.get('EXT_MODULE', False):
                        module_name, method_name = process["NAME"].rsplit('.', 1)
                        module_call = importlib.import_module(module_name)
                    
                    process_method = getattr(module_call, method_name)

                    if process_method is not None and callable(process_method):
                        self.data_processor_queue[channel_name].append(process_method(config = process))
                    else:
                        raise NotImplementedError
    
    
    def sample_cache(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_cache, config=config)
        
        file_path = config.get("FILENAME", None)
        frequency = config.get("SAMPLE_FREQUENCY", 50)
        points = data_dict.get("points", None)
        gt = data_dict.get("gt_boxes", None)
        
        if self.idx % frequency == 0:
            filename = f"{int(self.idx / frequency):05d}"
            if points is not None \
                and gt is not None \
                and file_path is not None:
                    np.save(os.path.join(file_path, "points", filename), points)
                    np.save(os.path.join(file_path, "gt", filename), gt)
        
        return data_dict
                
    def raw_data_remain(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.raw_data_remain, config=config)
        
        data_dict["raw_points"] = np.copy(data_dict["points"])
        
        return data_dict
        
    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        if config.get('DOUBLE_FLIP', False):
            voxels_list, voxel_coords_list, voxel_num_points_list = [voxels], [coordinates], [num_points]
            points_yflip, points_xflip, points_xyflip = self.double_flip(points)
            points_list = [points_yflip, points_xflip, points_xyflip]
            keys = ['yflip', 'xflip', 'xyflip']
            for i, key in enumerate(keys):
                voxel_output = self.voxel_generator.generate(points_list[i])
                voxels, coordinates, num_points = voxel_output

                if not data_dict['use_lead_xyz']:
                    voxels = voxels[..., 3:]
                voxels_list.append(voxels)
                voxel_coords_list.append(coordinates)
                voxel_num_points_list.append(num_points)

            data_dict['voxels'] = voxels_list
            data_dict['voxel_coords'] = voxel_coords_list
            data_dict['voxel_num_points'] = voxel_num_points_list
        else:
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
        return data_dict
    
    def mask_points_by_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_by_range, config=config)
        points = data_dict["points"]
        gt = data_dict.get("gt_boxes", None)
        limit_range = config["POINT_CLOUD_RANGE"]
        
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
            & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
        
        data_dict['points'] = data_dict['points'][mask]
        data_dict['intensity'] = data_dict['intensity'][mask]
        data_dict['use_lead_xyz'] = True
        
        if gt is not None:
            mask = (gt[:, 0] >= limit_range[0]) & (gt[:, 0] <= limit_range[3]) \
            & (gt[:, 1] >= limit_range[1]) & (gt[:, 1] <= limit_range[4])
            data_dict["gt_boxes"] = data_dict["gt_boxes"][mask]
        
        return data_dict
    
    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        batch_size_ratio = 1

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    if isinstance(val[0], list):
                        batch_size_ratio = len(val[0])
                        val = [i for item in val for i in item]
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['raw_points', 'points', 'voxel_coords']:
                    coors = []
                    if isinstance(val[0], list):
                        val =  [i for item in val for i in item]
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_boxes']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k,:, :val[k].shape[1], :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_scores', 'roi_labels']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k,:, :val[k].shape[1]] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    raise NotImplementedError
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        pad_value = 0

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                elif key in ['calib']:
                    ret[key] = val
                elif key in ["points_2d"]:
                    max_len = max([len(_val) for _val in val])
                    pad_value = 0
                    points = []
                    for _points in val:
                        pad_width = ((0, max_len-len(_points)), (0,0))
                        points_pad = np.pad(_points,
                                pad_width=pad_width,
                                mode='constant',
                                constant_values=pad_value)
                        points.append(points_pad)
                    ret[key] = np.stack(points, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size * batch_size_ratio
        return ret
    
    def prepare_data(self, data_dict, channel_name):
        """
            \"points\": Tensor[N, 4] : [N, (x, y, z, r)]
        """
        
        # if data_dict.get('points', None) is not None:
        #     mask = self.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        #     data_dict['points'] = data_dict['points'][mask]
        #     data_dict['intensity'] = data_dict['intensity'][mask]
        #     data_dict['use_lead_xyz'] = True
        
        for process in self.data_processor_queue[channel_name]:
            data_dict = process(data_dict)
        
        return data_dict
    
    def __getitem__(self, index, points=None, gt=None):
        if points is None:
            self.idx = self.idx + 1
            points, gt = self.lidar.get_single_frame()
        
        input_dict = {
            'points': None,
            'intensity': None,
        }
        
        if gt is not None:
            input_dict["gt_boxes"] = gt
        
        if points is not None:
            points = points.reshape(-1, 4)
            input_dict['intensity'] = points[:, 3].copy()
            points[:, 3] = 0
            input_dict['points'] = points
        
        data_dict = {}
        
        import time
        
        if points is not None:
            for channel_name in self.data_processor_queue:
                before_time = time.perf_counter()
                
                temp_dict = input_dict
                if self.data_processor_queue.__len__() > 1:
                    temp_dict = copy.deepcopy(temp_dict)
                
                data_dict[channel_name] = self.prepare_data(data_dict=temp_dict, channel_name=channel_name)
                after_time = time.perf_counter()
                data_dict[channel_name]["pre_time"] = after_time - before_time
            
        del input_dict
        # data_dict["pre_time"] = after_time - before_time

        return data_dict

class carla_point_cloud_dataset(point_cloud_dataset_base):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None, lidar=None) -> None:
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger, lidar=lidar
        )
        
    def __getitem__(self, index):
        return super().__getitem__(index)

class beamng_point_cloud_dataset(point_cloud_dataset_base):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None, lidar=None) -> None:
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger, lidar=lidar
        )
        
    def __getitem__(self, index):
        return super().__getitem__(index)
    
class file_point_cloud_dataset(point_cloud_dataset_base):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None) -> None:
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger, lidar=None
        )
        self.root_path = root_path
        self.filenames = os.listdir(os.path.join(self.root_path, "points"))
        print(self.filenames)
        
        self.points = []
        self.gt = []
        
        for name in self.filenames:
            self.points.append(np.load(os.path.join(self.root_path, "points", name)))
            try:
                self.gt.append(np.load(os.path.join(self.root_path, "gt", name)))
            except:
                self.gt.append(None)
        
        
    def __getitem__(self, index):
        return super().__getitem__(index, self.points[index], self.gt[index])
    


