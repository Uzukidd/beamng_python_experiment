import numpy as np
import torch

from pcdet.datasets import DatasetTemplate
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder

class carla_point_cloud_dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None, lidar=None) -> None:
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        
        self.lidar = lidar
        self.dataset_cfg = dataset_cfg
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )
        
    def point_cloud_input(self, points):
        input_dict = {
            'points': points.reshape(-1, 4),
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
        
        
    def __getitem__(self, index):
        points = self.lidar.get_single_frame()
        
        input_dict = {
            'points': None if points is None else points.reshape(-1, 4),
        }
        data_dict = input_dict
        if points is not None:
            data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
    
    def prepare_data(self, data_dict):
        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        return data_dict

class beamng_point_cloud_dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None, lidar=None, encoder=None) -> None:
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        
        self.lidar = lidar
        self.dataset_cfg = dataset_cfg
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )
        
    def point_cloud_input(self, points):
        input_dict = {
            'points': points.reshape(-1, 4),
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
        
        
    def __getitem__(self, index):
        points = self.lidar.get_single_frame()
        input_dict = {
            'points': torch.from_numpy(points.reshape(-1, 4)),
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
    
    def prepare_data(self, data_dict):
        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        return data_dict


class beamng_point_cloud_process_tools(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None, ext='.bin') -> None:
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.dataset_cfg = dataset_cfg
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )
        
    def point_cloud_input(self, points):
        input_dict = {
            'points': points.reshape(-1, 4),
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
        
    def prepare_data(self, data_dict):
        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        return data_dict
