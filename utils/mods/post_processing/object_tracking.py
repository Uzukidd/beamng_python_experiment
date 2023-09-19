import torch
import numpy as np
from pytorch3d.ops import knn_points, knn_gather
from collections import deque

from typing import List, Dict

class object_tracker_base:
    def __init__(self) -> None:
        self.objects_dict = {} # object_dict_idx -> raw_data_idx
        
    def updates_object(self, box_anchors:torch.Tensor) -> None:
        raise NotImplementedError
    
    
class mono_label_distance_tracker(object_tracker_base):
    def __init__(self, track_length = 3, max_movement = 1.0) -> None:
        super().__init__()
        self.raw_box_anchors = None
        self.uu_id = 0
        self.previous_map = {} # raw_data_idx -> object_dict_idx
        self.interval_previous_map = {} # raw_data_idx -> object_dict_idx
        
        self.tracks = {}
        self.track_length = track_length
        
        self.max_movement = max_movement
        
    def get_all_object(self):
        all_uuid = list(self.objects_dict.keys())
        for uuid in all_uuid:
            yield uuid
            
    def get_raw_id(self, idx:int):
        return self.objects_dict[idx]
            
    def get_objects(self, idx:int):
        return self.raw_box_anchors[0, self.objects_dict[idx]]
    
    def get_track(self, idx:int):
        return self.tracks[idx]
    
    def get_all_bounding_box(self):
        all_uuid = list(self.objects_dict.keys())
        res = []
        for uuid in all_uuid:
            res.append(self.get_objects(uuid))
            
        if res.__len__() != 0:
            res = torch.stack(res)
        else:
            res = None
        return res
    
    def get_all_tracks(self):
        all_uuid = list(self.objects_dict.keys())
        res = []
        for uuid in all_uuid:
            res.append(self.get_track(uuid))
            
        return res
    
    def get_last_uuid(self):
        return self.uu_id
    
    def updates_object(self, box_anchors:torch.Tensor) -> None:
        """
            center = gt_boxes[0:3]
        """
        batch_dim = box_anchors.size(0)
        N_dim = box_anchors.size(1)
        if N_dim == 0:
            box_anchors = None
            updated_map = {u_id:False for u_id in self.objects_dict.keys()}
            self.__recycle_objects__(updated_map)
        elif self.raw_box_anchors is not None:
            register_map = torch.ones((batch_dim, N_dim)).int() * -1
            knn = knn_points(self.raw_box_anchors[:, :, :3], box_anchors[:, :, :3])
            # knn.idx [b, N2, K]
            # knn_neighbors = knn_gather(box_anchors[:, :, :3], knn.idx) # [b, N2, K, 3]
            knn_dists = knn.dists # [b, N2, K]
            
            for b in range(batch_dim):
                for n in range(knn.idx.size(1)):
                    neigh_idx = knn.idx[b, n, 0]
                    dist = knn_dists[b, n, 0]
                    
                    if dist > self.max_movement:
                        continue
    
                    if register_map[b, neigh_idx] == -1:
                        register_map[b, neigh_idx] = n
                    elif knn_dists[b, register_map[b, neigh_idx], 0] > dist:
                        register_map[b, neigh_idx] = n
                    
            self.__updates_object_dict__(register_map, box_anchors)
        else:
            for b in range(batch_dim):
                for n in range(N_dim):
                    self.__register_object__(n)
        
        self.__overwrite_previous_map__()
                    
        self.raw_box_anchors = box_anchors
    
    def __register_object__(self, raw_data_idx:int) -> int:
        obj_u_id = self.uu_id
        self.uu_id = self.uu_id + 1
        self.interval_previous_map[raw_data_idx] = obj_u_id
        self.objects_dict[obj_u_id] = raw_data_idx
        self.tracks[obj_u_id] = deque(maxlen = self.track_length)
        
        return obj_u_id
        
    def __recycle_objects__(self, updated_map:Dict) -> None:
        for key, val in updated_map.items():
            if not val:
                del self.objects_dict[key]
                del self.tracks[key]
                
    def __overwrite_previous_map__(self) -> None:
        self.previous_map = self.interval_previous_map
        self.interval_previous_map = {}
    
    def __updates_object_dict__(self, register_map:torch.Tensor, box_anchors:torch.Tensor) -> None:
        batch_dim = register_map.size(0)
        N_dim = register_map.size(1)
        updated_map = {u_id:False for u_id in self.objects_dict.keys()}
        assert batch_dim == 1
        for b in range(batch_dim):
            for n in range(N_dim):
                if register_map[b, n] == -1:
                    obj_u_id = self.__register_object__(n)
                else:
                    obj_u_id = self.previous_map[register_map[b, n].item()]
                    self.interval_previous_map[n] = obj_u_id
                    self.objects_dict[obj_u_id] = n
                    updated_map[obj_u_id] = True
                
                self.tracks[obj_u_id].append(box_anchors[b, n, :3])
                    
        self.__recycle_objects__(updated_map)
                    
        
    
class multi_classes_assemble_tracker(object_tracker_base):
    def __init__(self, num_classes, track_length = 3, max_movement = 1.0, multi_head = False, mono_tracker:mono_label_distance_tracker = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.mono_tracker = mono_tracker
        self.track_length = track_length
        self.max_movement = max_movement
        self.multi_head = multi_head
        
        self.raw_box_scores = None
        self.raw_box_labels = None
        
        if self.mono_tracker is None:
            self.build_mono_tracker()
        
    def build_mono_tracker(self) -> None:
        if self.multi_head:
            self.mono_tracker = []
            for idx in range(self.num_classes):
                track_length = self.track_length
                if isinstance(track_length, list):
                    track_length = track_length[idx]
                
                max_movement = self.max_movement
                if isinstance(max_movement, list):
                    max_movement = max_movement[idx]
                
                self.mono_tracker.append(mono_label_distance_tracker(track_length=track_length, max_movement=max_movement))
        else:
            self.mono_tracker = mono_label_distance_tracker(track_length=self.track_length, max_movement=self.max_movement)
            
            
    def get_all_object(self):
        if self.multi_head:
            bounding_boxes = []
            box_scores = []
            box_labels = []
            tracks = []
            for label in range(self.num_classes):
                for uuid in self.mono_tracker[label].get_all_object():
                    raw_data_id = self.mono_tracker[label].get_raw_id(uuid)
                    bounding_boxes.append(self.mono_tracker[label].get_objects(uuid))
                    box_scores.append(self.raw_box_scores[label][raw_data_id])
                    box_labels.append(self.raw_box_labels[label][raw_data_id])
                    tracks.append(self.mono_tracker[label].get_track(uuid))
                
            bounding_boxes = torch.stack(bounding_boxes) if bounding_boxes.__len__() != 0 else None
            box_scores = torch.stack(box_scores) if box_scores.__len__() != 0 else None
            box_labels = torch.stack(box_labels) if box_labels.__len__() != 0 else None
        else:
            bounding_boxes = []
            box_scores = []
            box_labels = []
            tracks = []
            for uuid in self.mono_tracker.get_all_object():
                raw_data_id = self.mono_tracker.get_raw_id(uuid)
                bounding_boxes.append(self.mono_tracker.get_objects(uuid))
                box_scores.append(self.raw_box_scores[0, raw_data_id])
                box_labels.append(self.raw_box_labels[0, raw_data_id])
                tracks.append(self.mono_tracker.get_track(uuid))
                
            bounding_boxes = torch.stack(bounding_boxes) if bounding_boxes.__len__() != 0 else None
            box_scores = torch.stack(box_scores) if box_scores.__len__() != 0 else None
            box_labels = torch.stack(box_labels) if box_labels.__len__() != 0 else None
        
        return bounding_boxes, box_scores, box_labels, tracks
    
    # def get_all_bounding_box(self):
    #     if self.multi_head:
    #         raise NotImplementedError
    #     else:
    #         return self.mono_tracker.get_all_bounding_box()

    # def get_all_tracks(self):
    #     if self.multi_head:
    #         raise NotImplementedError
    #     else:
    #         return self.mono_tracker.get_all_tracks()
    
    def get_last_uuid(self):
        if self.multi_head:
            res = []
            for label in range(self.num_classes):
                res.append(self.mono_tracker[label].get_last_uuid())
            return res
        else:
            return self.mono_tracker.get_last_uuid()
        
    
    def updates_object(self, box_anchors:torch.Tensor, box_labels:torch.Tensor, box_scores:torch.Tensor) -> None:
        """
            center = gt_boxes[0:3]
            lwh = gt_boxes[3:6]
            axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
        """
        if self.multi_head:
            self.raw_box_scores = []
            self.raw_box_labels = []
            for label in range(self.num_classes):
                label_idx = box_labels == label
                self.mono_tracker[label].updates_object(box_anchors[label_idx][np.newaxis, :])
                self.raw_box_scores.append(box_scores[label_idx])
                self.raw_box_labels.append(box_labels[label_idx])
                
        else:
            self.mono_tracker.updates_object(box_anchors)
            self.raw_box_scores = box_scores
            self.raw_box_labels = box_labels
        # raise NotImplementedError