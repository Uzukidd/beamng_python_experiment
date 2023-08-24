import torch
from pytorch3d.ops import knn_points, knn_gather
from collections import deque

from typing import List, Dict

class object_tracker_base:
    def __init__(self) -> None:
        self.objects_dict = {} # object_dict_idx -> raw_data_idx
        
    def updates_object(self, objects:torch.Tensor) -> None:
        raise NotImplementedError
    
    
class mono_label_distance_tracker(object_tracker_base):
    def __init__(self, track_length = 3) -> None:
        super().__init__()
        self.raw_box_anchors = None
        self.uu_id = 0
        self.previous_map = {} # raw_data_idx -> object_dict_idx
        self.interval_previous_map = {} # raw_data_idx -> object_dict_idx
        
        self.tracks = {}
        self.track_length = track_length
        
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
            self.__overwrite_previous_map__()
        elif self.raw_box_anchors is not None:
            # print(self.raw_box_anchors, box_anchors)
            
            
            register_map = torch.ones((batch_dim, N_dim)).int() * -1
            knn = knn_points(self.raw_box_anchors[:, :, :3], box_anchors[:, :, :3])
            # knn.idx [b, N2, K]
            # knn_neighbors = knn_gather(box_anchors[:, :, :3], knn.idx) # [b, N2, K, 3]
            knn_dists = knn.dists # [b, N2, K]
            
            for b in range(batch_dim):
                for n in range(knn.idx.size(1)):
                    neigh_idx = knn.idx[b, n, 0]
                    if register_map[b, neigh_idx] == -1:
                        register_map[b, neigh_idx] = n
                    else:
                        p1 = knn_dists[b, register_map[b, neigh_idx], 0]
                        p2 = knn_dists[b, n, 0]
                        if p1 > p2:
                            register_map[b, neigh_idx] = n
            self.__updates_object_dict__(register_map, box_anchors)
            self.__overwrite_previous_map__()
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
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.mono_tracker = []
        
    def build_mono_tracker(self) -> None:
        for _ in range(self.num_classes):
            self.mono_tracker.append(mono_label_distance_tracker())
            
    
    def updates_object(self, objects_list:List[torch.Tensor]) -> None:
        """
            center = gt_boxes[0:3]
            lwh = gt_boxes[3:6]
            axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
        """
        raise NotImplementedError