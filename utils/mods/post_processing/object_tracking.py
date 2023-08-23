import torch
from pytorch3d.ops import knn_points, knn_gather

from typing import List, Dict

class object_tracker_base:
    def __init__(self) -> None:
        self.objects_dict = {} # object_dict_idx -> raw_data_idx
        
    def updates_object(self, objects:torch.Tensor) -> None:
        raise NotImplementedError
    
    
class mono_label_distance_tracker(object_tracker_base):
    def __init__(self) -> None:
        super().__init__()
        self.raw_box_anchors = None
        self.uu_id = 0
        self.previous_map = {} # raw_data_idx -> object_dict_idx
    
    def updates_object(self, box_anchors:torch.Tensor) -> None:
        """
            center = gt_boxes[0:3]
        """
        batch_dim = box_anchors.size(0)
        N_dim = box_anchors.size(1)
        print(f"uu_id:{self.uu_id}")
        if self.raw_box_anchors is not None:
            # print(self.raw_box_anchors, box_anchors)
            register_map = torch.ones((batch_dim, N_dim)).int() * -1
            knn = knn_points(self.raw_box_anchors, box_anchors)
            # knn.idx [b, N2, K]
            knn_neighbors = knn_gather(box_anchors, knn.idx) # [b, N2, K, 3]
            knn_dists = knn.dists # [b, N2, K]
            
            for b in range(batch_dim):
                for n in range(knn_neighbors.size(1)):
                    neigh_idx = knn.idx[b, n, 0]
                    if register_map[b, neigh_idx] == -1:
                        register_map[b, neigh_idx] = n
                    elif knn_dists[b, register_map[b, neigh_idx], 0] > knn_dists[b, neigh_idx, 0]:
                        register_map[b, neigh_idx] = n
            self.__updates_object_dict__(register_map)
        else:
            for b in range(batch_dim):
                for n in range(N_dim):
                    self.__register_object__(n)
                    
        self.raw_box_anchors = box_anchors
        
    # def linear_search(self, start) -> int:
    #     for i in range(start):
    #         pass
    
    def __register_object__(self, raw_data_idx:int):
        self.previous_map[raw_data_idx] = self.uu_id
        self.objects_dict[self.uu_id] = raw_data_idx
        self.uu_id = self.uu_id + 1
        
    def __recycle_objects__(self, updated_map:Dict):
        for key, val in updated_map.items():
            if not val:
                del self.previous_map[raw_data_idx] = self.uu_id
                del self.objects_dict[key]
    
    def __updates_object_dict__(self, register_map:torch.Tensor):
        batch_dim = register_map.size(0)
        N_dim = register_map.size(1)
        updated_map = {u_id:False for u_id in self.objects_dict.keys()}
        assert batch_dim == 1
        for b in range(batch_dim):
            for n in range(N_dim):
                if register_map[b, n] == -1:
                    self.__register_object__(n)
                else:
                    obj_u_id = self.previous_map[register_map[b, n].item()]
                    self.objects_dict[obj_u_id] = n
                    updated_map[obj_u_id] = True
                    
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