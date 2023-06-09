
try:
    from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
    from beamngpy.sensors import Lidar
    from beamngpy.sensors.lidar import MAX_LIDAR_POINTS, LidarVisualiser
except:
    pass

try:
    import carla
except:
    pass

from threading import Thread
import time
import numpy as np
from queue import Queue

class lidar:
    def __init__(self, client, vehicle, lidar_para={}, callback=None, logger=None) -> None:
        self.stream_thread = Thread(target=self.update_stream, args=[])
        self.vehicle = vehicle
        self.callback = callback
        self.lidar = Lidar('lidar', client, vehicle, **lidar_para)
        self.logger = logger

    def get_single_frame(self) -> np.array:
        def rotate(points, n):
            # Step 1
            n = n / np.linalg.norm(n)
            # Step 2
            theta = np.arctan2(n[1], n[0])
            phi = -np.arctan2(n[2], np.sqrt(n[0]**2 + n[1]**2))
            # print(phi)
            # Step 3
            R_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
            
            R_y = np.array([[np.cos(phi), 0, np.sin(phi)],
                        [0, 1, 0],
                        [-np.sin(phi), 0, np.cos(phi)]])
            points = np.dot(points, R_z)
            points = np.dot(points, R_y)
            return points
        
        self.vehicle.sensors.poll()
        points = self.lidar.poll()['pointCloud']
        dir_lidar = np.array(self.lidar.get_direction())
        pos_lidar = np.array(self.lidar.get_position())

        points_np = np.array(points).reshape(-1, 3)
        points_np = points_np[points_np != [0, 0, 0]].reshape(-1, 3)
        points_np = points_np - pos_lidar[np.newaxis, :]
        points_np = rotate(points_np, dir_lidar)
        count = points_np.shape[0]
        _ = np.zeros(shape = (count,4))
        _[:, 0:3] = points_np
        points_np = _
        
        return points_np
        
    def start_stream(self) -> None:
        self.stream_thread.run()
    
    def update_stream(self) -> None:
        while True:
            pre_time = time.perf_counter()
            points_np = self.get_single_frame()
            self.callback(points_np)
            aft_time = time.perf_counter()
            # self.logger.info(f"FPS:{1.0/(aft_time - pre_time)}")
            
class lidar_carla:
    def __init__(self, carla_world, vehicle, lidar_para={
            "upper_fov":"15.0",
            "lower_fov":"-25.0",
            "channels":"64.0",
            "range":"100.0",
            "points_per_second":"6600000",
            "semantic":False,
            "no_noise":False,
            "delta":0.05,
            "rotation_frequency":"20"
        }, logger=None) -> None:
        self.pcs_frames = Queue(1)
        self.vehicle = vehicle
        self.carla_world = carla_world
        self.logger = logger
        self.lidar_para = lidar_para
        self.lidar = None
        
    def init_lidar(self):
        lidar_bp = self.generate_lidar_bp(self.lidar_para, self.carla_world)
        
        user_offset = carla.Location(0.0, 0.0, 0.0)
        lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8) + user_offset)

        self.lidar = self.carla_world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.lidar.listen(lambda point_cloud: self._pcs_callback(point_cloud))
        
    def generate_lidar_bp(self, arg, world):
        """Generates a CARLA blueprint based on the script parameters"""
        if arg["semantic"]:
            lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        else:
            lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
            if arg["no_noise"]:
                lidar_bp.set_attribute('dropoff_general_rate', '0.0')
                lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
                lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
            else:
                lidar_bp.set_attribute('noise_stddev', '0.2')

        lidar_bp.set_attribute('upper_fov', arg["upper_fov"])
        lidar_bp.set_attribute('lower_fov', str(arg["lower_fov"]))
        lidar_bp.set_attribute('channels', str(arg["channels"]))
        lidar_bp.set_attribute('range', str(arg["range"]))
        lidar_bp.set_attribute('rotation_frequency', str(arg["rotation_frequency"]))
        lidar_bp.set_attribute('points_per_second', str(arg["points_per_second"]))
        return lidar_bp

    def get_single_frame(self) -> np.array:
        res = None
        try:
            res = self.pcs_frames.get(True, 1.0)
        except:
            pass
        return res
        
    def _pcs_callback(self, point_cloud) -> None:
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = data.reshape((-1, 4))
        data[:,1] = -data[:,1] 
        self.pcs_frames.put(data)
        