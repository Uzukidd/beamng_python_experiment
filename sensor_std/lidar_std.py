from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Lidar
from beamngpy.sensors.lidar import MAX_LIDAR_POINTS, LidarVisualiser

from threading import Thread
import time
import numpy as np

class lidar:
    def __init__(self, client, vehicle, callback, logger=None) -> None:
        self.stream_thread = Thread(target=self.update_stream, args=[])
        self.vehicle = vehicle
        self.callback = callback
        self.lidar = Lidar('lidar', client, vehicle, requested_update_time=0.01, is_using_shared_memory=True)
        self.logger = logger
        
        
    def start_stream(self) -> None:
        self.stream_thread.run()
    
    def update_stream(self) -> None:
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
            # R = np.dot(R, np.array([[1, 0, 0],
            #                         [0, np.cos(phi), -np.sin(phi)],
            #                         [0, np.sin(phi), np.cos(phi)]]))
            points = np.dot(points, R_z)
            points = np.dot(points, R_y)
            return points
        while True:
            pre_time = time.perf_counter()
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
            self.callback(points_np)
            aft_time = time.perf_counter()
            self.logger.info(f"FPS:{1.0/(aft_time - pre_time)}")
            # time.sleep(1.0/60)