from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Lidar
from beamngpy.sensors.lidar import MAX_LIDAR_POINTS, LidarVisualiser

from threading import Thread
import time

class lidar:
    def __init__(self, client, vehicle) -> None:
        self.stream_thread = Thread(target=self.update_stream, args=[])
        self.vehicle = vehicle
        self.lidar = Lidar('lidar', client, vehicle, requested_update_time=0.01, is_using_shared_memory=True)
        
        
    def start_stream(self) -> None:
        self.stream_thread.run()
    
    def update_stream(self) -> None:
        while True:
            self.vehicle.sensors.poll()
            points = self.lidar.poll()['pointCloud']
            print(points)
            time.sleep(0.0)