import carla
import time
import numpy as np
from threading import Thread
import random
import open3d as o3d

from sensor_std import lidar_carla


class carla_client:
    def __init__(self, host="127.0.0.1", port=2000, rendering=True, logger=None) -> None:
        self.carla_client = None
        self.carla_world = None
        self.host = host
        self.port = port
        self.rendering = rendering
        self.logger = logger

        self.lidar_t = None
        # self.tick_thread = Thread(target=self.world_tick, args=[])

    def init_client(self, timeout=10.0) -> None:
        self.carla_client = carla.Client(self.host, self.port)
        self.carla_client.set_timeout(timeout)

    def start_client(self) -> None:
        self.carla_world = self.carla_client.load_world('Town03')

    def debug_luanch_test(self) -> None:
        original_settings = self.carla_world.get_settings()
        settings = self.carla_world.get_settings()
        traffic_manager = self.carla_client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        delta = 0.05

        settings.fixed_delta_seconds = delta
        settings.synchronous_mode = True
        settings.no_rendering_mode = not self.rendering
        self.carla_world.apply_settings(settings)

        blueprint_library = self.carla_world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("model3")[0]
        vehicle_transform = self.carla_world.get_map().get_spawn_points()
        vehicle_transform_z = np.array(
            [tr.location.z for tr in vehicle_transform])
        hignest_point_idx = np.argmax(vehicle_transform_z)
        # vehicle_transform = vehicle_transform[]
        # pos_idx = np.random.randint(0, vehicle_transform.__len__())
        print(
            f"The ego vehicle has been spawned at position \t{hignest_point_idx}/\t{vehicle_transform.__len__()}")
        vehicle_transform = vehicle_transform[hignest_point_idx]

        self.vehicle = self.carla_world.spawn_actor(
            vehicle_bp, vehicle_transform)
        self.vehicle.set_autopilot(True)

        self.lidar_t = lidar_carla(
            self.carla_world, self.vehicle, pcs_cache=False, need_gt=True)
        self.lidar_t.init_lidar()

    def generate_lidar_bp(self, arg, world, blueprint_library, delta):
        """Generates a CARLA blueprint based on the script parameters"""
        if arg["semantic"]:
            lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        else:
            lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
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
        lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
        lidar_bp.set_attribute('points_per_second',
                               str(arg["points_per_second"]))
        return lidar_bp

    def close_client(self) -> None:
        self.carla_client
