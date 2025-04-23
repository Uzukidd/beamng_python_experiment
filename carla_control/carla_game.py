import carla
import time
import numpy as np
from threading import Thread
import random
import open3d as o3d

from sensor_std import lidar_carla


class carla_client:
    def __init__(
        self, host="127.0.0.1", port=2000, delta=0.05, rendering=True, logger=None
    ) -> None:
        self.carla_client = None
        self.carla_world = None
        self.host = host
        self.port = port
        self.delta = delta
        self.rendering = rendering
        self.logger = logger

        self.lidar_t = None
        # self.tick_thread = Thread(target=self.world_tick, args=[])

    def init_client(self, timeout=10.0) -> None:
        self.carla_client = carla.Client(self.host, self.port)
        self.carla_client.set_timeout(timeout)

    def start_client(self) -> None:
        # self.carla_world = self.carla_client.load_world('Town10HD_Opt')
        self.carla_world = self.carla_client.get_world()

    def debug_luanch_test_1(
        self,
    ) -> None:
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
        # vehicle_transform_z = np.array(
        #     [tr.location.z for tr in vehicle_transform])
        # hignest_point_idx = np.argmax(vehicle_transform_z)
        vehicle_transform = vehicle_transform
        pos_idx = np.random.randint(0, vehicle_transform.__len__())
        print(
            f"The ego vehicle has been spawned at position \t{pos_idx}/\t{vehicle_transform.__len__()}"
        )
        vehicle_transform = vehicle_transform[pos_idx]

        self.vehicle = self.carla_world.spawn_actor(vehicle_bp, vehicle_transform)
        self.vehicle.set_autopilot(True)

        self.lidar_t = lidar_carla(
            self.carla_world, self.vehicle, pcs_cache=False, need_gt=True
        )
        self.lidar_t.init_lidar()

    def replay_file(self, recorder_filename):
        self.carla_client.replay_file(recorder_filename, 0.0, 0.0, 0, False)

    def connect_to_vehicle(self, rolename, noisy_lidar=True):
        self.vehicle = None
        while self.vehicle is None:
            possible_vehicles = self.carla_world.get_actors().filter("vehicle.*")
            for vehicle in possible_vehicles:
                if vehicle.attributes["role_name"] == rolename:
                    print("Ego vehicle found")
                    self.vehicle = vehicle
                    break

        self.lidar_t = lidar_carla(
            self.carla_world,
            self.vehicle,
            {
                "upper_fov": "2.0",
                "lower_fov": "-24.8",
                "channels": "64.0",
                "range": "120.0",
                "points_per_second": "3300000",
                "semantic": False,
                "no_noise": not noisy_lidar,
                "delta": self.delta,
                # "rotation_frequency":"20"
            },
            pcs_cache=False,
            need_gt=True,
        )
        self.lidar_t.init_lidar()

    def synchronize_client(self):
        settings = self.carla_world.get_settings()
        traffic_manager = self.carla_client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        settings.fixed_delta_seconds = self.delta
        settings.synchronous_mode = True
        settings.no_rendering_mode = not self.rendering
        self.carla_world.apply_settings(settings)

    def debug_luanch_test_2(self, rolename) -> None:
        settings = self.carla_world.get_settings()
        # traffic_manager = self.carla_client.get_trafficmanager(8000)
        # traffic_manager.set_synchronous_mode(True)

        delta = 0.05

        settings.fixed_delta_seconds = delta
        settings.synchronous_mode = False
        settings.no_rendering_mode = not self.rendering
        self.carla_world.apply_settings(settings)

        self.vehicle = None
        while self.vehicle is None:
            possible_vehicles = self.carla_world.get_actors().filter("vehicle.*")
            for vehicle in possible_vehicles:
                if vehicle.attributes["role_name"] == rolename:
                    print("Ego vehicle found")
                    self.vehicle = vehicle
                    break

        if self.vehicle is None:
            # traffic_manager.set_synchronous_mode(False)
            import sys

            sys.exit(0)

        self.lidar_t = lidar_carla(
            self.carla_world, self.vehicle, pcs_cache=False, need_gt=True
        )
        self.lidar_t.init_lidar()

    def generate_lidar_bp(self, arg, world, blueprint_library, delta):
        """Generates a CARLA blueprint based on the script parameters"""
        if arg["semantic"]:
            lidar_bp = world.get_blueprint_library().find(
                "sensor.lidar.ray_cast_semantic"
            )
        else:
            lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
            if arg["no_noise"]:
                lidar_bp.set_attribute("dropoff_general_rate", "0.0")
                lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")
                lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")
            else:
                lidar_bp.set_attribute("noise_stddev", "0.2")

        lidar_bp.set_attribute("upper_fov", arg["upper_fov"])
        lidar_bp.set_attribute("lower_fov", str(arg["lower_fov"]))
        lidar_bp.set_attribute("channels", str(arg["channels"]))
        lidar_bp.set_attribute("range", str(arg["range"]))
        lidar_bp.set_attribute("rotation_frequency", str(1.0 / delta))
        lidar_bp.set_attribute("points_per_second", str(arg["points_per_second"]))
        return lidar_bp

    def close_client(self) -> None:
        self.carla_client.stop_replayer(False)

        settings = self.carla_world.get_settings()
        traffic_manager = self.carla_client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(False)

        settings.fixed_delta_seconds = None
        settings.synchronous_mode = False
        settings.no_rendering_mode = not self.rendering
        self.carla_world.apply_settings(settings)

        del self.carla_client
