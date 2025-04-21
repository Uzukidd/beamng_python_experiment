
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
    def __init__(self, client, vehicle, lidar_para={}, callback=None, logger=None, need_gt=False, pcs_cache=False) -> None:
        self.stream_thread = Thread(target=self.update_stream, args=[])
        self.vehicle = vehicle
        self.callback = callback
        self.lidar = Lidar('lidar', client, vehicle, **lidar_para)
        self.logger = logger
        self.need_gt = need_gt

        self.pcs_cache = pcs_cache

    def get_single_frame(self) -> np.array:
        def rotate(points, n):
            # Step 1
            n = n / np.linalg.norm(n)
            # Step 2
            theta = np.arctan2(n[1], n[0])
            phi = -np.arctan2(n[2], np.sqrt(n[0]**2 + n[1]**2))
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
        _ = np.zeros(shape=(count, 4))
        _[:, 0:3] = points_np
        points_np = _

        if self.pcs_cache:
            points_np.astype(np.float32).tofile("./.np_cache/beamng_pcs.bin")

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
        "upper_fov": "2.0",
        "lower_fov": "-24.8",
        "channels": "64.0",
        "range": "120.0",
        "points_per_second": "3300000",
        "semantic": False,
        "no_noise": False,
        "delta": 0.05,
        # "rotation_frequency":"20"
    }, logger=None, need_gt=False, pcs_cache=False, pcs_frames_cache=1) -> None:
        # self.pcs_frames = Queue(pcs_frames_cache)
        self.pcs_frames = None
        self.vehicle = vehicle
        self.carla_world = carla_world
        self.logger = logger
        self.lidar_para = lidar_para
        self.lidar = None

        self.need_gt = need_gt
        self.ground_truth = None

        self.pcs_cache = pcs_cache

    def init_lidar(self):
        lidar_bp = self.generate_lidar_bp(self.lidar_para, self.carla_world)

        user_offset = carla.Location(0.0, 0.0, 0.0)
        lidar_transform = carla.Transform(
            carla.Location(x=-0.5, z=1.8) + user_offset)

        self.lidar = self.carla_world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.lidar.listen(self._pcs_callback)

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
                lidar_bp.set_attribute('noise_stddev', '0.02')

        lidar_bp.set_attribute('upper_fov', arg["upper_fov"])
        lidar_bp.set_attribute('lower_fov', str(arg["lower_fov"]))
        lidar_bp.set_attribute('channels', str(arg["channels"]))
        lidar_bp.set_attribute('range', str(arg["range"]))
        lidar_bp.set_attribute('rotation_frequency', str(1/arg["delta"]))
        lidar_bp.set_attribute('points_per_second',
                               str(arg["points_per_second"]))
        return lidar_bp

    def _update_ground_truth(self) -> None:
        ground_truth = []

        def rotate_yaw(points, yaw):
            theta = yaw
            R_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])

            points = np.dot(points, R_z)
            return points

        def rotate_pitch(points, pitch):
            theta = pitch
            R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)],
                            ])

            points = np.dot(points, R_y)
            return points

        for npc in self.carla_world.get_actors().filter('*vehicle*'):
            if npc.id != self.vehicle.id:
                dist = npc.get_transform().location.distance(
                    self.lidar.get_transform().location)

                if dist < 120:
                    # coordinate transformation bbox -> world -> lidar(ego vehicle)
                    bounding_box_loc = npc.bounding_box.location + npc.get_transform().location - \
                        self.lidar.get_transform().location
                    bounding_box_lwh = npc.bounding_box.extent
                    bounding_box_rot = npc.bounding_box.rotation.yaw + \
                        npc.get_transform().rotation.yaw - self.lidar.get_transform().rotation.yaw

                    boudning_box_np = np.zeros(7)

                    boudning_box_np[0] = bounding_box_loc.x
                    boudning_box_np[1] = bounding_box_loc.y
                    boudning_box_np[2] = bounding_box_loc.z

                    # semi lwh -> full lwh
                    boudning_box_np[3] = bounding_box_lwh.x * 2.0
                    boudning_box_np[4] = bounding_box_lwh.y * 2.0
                    boudning_box_np[5] = bounding_box_lwh.z * 2.0

                    boudning_box_np[6] = -np.radians(bounding_box_rot)

                    boudning_box_np[0:3] = rotate_yaw(boudning_box_np[0:3], np.radians(
                        self.lidar.get_transform().rotation.yaw))
                    boudning_box_np[0:3] = rotate_pitch(boudning_box_np[0:3], np.radians(
                        -self.lidar.get_transform().rotation.pitch))
                    boudning_box_np[1] = -boudning_box_np[1]
                    ground_truth.append(boudning_box_np)

        if ground_truth.__len__() != 0:
            self.ground_truth = np.stack(ground_truth)
        else:
            self.ground_truth = None

    def get_single_frame(self) -> np.array:
        res = self.pcs_frames
        if self.need_gt:
            self._update_ground_truth()
        # try:
        #     res = self.pcs_frames.get(True, 1.0)
        # except:
        #     pass
        return res, self.ground_truth

    def _pcs_callback(self, point_cloud) -> None:
        data = np.copy(np.frombuffer(
            point_cloud.raw_data, dtype=np.dtype('f4')))
        data = data.reshape((-1, 4))
        data[:, 1] = -data[:, 1]
        # data[:, 3] = 0
        # self.pcs_frames.put(data)
        self.pcs_frames = data

        if self.pcs_cache:
            data.astype(np.float32).tofile("./.np_cache/carla_pcs.bin")
