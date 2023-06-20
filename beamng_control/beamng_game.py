from beamngpy import BeamNGpy, Scenario, Vehicle, angle_to_quat

class beamng_server:
    def __init__(self, host = "127.0.0.1", port = 64256) -> None:
        self.beamng_terminal = None
        self.host = host
        self.port = port
        
    def init_server(self) -> None:
        self.beamng_terminal = BeamNGpy(self.host, self.port)
        
    def launch_server(self) -> None:
        self.beamng_terminal.open(launch=True)
        
    def debug_luanch_test(self) -> None:
        scenario = Scenario('west_coast_usa', 'tag')
        av_a = Vehicle('vehicleA', model='etk800')

        scenario.add_vehicle(av_a, pos=(-712.591, 535.908, 119.860), rot_quat=angle_to_quat((-0.221, 0.254, -64.008)))
        # scenario.add_vehicle(av_a, pos=(-712.591, 535.908, 119.860), rot_quat=angle_to_quat((0.0, 0.0, 180.0)))

        # scenario.add_vehicle(av_b)
        scenario.make(self.beamng_terminal)
        self.beamng_terminal.scenario.load(scenario)
        self.beamng_terminal.scenario.start()
        
    # def debug_luanch_test(self) -> None:
    #     scenario = Scenario('smallgrid', 'tag')
    #     av_a = Vehicle('vehicleA', model='etk800')
    #     av_b = Vehicle('vehicleB', model='etk800')
    #     scenario.add_vehicle(av_a, pos=(0, -10, 0))
    #     scenario.add_vehicle(av_b)
    #     scenario.make(self.beamng_terminal)
    #     self.beamng_terminal.scenario.load(scenario)
    #     self.beamng_terminal.scenario.start()
        
    def close_server(self) -> None:
        self.beamng_terminal.close()
        
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Lidar
from beamngpy.sensors.lidar import MAX_LIDAR_POINTS, LidarVisualiser

from sensor_std import lidar
    
class beamng_client:
    def __init__(self, host = "127.0.0.1", port = 64256, logger = None) -> None:
        self.beamng_terminal = None
        self.host = host
        self.port = port
        self.logger = logger
        
        self.lidar_t = None
        
    def init_client(self) -> None:
        self.beamng_terminal = BeamNGpy(self.host, self.port)
        
    def launch_client(self) -> None:
        self.beamng_terminal.open(launch=False)
        
    def debug_lidar_resize(self, width, height):
        if height == 0:
            height = 1

        glViewport(0, 0, width, height)


    def debug_open_window(self, width, height):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
        glutInitWindowSize(width, height)
        window = glutCreateWindow(b'Lidar Tour')
        self.debug_lidar_resize(width, height)
        return window
        
    def debug_luanch_test(self, callback=None, lidar_para={}) -> None:
        running_scenario = self.beamng_terminal.scenario.get_current()
        print(running_scenario.name)
        active_vehicles = self.beamng_terminal.vehicles.get_current()
        print(active_vehicles)
        vehicle = active_vehicles["vehicleA"]
        vehicle.connect(self.beamng_terminal)
        vehicle.ai.set_mode('disabled')
        self.lidar_t = lidar(self.beamng_terminal, vehicle, lidar_para, callback, self.logger)
        # lidar_t.start_stream()
        
        
    def disconnect_client(self) -> None:
        self.beamng_terminal.disconnect()