from beamngpy import BeamNGpy, Scenario, Vehicle

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
        scenario = Scenario('smallgrid', 'tag')
        av_a = Vehicle('vehicleA', model='etk800')
        av_b = Vehicle('vehicleB', model='etk800')
        scenario.add_vehicle(av_a, pos=(0, -10, 0))
        scenario.add_vehicle(av_b)
        scenario.make(self.beamng_terminal)
        self.beamng_terminal.scenario.load(scenario)
        self.beamng_terminal.scenario.start()
        
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
    def __init__(self, host = "127.0.0.1", port = 64256) -> None:
        self.beamng_terminal = None
        self.host = host
        self.port = port
        
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
        
    def debug_luanch_test(self) -> None:
        SIZE = 1024
        window = self.debug_open_window(SIZE, SIZE)
        lidar_vis = LidarVisualiser(MAX_LIDAR_POINTS)
        lidar_vis.open(SIZE, SIZE)
        running_scenario = self.beamng_terminal.scenario.get_current()
        print(running_scenario.name)
        active_vehicles = self.beamng_terminal.vehicles.get_current()
        print(active_vehicles)
        vehicle = active_vehicles["vehicleA"]
        vehicle.connect(self.beamng_terminal)
        lidar_t = lidar(self.beamng_terminal, vehicle)
        lidar_t.start_stream()
        
        
    def disconnect_client(self) -> None:
        self.beamng_terminal.disconnect()