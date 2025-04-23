# ad-python-experiment

https://github.com/Uzukidd/beamng_python_experiment/assets/53105227/c22ee48c-eca6-4ddb-a4d7-1afd7d38181a

## Dependencies

* OpenPCDet (Optional, if you want to use the model built using OpenPCDet)
* CarLA Client or BeamNG Client

## Usage

* CarLA
  ```
  usage: carla_replay.py [-h] [--host H] [-p P] [-f F] [-c CONFIG_FILENAME] [-r ROLENAME] [-v] [-n]
                         [-e]

  options:
    -h, --help            show this help message and exit
    --host H              IP of the host server (default: 127.0.0.1)
    -p P, --port P        TCP port to listen to (default: 2000)
    -f F, --recorder-filename F
                          recorder filename (test1.log)
    -c CONFIG_FILENAME, --config-filename CONFIG_FILENAME
                          config filename (*.yaml)
    -r ROLENAME, --rolename ROLENAME
                          rolename of the ego vehicle
    -v, --preview         open an open3d windows to preview current frame
    -n, --noisy-lidar     switch to noisy lidar
    -e, --evaluate        evaluate result after finishing repla
  ```
* BeamNG
  * see `main_playground.ipynb` for more details.

## Change log

[2025.4.23] Fixed open3d windows blank screen.
