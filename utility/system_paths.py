""" Module managing system path configurations. """

import socket
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SystemBasePaths:
    """ Dataclass which stores the basic path configuration of a specific system. """
    description: str    # easy-to-read description
    hostname: str       # output from socket.gethostname()
    log_dir: Path       # path to logs folder (where experiment logs are stored and saved)
    root_dir: Path      # path to dataset (should contain sub-folders 'Dataset', 'HD_PointCloud_Tiles', etc.)


def get_known_systems():
    """ Returns list of all known systems (in SystemBasePaths format). If this project is run on a new system, make
        sure to configure the basic paths here first.
    """
    known_systems = [  # add your own system here
        SystemBasePaths(description='Windows example',
                        hostname='example-hostname',
                        log_dir=Path(r"C:\example\path\to\logs"),
                        root_dir=Path(r"C:\example\path\to\3DHD_CityScenes")
                        ),
        SystemBasePaths(description='Linux example',
                        hostname='example-hostname',
                        log_dir=Path("/example/path/to/logs"),
                        root_dir=Path("/example/path/to/3DHD_CityScenes")
                        ),
    ]
    return known_systems


def get_current_host():
    current_host = socket.gethostname()
    return current_host


def is_system_known():
    current_host = get_current_host()
    known_hosts = [s.hostname for s in get_known_systems()]
    return current_host in known_hosts


def get_system_paths():
    current_host = get_current_host()

    if not is_system_known():
        help_str = f"Unknown system '{current_host}'! Make sure to configure the basic paths in " \
                   f"utility/system_paths.py (get_known_systems()), using {current_host} as hostname."
        raise ValueError(help_str)

    paths_obj = [s for s in get_known_systems() if s.hostname == current_host][0]
    log_dir = paths_obj.log_dir
    root_dir = paths_obj.root_dir

    return {'log_dir': log_dir,
            'root_dir': root_dir,
            'dataset_dir': root_dir / 'Dataset',
            'hdpc_tiles_dir': root_dir / 'HD_PointCloud_Tiles',
            'map_metadata_dir': root_dir / 'HD_Map_MetaData',
            'map_base_dir': root_dir / 'HD_Map',
            'samples_dir': root_dir / 'Samples',
            'host_system': current_host}


def main():
    print(f"Hostname: {socket.gethostname()}")


if __name__ == "__main__":
    main()
