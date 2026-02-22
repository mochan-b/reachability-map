from __future__ import annotations

import os

from roboticstoolbox.robot import ERobot


def load_robot(urdf_path: str, gripper_link: str) -> ERobot:
    """Load a robot from a URDF file and return an ERobot instance.

    roboticstoolbox requires URDF_read to be called on a proper ERobot
    subclass, so we create a minimal subclass dynamically that captures
    the URDF path and gripper link via closure.

    Parameters
    ----------
    urdf_path:
        Absolute or relative path to the URDF file.
    gripper_link:
        Name of the link to use as the end-effector / gripper.

    Returns
    -------
    ERobot
        Loaded robot with gripper_links set.

    Raises
    ------
    ValueError
        If gripper_link is not found among the links in the URDF.
    """
    urdf_path = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(urdf_path)
    urdf_filename = os.path.basename(urdf_path)

    class _DynamicRobot(ERobot):
        def __init__(self):
            links, name, urdf_str, urdf_fp = self.URDF_read(
                file_path=urdf_filename,
                tld=urdf_dir,
            )
            gripper = next((l for l in links if l.name == gripper_link), None)
            if gripper is None:
                available = [l.name for l in links]
                raise ValueError(
                    f"Link '{gripper_link}' not found in URDF. "
                    f"Available links: {available}"
                )
            super().__init__(
                links,
                name=name,
                gripper_links=gripper,
                urdf_string=urdf_str,
                urdf_filepath=urdf_fp,
            )

    return _DynamicRobot()


def get_ets(robot: ERobot, gripper_link: str):
    """Return the picklable ETS kinematic chain to the gripper link.

    The ETS object (unlike ERobot) is safe to pass across multiprocessing
    boundaries, so workers use this instead of the full robot object.
    """
    return robot.ets(end=gripper_link)
