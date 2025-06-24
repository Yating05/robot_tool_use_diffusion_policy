import json
import time
import argparse
from typing import Dict, Any, Callable, Optional, Union, List
import numpy as np
import torch

from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped, WrenchStamped, Wrench
from victor_hardware_interfaces.msg import (
    MotionStatus, 
    Robotiq3FingerCommand, 
    Robotiq3FingerStatus,
    JointValueQuantity,
    Robotiq3FingerActuatorCommand
)

def wrench_to_tensor(msg: WrenchStamped, device: str = "cpu") -> Optional[torch.Tensor]:
    w = msg.wrench
    wrench = [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]
    return torch.tensor(wrench, dtype=torch.float32, device=device)

def gripper_status_to_tensor(msg: Robotiq3FingerStatus, device: str = "cpu"):
    left_gripper = msg
    gripper_obs = [left_gripper.finger_a_status.position_request, left_gripper.finger_a_status.position,
                    left_gripper.finger_b_status.position_request, left_gripper.finger_b_status.position,
                    left_gripper.finger_c_status.position_request, left_gripper.finger_c_status.position,
                    left_gripper.scissor_status.position_request, left_gripper.scissor_status.position]
    return torch.tensor(gripper_obs, dtype=torch.float32, device=device)