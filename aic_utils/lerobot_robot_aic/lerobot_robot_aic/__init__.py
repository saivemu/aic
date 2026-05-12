#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# Defer cv2-dependent submodule imports. Eagerly importing them broke
# `lerobot-train` (cv2 -> libtiff hits an ABI mismatch when conda's libjpeg
# is pre-loaded by torch/transformers via lerobot-train's discovery scan).
# The submodules still work for runtime callers — they just lazy-load.
from importlib import import_module
from typing import TYPE_CHECKING

_LAZY: dict[str, str] = {
    "AICRobotAICController": ".aic_robot_aic_controller",
    "AICRobotAICControllerConfig": ".aic_robot_aic_controller",
    "AICKeyboardEETeleop": ".aic_teleop",
    "AICKeyboardEETeleopConfig": ".aic_teleop",
    "AICKeyboardJointTeleop": ".aic_teleop",
    "AICKeyboardJointTeleopConfig": ".aic_teleop",
    "AICSpaceMouseTeleop": ".aic_teleop",
    "AICSpaceMouseTeleopConfig": ".aic_teleop",
}


def __getattr__(name: str):
    mod_path = _LAZY.get(name)
    if mod_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = import_module(mod_path, __name__)
    val = getattr(mod, name)
    globals()[name] = val
    return val


if TYPE_CHECKING:
    from .aic_robot_aic_controller import (
        AICRobotAICController,
        AICRobotAICControllerConfig,
    )
    from .aic_teleop import (
        AICKeyboardEETeleop,
        AICKeyboardEETeleopConfig,
        AICKeyboardJointTeleop,
        AICKeyboardJointTeleopConfig,
        AICSpaceMouseTeleop,
        AICSpaceMouseTeleopConfig,
    )
