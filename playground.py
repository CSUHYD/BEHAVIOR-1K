"""
Example script demo'ing robot manipulation control with grasping.
"""
import os
import yaml
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.sensors import VisionSensor
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitives,
    SymbolicSemanticActionPrimitiveSet,
)

# Don't use GPU dynamics and Use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def run_symbolic_grasp(env, robot, grasp_obj, target_obj):
    controller = SymbolicSemanticActionPrimitives(env, robot)
    print(f"Executing grasp for {grasp_obj.name}")
    execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, grasp_obj), env)
    print(f"Executing place_on_top for {grasp_obj.name} onto {target_obj.name}")
    execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP, target_obj), env)


def run_keyboard_grasp(env, robot, random_selection=False, short_exec=False):
    action_generator = KeyboardRobotController(robot=robot)
    action_generator.print_keyboard_teleop_info()
    print("Press ESC to quit")

    max_steps = -1 if not short_exec else 100
    step = 0
    while step != max_steps:
        if random_selection:
            if step % 30 == 0:
                random_action = action_generator.get_random_action() * 0.05
            action = random_action
        else:
            action = action_generator.get_teleop_action()
        env.step(action=action)
        step += 1


def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot grasping mode demo with selection
    Queries the user to select a type of grasping mode
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Load the config
    config_filename = os.path.join(og.example_config_path, "playground.yaml")
    with open(config_filename, "r") as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)

    # Create the environment
    env = og.Environment(configs=cfg)

    # Reset the robot
    robot = env.robots[0]
    robot.set_position_orientation(position=[3, 3, 0.2])
    robot.reset()
    robot.keep_still()

    # Open the gripper(s) to match cuRobo's default state
    for arm_name in robot.gripper_control_idx.keys():
        gripper_control_idx = robot.gripper_control_idx[arm_name]
        robot.set_joint_positions(th.ones_like(gripper_control_idx), indices=gripper_control_idx, normalized=True)
    robot.keep_still()

    for _ in range(5):
        og.sim.step()

    env.scene.update_initial_file()
    env.scene.reset()

    # Make the robot's camera(s) high-res
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.image_height = 720
            sensor.image_width = 720

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([-2.39951, 2.26469, 2.66227]),
        orientation=th.tensor([-0.23898481, 0.48475231, 0.75464013, -0.37204802]),
    )
    og.sim.enable_viewer_camera_teleoperation()

    mode = choose_from_options(
        options=["symbolic", "keyboard"],
        name="grasp mode"
    )

    if mode == "symbolic":
        grasp_obj = env.scene.object_registry("name", "box")
        target_obj = env.scene.object_registry("name", "chair")
        if grasp_obj is None or target_obj is None:
            raise RuntimeError("Expected 'box' and 'chair' objects to exist for symbolic mode.")
        run_symbolic_grasp(env, robot, grasp_obj, target_obj)
    else:
        run_keyboard_grasp(env, robot, random_selection=random_selection, short_exec=short_exec)

    # Always shut down the environment cleanly at the end
    og.shutdown()


if __name__ == "__main__":
    main()