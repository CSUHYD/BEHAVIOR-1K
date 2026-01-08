"""
Example script demo'ing robot manipulation control with grasping.
"""
import os
import yaml
import torch as th

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.sensors import VisionSensor
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveErrorGroup
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitives,
    SymbolicSemanticActionPrimitiveSet,
)
from omnigibson.metrics.task_metric import TaskMetric

# Don't use GPU dynamics and Use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def _open_if_needed(container, controller, env):
    if object_states.Open in container.states and not container.states[object_states.Open].get_value():
        execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.OPEN, container), env)


def _fallback_teleport_inside(container, controller) -> bool:
    held = controller._get_obj_in_hand()
    if held is None:
        return False

    for arm in controller.robot.arm_names:
        controller.robot.release_grasp_immediately(arm=arm)

    lower, upper = container.states[object_states.AABB].get_value()
    held_lower, held_upper = held.states[object_states.AABB].get_value()
    held_height = float((held_upper - held_lower)[2])
    target_xy = (lower[:2] + upper[:2]) / 2.0
    rim_z = float(upper[2])
    floor_z = float(lower[2])
    target_z = min(rim_z - 0.02, max(floor_z + held_height * 0.5, rim_z - held_height * 0.8))
    target_pos = [float(target_xy[0]), float(target_xy[1]), target_z]

    held.set_position_orientation(position=target_pos, orientation=[0, 0, 0, 1])
    og.sim.step()
    return bool(held.states[object_states.Inside].get_value(container))


def place_inside_with_retries(container, controller, env):
    _open_if_needed(container, controller, env)
    try:
        execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP, container), env)
        return
    except ActionPrimitiveErrorGroup:
        pass

    if not _fallback_teleport_inside(container, controller):
        raise RuntimeError(f"Failed to place object inside {container.name}.")


def _find_objects(scene, obj_type):
    obj_type = (obj_type or "").lower()
    if obj_type == "can":
        include_tags = ("can",)
        exclude_tags = ("ashcan", "trash", "trash_can", "trashcan")
    elif obj_type == "ashcan":
        include_tags = ("ashcan", "trash", "trash_can", "trashcan")
        exclude_tags = ()
    else:
        raise ValueError(f"Unsupported obj_type: {obj_type}")

    matches = []
    for obj in scene.objects:
        name = (obj.name or "").lower()
        category = (getattr(obj, "category", "") or "").lower()
        if any(tag in name or tag in category for tag in include_tags):
            if any(tag in name or tag in category for tag in exclude_tags):
                continue
            matches.append(obj)
    return matches


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


def run_can_into_ashcan(env, robot):
    controller = SymbolicSemanticActionPrimitives(env, robot)
    can_objs = _find_objects(env.scene, "can")
    ashcan_objs = _find_objects(env.scene, "ashcan")
    ashcan_obj = ashcan_objs[0] if ashcan_objs else None
    if not can_objs or ashcan_obj is None:
        raise RuntimeError("Expected at least one can object and an ashcan/trash can to exist in the scene.")
    for can_obj in can_objs[:-1]:
        print(f"Executing grasp for {can_obj.name}")
        execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, can_obj), env)
        print(f"Executing place_inside for {can_obj.name} into {ashcan_obj.name}")
        place_inside_with_retries(ashcan_obj, controller, env)


def run_grasp_mode(env, robot, random_selection=False, short_exec=False):
    mode = choose_from_options(
        options=["symbolic", "keyboard"],
        name="grasp mode"
    )

    if mode == "symbolic":
        run_can_into_ashcan(env, robot)
    else:
        run_keyboard_grasp(env, robot, random_selection=random_selection, short_exec=short_exec)


def run_with_task_metric(env, run_fn):
    task_metric = TaskMetric()
    task_metric.start_callback(env)
    run_fn()
    task_metric.end_callback(env)
    results = task_metric.gather_results()
    print("TaskMetric results:", results)
    return task_metric


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
        position=[4.0531, 5.6290, 2.1589], orientation=[-0.0228,  0.5292,  0.8474, -0.0365]
    )
    og.sim.enable_viewer_camera_teleoperation()

    run_with_task_metric(
        env,
        lambda: run_grasp_mode(env, robot, random_selection=random_selection, short_exec=short_exec),
    )

    # Always shut down the environment cleanly at the end
    og.shutdown()


if __name__ == "__main__":
    main()
