import os
from typing import Optional

import numpy as np
import yaml

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import choose_from_options

from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveErrorGroup
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitives,
    SymbolicSemanticActionPrimitiveSet,
)
from omnigibson.metrics.task_metric import TaskMetric

# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def _open_if_needed(container, controller, env):
    """
    Try to open a receptacle before placing items inside it.
    """
    if object_states.Open in container.states and not container.states[object_states.Open].get_value():
        print(f"Opening {container.name} before placing items inside")
        execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.OPEN, container), env)


def _fallback_teleport_inside(container, controller, env) -> bool:
    """
    Release the held object and teleport it into the container as a last resort if semantic sampling fails.
    Returns True on success.
    """
    held = controller._get_obj_in_hand()
    if held is None:
        print("Fallback placement skipped: no object in hand.")
        return False

    # Release grasp without stepping so we can immediately reposition.
    for arm in controller.robot.arm_names:
        controller.robot.release_grasp_immediately(arm=arm)

    lower, upper = container.states[object_states.AABB].get_value()
    held_lower, held_upper = held.states[object_states.AABB].get_value()
    held_height = float((held_upper - held_lower)[2])

    # Drop near the container center, slightly below the rim.
    target_xy = (lower[:2] + upper[:2]) / 2.0
    rim_z = float(upper[2])
    floor_z = float(lower[2])
    target_z = min(rim_z - 0.02, max(floor_z + held_height * 0.5, rim_z - held_height * 0.8))
    target_pos = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)

    held.set_position_orientation(position=target_pos, orientation=[0, 0, 0, 1])
    og.sim.step()

    success = bool(held.states[object_states.Inside].get_value(container))
    if success:
        print(f"Fallback placement succeeded for {held.name} -> {container.name}")
    else:
        print(f"Fallback placement still not inside for {held.name} -> {container.name}")
    return success


def place_inside_with_retries(container, controller, env):
    """
    Try semantic placement first, then fall back to a deterministic teleport if sampling fails.
    """
    _open_if_needed(container, controller, env)
    try:
        execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_INSIDE, container), env)
        return
    except ActionPrimitiveErrorGroup as e:
        print(f"Semantic PLACE_INSIDE failed for {container.name}, using fallback. Error: {e}")

    if not _fallback_teleport_inside(container, controller, env):
        raise RuntimeError(f"Failed to place object inside {container.name} using both semantic and fallback paths.")


def report_task_objects(env):
    """
    Print whether all BDDL task instances are bound to real objects/systems in the scene.
    """
    task = env.task
    scene = env.scene
    print("Task object scope status:")
    for inst, entity in task.object_scope.items():
        kind = "system" if entity.is_system else "object"
        status = "loaded" if entity.exists else "MISSING"
        name = entity.name if entity.exists else "None"
        line = f"  {inst} [{kind}, synset={entity.synset}]: {status} (name={name})"
        if not entity.exists:
            candidates = [
                obj.name
                for obj in scene.objects
                if hasattr(obj, "category") and obj.category in set(entity.og_categories)
            ]
            if candidates:
                line += f" | candidates by category: {candidates[:5]}"
        print(line)


def log_loaded_task_objects(env):
    """
    Return and print loaded task-bound objects for quick inspection.
    """
    loaded = [entity.wrapped_obj for entity in env.task.object_scope.values() if entity.exists]
    print("Loaded task objects:", [obj.name for obj in loaded])
    return loaded


def compute_goal_completion(env):
    """
    Compute BDDL goal completion ratio: per grounded option, count satisfied predicates / total,
    then take the max ratio across options (same logic as TaskMetric partial credit).
    Returns (best_ratio, per_option_ratios).
    """
    option_ratios = []
    for option in env.task.ground_goal_state_options:
        satisfied = sum(1 for pred in option if pred.evaluate())
        option_ratios.append(satisfied / len(option) if option else 0.0)
    best_ratio = max(option_ratios) if option_ratios else 0.0
    return best_ratio, option_ratios



def main(random_selection=False, headless=False, short_exec=False):
    """
    Generates a BEHAVIOR Task environment in an online fashion.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "clean_up.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Load the environment
    env = og.Environment(configs=cfg)
    scene = env.scene
    # Reset the robot
    robot = env.robots[0]
    robot.set_position_orientation(position=[5, 5, 0.05])
    robot.reset()
    robot.keep_still()

    # Print the names of all loaded objects for quick inspection
    obj_names = sorted([obj.name for obj in scene.objects])
    print(f"Loaded objects ({len(obj_names)}): {obj_names}")
    report_task_objects(env)
    log_loaded_task_objects(env)

    # Move camera to a good position
    og.sim.viewer_camera.set_position_orientation(
        position=[1.6, 6.15, 1.5], orientation=[-0.2322, 0.5895, 0.7199, -0.2835]
    )

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Use symbolic primitives to teleport objects directly instead of running full motion planning / physics
    controller = SymbolicSemanticActionPrimitives(env, robot)
    # Track BDDL goal progress / success
    task_metric = TaskMetric()
    task_metric.start_callback(env)

    # Find a target table
    for _ in range(10000):
        og.sim.step()

    target = scene.object_registry("name", "desk_bhkhxo_0")
    objects = [
        obj for obj in scene.objects
        if "laptop" in (getattr(obj, "category", "") or "") or "laptop" in obj.name
    ]

    for obj in objects:
        print(f"Executing grasp for {obj.name}")
        execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, obj), env)
        print(f"Executing place_on_top for {obj.name} onto target")
        execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP, target), env)

    print(f"Finished placing {obj.name}")

    for _ in range(100):
        og.sim.step()

    # Evaluate success / partial credit based on BDDL goals
    task_metric.end_callback(env)
    results = task_metric.gather_results()
    print("Task success:", env.task.success)
    print("Task q_score:", results["q_score"]["final"])
    best_ratio, option_ratios = compute_goal_completion(env)
    print(f"Goal completion ratio (max over options): {best_ratio:.3f}, per option: {option_ratios}")

    # Always close the environment at the end
    og.shutdown()


if __name__ == "__main__":
    main()
