"""
Example script demo'ing robot manipulation control with grasping.
"""
import os
import yaml
import torch as th

import omnigibson as og
from omnigibson import object_states
from omnigibson.sensors import VisionSensor
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveErrorGroup
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitives,
    SymbolicSemanticActionPrimitiveSet,
)
from omnigibson.metrics.task_metric import TaskMetric
from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY
from bddl3.bddl.action_translator import BDDLActionTranslator, ACTION_ENUM_DEFAULT

# Don't use GPU dynamics and Use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True


def _print_task_action_plan(
    behavior_activity: str = "picking_up_trash", activity_definition: int = 0
):
    """
    Parse a BDDL task and print the grounded action plans.
    """

    translator = BDDLActionTranslator(action_enum=ACTION_ENUM_DEFAULT)
    plans = translator.translate(
        behavior_activity=behavior_activity, activity_definition=activity_definition
    )
    print(f"Action plans for task '{behavior_activity}' (definition {activity_definition}): {len(plans)} option(s)")
    if not plans:
        return

    for idx, plan in enumerate(plans, 1):
        print(f"  Plan {idx}:")
        for line in plan.to_strings():
            print(f"    - {line}")




def translate_task_plans(
    behavior_activity: str, activity_definition: int, action_enum=SymbolicSemanticActionPrimitiveSet
):
    """
    Translate a BDDL task into grounded action plans using the requested primitive enum.
    """

    translator = BDDLActionTranslator(action_enum=action_enum)
    return translator.translate(
        behavior_activity=behavior_activity, activity_definition=activity_definition
    )


def _find_objects(env, task, bddl_inst: str):
    """
    Resolve a BDDL instance name to a scene object using resolve_bddl_instance.
    Returns the resolved BaseObject / BaseSystem.
    """

    mapping, missing = resolve_bddl_instance(env, task, bddl_inst)
    if missing:
        raise RuntimeError(f"Failed to resolve BDDL instance '{bddl_inst}'.")
    return mapping[bddl_inst]


def resolve_bddl_instance(env, task, bddl_inst: str = None):
    """
    解析 task BDDL 中所有相关物体（或指定单个实例）。
    返回 (mapping, missing)，其中 mapping 为 {bddl_inst: entity}。
    解析顺序：
    1) task.object_scope
    2) task metadata 的 inst_to_name
    3) synset -> category 映射
    """

    def _resolve_one(inst):
        if hasattr(task, "object_scope") and inst in task.object_scope:
            entity = task.object_scope[inst]
            if entity.exists:
                return entity.wrapped_obj

        inst_to_name = env.scene.get_task_metadata(key="inst_to_name")
        if inst_to_name and inst in inst_to_name:
            name = inst_to_name[inst]
            if name in env.scene.available_systems:
                return env.scene.get_system(name)
            obj = env.scene.object_registry("name", name)
            if obj is not None:
                return obj

        synset = "_".join(inst.split("_")[:-1])
        try:
            categories = OBJECT_TAXONOMY.get_categories(synset)
        except Exception:
            categories = []
        for category in categories:
            objs = env.scene.object_registry("category", category, [])
            if objs:
                return objs[0]
        return None

    if bddl_inst is not None:
        entity = _resolve_one(bddl_inst)
        mapping = {bddl_inst: entity} if entity is not None else {}
        missing = [] if entity is not None else [bddl_inst]
        return mapping, missing

    if hasattr(task, "object_scope") and task.object_scope:
        instances = list(task.object_scope.keys())
    else:
        inst_to_name = env.scene.get_task_metadata(key="inst_to_name") or {}
        instances = list(inst_to_name.keys())

    mapping = {}
    missing = []
    for inst in instances:
        entity = _resolve_one(inst)
        if entity is None:
            missing.append(inst)
        else:
            mapping[inst] = entity
    return mapping, missing


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def _open_if_needed(container, controller, env):
    if object_states.Open in container.states and not container.states[object_states.Open].get_value():
        execute_controller(
            controller.apply_ref(SymbolicSemanticActionPrimitiveSet.OPEN, container),
            env,
        )

def place_inside_with_retries(container, controller, env):
    _open_if_needed(container, controller, env)
    try:
        execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP, container), env)
        return
    except ActionPrimitiveErrorGroup:
        pass

    if not _fallback_teleport_inside(container, controller):
        raise RuntimeError(f"Failed to place object inside {container.name}.")


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
    target_z = min(rim_z - 0.02, max(floor_z + held_height * 0.5, rim_z - held_height * 0.8)) + 0.5
    target_pos = [float(target_xy[0]), float(target_xy[1]), target_z]

    held.set_position_orientation(position=target_pos, orientation=[0, 0, 0, 1])
    for i in range(100):
        og.sim.step()
    return bool(held.states[object_states.Inside].get_value(container))



def execute_action_plan(env, robot, plan):
    """
    Execute a grounded action plan using symbolic semantic primitives.
    """

    controller = SymbolicSemanticActionPrimitives(env, robot)
    for step in plan.steps:
        prim = step.primitive
        if prim == SymbolicSemanticActionPrimitiveSet.GRASP:
            obj = _find_objects(env, env.task, step.primary_object)
            print(f"Executing grasp for {obj.name}")
            execute_controller(controller.apply_ref(prim, obj), env)
        elif prim == SymbolicSemanticActionPrimitiveSet.PLACE_INSIDE:
            if not step.reference_object:
                raise RuntimeError("PLACE_INSIDE requires reference_object.")
            obj = _find_objects(env, env.task, step.primary_object)
            ref_obj = _find_objects(env, env.task, step.reference_object)
            print(f"Executing place_inside for {obj.name} into {ref_obj.name}")
            place_inside_with_retries(ref_obj, controller, env)
        elif prim == SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP:
            if not step.reference_object:
                raise RuntimeError("PLACE_ON_TOP requires reference_object.")
            obj = _find_objects(env, env.task, step.primary_object)
            ref_obj = _find_objects(env, env.task, step.reference_object)
            print(f"Executing place_on_top for {obj.name} onto {ref_obj.name}")
            execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP, ref_obj), env)
        elif prim in (SymbolicSemanticActionPrimitiveSet.OPEN, SymbolicSemanticActionPrimitiveSet.CLOSE):
            obj = _find_objects(env, env.task, step.primary_object)
            print(f"Executing {prim.name.lower()} for {obj.name}")
            execute_controller(controller.apply_ref(prim, obj), env)
        else:
            raise RuntimeError(f"Unsupported primitive for execution: {prim}")


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


def run_task_mode(
    env,
    robot,
    behavior_activity: str,
    activity_definition: int = 0,
    random_selection: bool = False,
    short_exec: bool = False,
):
    """
    Generic task runner: print a grounded plan, execute a grounded plan, or fall back to keyboard teleop.
    """

    mode = choose_from_options(
        options=["print_plan", "execute_plan", "keyboard"],
        name="task mode",
    )

    if mode == "print_plan":
        _print_task_action_plan(behavior_activity, activity_definition)
        return

    if mode == "execute_plan":
        plans = translate_task_plans(
            behavior_activity=behavior_activity,
            activity_definition=activity_definition,
            action_enum=SymbolicSemanticActionPrimitiveSet,
        )
        if not plans:
            print("No grounded plans found for this task.")
            return
        # 场景加载完成后，报告已加载物体，以及是否包含任务相关名称
        relevant_names = set()
        for plan in plans:
            for step in plan.steps:
                relevant_names.add(step.primary_object)
                if step.reference_object:
                    relevant_names.add(step.reference_object)
        print(f"Executing all {len(plans)} grounded plan(s):")
        for idx, plan_to_run in enumerate(plans, 1):
            print(f"Plan {idx}:")
            for line in plan_to_run.to_strings():
                print(f"  - {line}")
            execute_action_plan(env, robot, plan_to_run)
        return

    run_keyboard_grasp(env, robot, random_selection=random_selection, short_exec=short_exec)


def run_with_task_metric(env, run_fn):
    task_metric = TaskMetric()
    task_metric.start_callback(env)
    run_fn()
    task_metric.end_callback(env)
    results = task_metric.gather_results()
    print("TaskMetric results:", results)
    return task_metric


def main(
    random_selection=False,
    headless=False,
    short_exec=False,
    behavior_activity: str = "picking_up_trash",
    activity_definition: int = 0,
):
    """
    Task-aware robot demo. Prints the BDDL-derived action plan, then lets you print/execute it
    or drive manually via keyboard teleop.
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

    # Test resolve_bddl_instance: resolve all task-relevant objects
    mapping, missing = resolve_bddl_instance(env, env.task)
    print(f"[resolve_bddl_instance] Resolved {len(mapping)} objects, missing {len(missing)}")
    for inst, entity in mapping.items():
        print(f"  {inst} -> {getattr(entity, 'name', entity)}")
    if missing:
        print(f"  Missing: {missing}")

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
        lambda: run_task_mode(
            env,
            robot,
            behavior_activity=behavior_activity,
            activity_definition=activity_definition,
            random_selection=random_selection,
            short_exec=short_exec,
        ),
    )

    # Always shut down the environment cleanly at the end
    og.shutdown()


if __name__ == "__main__":
    main()
