import os

import yaml

import omnigibson as og
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitives,
    SymbolicSemanticActionPrimitiveSet,
)
from omnigibson.objects.object_base import REGISTERED_OBJECTS
from omnigibson.scenes.scene_base import Scene
from omnigibson.tasks.task_base import BaseTask
from omnigibson.utils.python_utils import classproperty, create_class_from_registry_and_config


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def grasp_and_release(controller, env, obj, label):
    if obj is None:
        print(f"[skip] {label}: object not found in registry")
        return
    print(f"[grasp] {label}: {obj.name}")
    execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, obj), env)
    execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.RELEASE), env)


class SimpleLoadTask(BaseTask):
    def __init__(self, objects_config=None, **kwargs):
        self._objects_config = objects_config or []
        super().__init__(**kwargs)

    def _load(self, env):
        for obj_config in self._objects_config:
            obj = env.scene.object_registry("name", obj_config["name"])
            if obj is None:
                obj = create_class_from_registry_and_config(
                    cls_name=obj_config["type"],
                    cls_registry=REGISTERED_OBJECTS,
                    cfg=obj_config,
                    cls_type_descriptor="object",
                )
                env.scene.add_object(obj)
            obj_pos = obj_config.get("position", [0.0, 0.0, 0.0])
            obj_orn = obj_config.get("orientation", [0.0, 0.0, 0.0, 1.0])
            obj.set_position_orientation(position=obj_pos, orientation=obj_orn, frame="scene")

    def _create_termination_conditions(self):
        return {}

    def _create_reward_functions(self):
        return {}

    def _load_non_low_dim_observation_space(self):
        return {}

    def _get_obs(self, env):
        return {}, {}

    @classproperty
    def valid_scene_types(cls):
        return {Scene}

    @classproperty
    def default_termination_config(cls):
        return {}

    @classproperty
    def default_reward_config(cls):
        return {}


def main():
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet"]

    # 1) Scene-level object (added via config["objects"]).
    scene_obj_cfg = {
        "type": "DatasetObject",
        "name": "scene_apple",
        "category": "apple",
        "model": "agveuv",
        "position": [1.2, 0.0, 0.75],
        "orientation": [0, 0, 0, 1],
    }
    config["objects"] = [scene_obj_cfg]

    # 2) Task-level object (created inside SimpleLoadTask._load).
    task_obj_cfg = {
        "type": "DatasetObject",
        "name": "task_cologne",
        "category": "bottle_of_cologne",
        "model": "lyipur",
        "position": [0.6, -0.2, 0.75],
        "orientation": [0, 0, 0, 1],
    }
    config["task"] = {
        "type": "SimpleLoadTask",
        "objects_config": [task_obj_cfg],
    }

    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # 3) Manual object (added after env is created).
    manual_obj_cfg = {
        "type": "DatasetObject",
        "name": "manual_cologne",
        "category": "bottle_of_cologne",
        "model": "lyipur",
        "position": [0.9, 0.2, 0.75],
        "orientation": [0, 0, 0, 1],
    }
    manual_obj = create_class_from_registry_and_config(
        cls_name=manual_obj_cfg["type"],
        cls_registry=REGISTERED_OBJECTS,
        cfg=manual_obj_cfg,
        cls_type_descriptor="object",
    )
    scene.add_object(manual_obj)
    manual_obj.set_position_orientation(
        position=manual_obj_cfg["position"],
        orientation=manual_obj_cfg["orientation"],
        frame="scene",
    )

    for _ in range(30):
        og.sim.step()

    controller = SymbolicSemanticActionPrimitives(env, robot)

    grasp_and_release(controller, env, scene.object_registry("name", scene_obj_cfg["name"]), "scene object")
    grasp_and_release(controller, env, scene.object_registry("name", task_obj_cfg["name"]), "task object")
    grasp_and_release(controller, env, scene.object_registry("name", manual_obj_cfg["name"]), "manual object")

    for _ in range(200):
        og.sim.step()

    og.shutdown()


if __name__ == "__main__":
    main()
