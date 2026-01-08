import os

import yaml

import omnigibson as og
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitives,
    SymbolicSemanticActionPrimitiveSet,
)


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def main():
    """
    Same as solve_simple_task.py but uses SymbolicSemanticActionPrimitives to teleport objects.
    """
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]
    config["objects"] = [
        {
            "type": "DatasetObject",
            "name": "cologne",
            "category": "bottle_of_cologne",
            "model": "lyipur",
            "position": [-0.3, -0.8, 0.5],
            "orientation": [0, 0, 0, 1],
        },
        {
            "type": "DatasetObject",
            "name": "table",
            "category": "breakfast_table",
            "model": "rjgmmy",
            "scale": [0.3, 0.3, 0.3],
            "position": [-0.7, 0.5, 0.2],
            "orientation": [0, 0, 0, 1],
        },
    ]

    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Let objects settle a bit
    for _ in range(30):
        og.sim.step()

    og.sim.enable_viewer_camera_teleoperation()

    controller = SymbolicSemanticActionPrimitives(env, robot)

    grasp_obj = scene.object_registry("name", "cologne")
    execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, grasp_obj), env)

    table = scene.object_registry("name", "table")
    execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP, table), env)

    for _ in range(10000):
        og.sim.step()

    # Always close the environment at the end
    og.shutdown()
    
if __name__ == "__main__":
    main()
