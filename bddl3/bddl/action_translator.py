from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple, Type

from bddl.activity import (
    Conditions,
    get_goal_conditions,
    get_ground_goal_state_options,
    get_object_scope,
)
from bddl.trivial_backend import TrivialBackend
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitiveSet,
)

class AtomicAction(Enum):
    GRASP = auto()
    PLACE_ON_TOP = auto()
    PLACE_INSIDE = auto()
    OPEN = auto()
    CLOSE = auto()


ACTION_ENUM_DEFAULT: Type[Enum] = SymbolicSemanticActionPrimitiveSet


@dataclass(frozen=True)
class ActionStep:
    """
    A single robot action derived from a BDDL predicate.
    """

    primitive: Enum
    primary_object: str
    reference_object: Optional[str] = None
    source_predicate: str = ""


@dataclass
class ActionPlan:
    """
    Sequence of robot actions for one grounded goal option.
    """

    steps: List[ActionStep]
    unsupported_conditions: List[str]

    def as_primitive_sequence(self) -> List[Tuple[Enum, str, Optional[str]]]:
        """
        Returns the minimal tuple representation that downstream executors can consume directly.
        """

        return [
            (step.primitive, step.primary_object, step.reference_object)
            for step in self.steps
        ]

    def to_strings(self) -> List[str]:
        """
        Human-readable lines for logging or debugging.
        """
        lines = []
        for step in self.steps:
            target = f" -> {step.reference_object}" if step.reference_object else ""
            lines.append(f"{step.primitive.name}: {step.primary_object}{target}")
        if self.unsupported_conditions:
            lines.append(f"Unsupported: {', '.join(self.unsupported_conditions)}")
        return lines


class BDDLActionTranslator:
    """
    Translates BDDL goal conditions into executable high-level robot actions.

    Example:
        translator = BDDLActionTranslator()
        plans = translator.translate("assembling_gift_baskets", activity_definition=0)
        # plans[0].steps -> [ActionStep(GRASP, "candle.n.01_1"), ActionStep(PLACE_INSIDE, ...), ...]
    """

    def __init__(
        self,
        simulator_name: str = "omnigibson",
        backend=None,
        action_enum: Type[Enum] = ACTION_ENUM_DEFAULT,
    ):
        self.simulator_name = simulator_name
        self.backend = backend or TrivialBackend()
        self.action_enum = action_enum

    def translate(
        self,
        behavior_activity: Optional[str] = None,
        activity_definition: int = 0,
        predefined_problem: Optional[str] = None,
    ) -> List[ActionPlan]:
        """
        Load a BDDL problem and return action plans for each grounded goal option.
        """

        conds = self._load_conditions(
            behavior_activity=behavior_activity,
            activity_definition=activity_definition,
            predefined_problem=predefined_problem,
        )
        scope = get_object_scope(conds)
        goal_conditions = get_goal_conditions(conds, self.backend, scope, generate_ground_options=True)
        if not goal_conditions:
            return []

        goal_options = get_ground_goal_state_options(
            conds, self.backend, scope, goal_conditions
        )

        plans: List[ActionPlan] = []
        for option in goal_options:
            steps, unsupported = self._state_to_steps(option)
            plans.append(ActionPlan(steps=self._merge_redundant_steps(steps), unsupported_conditions=unsupported))
        return plans

    def _load_conditions(
        self,
        behavior_activity: Optional[str],
        activity_definition: int,
        predefined_problem: Optional[str],
    ) -> Conditions:
        if behavior_activity is None and predefined_problem is None:
            raise ValueError("Either behavior_activity or predefined_problem must be provided.")

        activity_name = behavior_activity or "custom_activity"
        return Conditions(
            behavior_activity=activity_name,
            activity_definition=activity_definition,
            simulator_name=self.simulator_name,
            predefined_problem=predefined_problem,
        )

    def _state_to_steps(self, compiled_state) -> Tuple[List[ActionStep], List[str]]:
        """
        Convert a compiled goal option (list of HEAD nodes) into robot action steps.
        """

        steps: List[ActionStep] = []
        unsupported: List[str] = []

        for head in compiled_state:
            negated, predicate, args = self._head_to_literal(head)
            new_steps = self._predicate_to_actions(predicate, args, negated)
            if new_steps:
                steps.extend(new_steps)
            else:
                unsupported.append(self._format_literal(negated, predicate, args))

        return steps, unsupported

    def _head_to_literal(self, head) -> Tuple[bool, str, Sequence[str]]:
        """
        Extract a simple (negated, predicate, args) tuple from a compiled HEAD node.
        """

        body = head.body
        if not body:
            raise ValueError("Encountered empty BDDL condition while translating to actions.")

        if body[0] == "not":
            predicate = body[1][0]
            args = body[1][1:]
            return True, predicate, args

        predicate = body[0]
        args = body[1:]
        return False, predicate, args

    def _predicate_to_actions(
        self, predicate: str, args: Sequence[str], negated: bool
    ) -> Optional[List[ActionStep]]:
        """
        Map a single grounded predicate into one or more robot actions.
        """

        if predicate == "ontop" and not negated and len(args) == 2:
            obj, support = args
            return [
                ActionStep(self.action_enum.GRASP, obj, source_predicate="ontop"),
                ActionStep(
                    self.action_enum.PLACE_ON_TOP,
                    obj,
                    reference_object=support,
                    source_predicate="ontop",
                ),
            ]

        if predicate == "inside" and not negated and len(args) == 2:
            obj, container = args
            return [
                ActionStep(self.action_enum.GRASP, obj, source_predicate="inside"),
                ActionStep(
                    self.action_enum.PLACE_INSIDE,
                    obj,
                    reference_object=container,
                    source_predicate="inside",
                ),
            ]

        if predicate == "contains" and not negated and len(args) == 2:
            container, obj = args
            return [
                ActionStep(self.action_enum.GRASP, obj, source_predicate="contains"),
                ActionStep(
                    self.action_enum.PLACE_INSIDE,
                    obj,
                    reference_object=container,
                    source_predicate="contains",
                ),
            ]

        if predicate == "open" and len(args) == 1:
            obj = args[0]
            primitive = (
                self.action_enum.CLOSE
                if negated
                else self.action_enum.OPEN
            )
            return [
                ActionStep(
                    primitive,
                    obj,
                    source_predicate="not open" if negated else "open",
                )
            ]

        if predicate == "closed" and len(args) == 1:
            obj = args[0]
            return [
                ActionStep(self.action_enum.CLOSE, obj, source_predicate="closed")
            ]

        if predicate == "grasped" and not negated and len(args) >= 1:
            obj = args[0]
            return [ActionStep(self.action_enum.GRASP, obj, source_predicate="grasped")]

        return None

    def _merge_redundant_steps(self, steps: List[ActionStep]) -> List[ActionStep]:
        """
        Remove obvious duplicates while preserving order.
        """

        seen = set()
        merged: List[ActionStep] = []
        for step in steps:
            key = (step.primitive, step.primary_object, step.reference_object)
            if key in seen and step.primitive in {
                self.action_enum.OPEN,
                self.action_enum.CLOSE,
            }:
                continue
            seen.add(key)
            merged.append(step)
        return merged

    def _format_literal(self, negated: bool, predicate: str, args: Sequence[str]) -> str:
        prefix = "not " if negated else ""
        return f"{prefix}{predicate}({', '.join(args)})"


def _demo_picking_up_trash():
    """
    Lightweight smoke test for the picking_up_trash task. Prints parsed sections and
    generates action plans using the built-in atomic actions.
    """

    behavior_activity = "picking_up_trash"
    activity_definition = 0
    simulator = "omnigibson"

    conds = Conditions(behavior_activity, activity_definition, simulator)
    print("Activity:", conds.behavior_activity)
    print("Objects:", conds.parsed_objects)
    print("Init conditions:", conds.parsed_initial_conditions)
    print("Goal conditions:", conds.parsed_goal_conditions)

    translator = BDDLActionTranslator(
        simulator_name=simulator, backend=TrivialBackend(), action_enum=AtomicAction
    )
    plans = translator.translate(
        behavior_activity=behavior_activity, activity_definition=activity_definition
    )
    print(f"Grounded goal options: {len(plans)}")
    if plans:
        first = plans[0]
        print(f"First plan steps ({len(first.steps)}):")
        for step in first.steps:
            print(" ", step)
        if first.unsupported_conditions:
            print("Unsupported predicates:", first.unsupported_conditions)


if __name__ == "__main__":
    _demo_picking_up_trash()
