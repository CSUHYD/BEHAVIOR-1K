
import numpy as np
from tqdm import tqdm
import benchmark
import utils
from openai_cache import Completion
import openai


def construct_summarization_prompt(objects, receptacles, placements):
    #功能：生成用于让 LLM 总结整理规则的 prompt。
    summarization_prompt_template = '''objects = ["dried figs", "protein bar", "cornmeal", "Macadamia nuts", "vinegar", "herbal tea", "peanut oil", "chocolate bar", "bread crumbs", "Folgers instant coffee"]
receptacles = ["top rack", "middle rack", "table", "shelf", "plastic box"]
pick_and_place("dried figs", "plastic box")
pick_and_place("protein bar", "shelf")
pick_and_place("cornmeal", "top rack")
pick_and_place("Macadamia nuts", "plastic box")
pick_and_place("vinegar", "middle rack")
pick_and_place("herbal tea", "table")
pick_and_place("peanut oil", "middle rack")
pick_and_place("chocolate bar", "shelf")
pick_and_place("bread crumbs", "top rack")
pick_and_place("Folgers instant coffee", "table")
# Summary: Put dry ingredients on the top rack, liquid ingredients in the middle rack, tea and coffee on the table, packaged snacks on the shelf, and dried fruits and nuts in the plastic box.

objects = ["yoga pants", "wool sweater", "black jeans", "Nike shorts"]
receptacles = ["hamper", "bed"]
pick_and_place("yoga pants", "hamper")
pick_and_place("wool sweater", "bed")
pick_and_place("black jeans", "bed")
pick_and_place("Nike shorts", "hamper")
# Summary: Put athletic clothes in the hamper and other clothes on the bed.

objects = ["Nike sweatpants", "sweater", "cargo shorts", "iPhone", "dictionary", "tablet", "Under Armour t-shirt", "physics homework"]
receptacles = ["backpack", "closet", "desk", "nightstand"]
pick_and_place("Nike sweatpants", "backpack")
pick_and_place("sweater", "closet")
pick_and_place("cargo shorts", "closet")
pick_and_place("iPhone", "nightstand")
pick_and_place("dictionary", "desk")
pick_and_place("tablet", "nightstand")
pick_and_place("Under Armour t-shirt", "backpack")
pick_and_place("physics homework", "desk")
# Summary: Put workout clothes in the backpack, other clothes in the closet, books and homeworks on the desk, and electronics on the nightstand.

objects = {objects_str}
receptacles = {receptacles_str}
{placements_str}
# Summary:'''
    objects_str = '[' + ', '.join(map(lambda x: f'"{x}"', objects)) + ']'
    receptacles_str = '[' + ', '.join(map(lambda x: f'"{x}"', receptacles)) + ']'
    placements_str = '\n'.join(map(lambda x: f'pick_and_place("{x[0]}", "{x[1]}")', placements))
    return summarization_prompt_template.format(objects_str=objects_str, receptacles_str=receptacles_str, placements_str=placements_str)

def construct_placement_prompt(summary, objects, receptacles):
    #功能：生成用于让 LLM 根据总结预测物体放置的 prompt。
    placement_prompt_template = '''# Summary: Put clothes in the laundry basket and toys in the storage box.
objects = ["socks", "toy car", "shirt", "Lego brick"]
receptacles = ["laundry basket", "storage box"]
pick_and_place("socks", "laundry basket")
pick_and_place("toy car", "storage box")
pick_and_place("shirt", "laundry basket")
pick_and_place("Lego brick", "storage box")

# Summary: {summary}
objects = {objects_str}
receptacles = {receptacles_str}
pick_and_place("{first_object}",'''
    objects_str = '[' + ', '.join(map(lambda x: f'"{x}"', objects)) + ']'
    receptacles_str = '[' + ', '.join(map(lambda x: f'"{x}"', receptacles)) + ']'
    return placement_prompt_template.format(summary=summary, objects_str=objects_str, receptacles_str=receptacles_str, first_object=objects[0])


def evaluate(scenarios, eval_split='unseen', model='gpt-4o', verbose=False):
    #功能：评估 LLM 总结整理规则和根据总结预测物体放置的能力。
    assert eval_split in {'unseen', 'seen'}
    completion = Completion()
    accuracies = []
    for i, scenario in enumerate(scenarios):
        if verbose:
            print(f'Scenario {i + 1} of {len(scenarios)}\n')

        # Summarization
        summarization_prompt = construct_summarization_prompt(
            scenario.seen_objects, scenario.receptacles, scenario.seen_placements)
        summarization_completion = completion.create(summarization_prompt, model=model).choices[0].message.content
        if verbose:
            print(summarization_prompt, end='')
            utils.print_colored(summarization_completion, 'blue')
            print('\n' + 10 * '-' + '\n')

        # Object placement
        summary = benchmark.parse_summary(summarization_completion)
        objects = scenario.seen_objects if eval_split == 'seen' else scenario.unseen_objects
        placement_prompt = construct_placement_prompt(summary, objects, scenario.receptacles)
        placement_completion = completion.create(placement_prompt, model=model).choices[0].message.content
        if verbose:
            print(placement_prompt, end='')
            utils.print_colored(placement_completion, 'blue')
            print('\n' + 10 * '-' + '\n')

        # Analysis
        predicted_placements = benchmark.parse_placements(placement_completion, objects)
        correct_placements = scenario.seen_placements if eval_split == 'seen' else scenario.unseen_placements
        corrects, accuracy = benchmark.check_placements(predicted_placements, correct_placements)
        accuracies.append(accuracy)
        if verbose:
            print(f'Annotator notes: {scenario.annotator_notes}\n')
            print('Correct placements:')
            for placement in correct_placements:
                print(placement)
            print('\nParsed placements:')
            for placement, correct in zip(predicted_placements, corrects):
                utils.print_colored(placement, 'green' if correct else 'red')
            print(f'\nAccuracy: {accuracy:.2f}')
            print('\n' + 80 * '-' + '\n')
    return accuracies

if __name__ == '__main__':
    scenarios = benchmark.load_scenarios()
    accuracies1 = evaluate(scenarios, eval_split='unseen', verbose=True)
    accuracies2 = evaluate(scenarios, eval_split='seen', verbose=True)
    print(f'Unseen accuracy: {np.mean(accuracies1):.2f}')
    print(f'Seen accuracy: {np.mean(accuracies2):.2f}')
    print(f'Total accuracy: {np.mean(accuracies1 + accuracies2):.2f}')
