# robotwin_bench

New benchmark for Robotwin

## Installation Instructions
```bash
git clone https://github.com/apexs-huawei/robotwin_bench.git
git submodule update --init --recursive
```
Then follow installation instructions on [https://robotwin-platform.github.io/doc/usage/robotwin-install.html](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) 

Copy the benchmark/bench_assets/120_storage-rack folder into  customized_robotwin/assets/objects_bench

Copy all the benchmark/bench_assets/embodiments/aloha-agilex yml files into customized_robotwin/assets/embodiments/aloha-agilex (will overwrite existing files)

## Usage
** switch to benchmark/implementation for both repos
```bash
cd customized_robotwin
```
### Per-Session Commands
Run the following commands every session to set env vars:
```bash
source set_env.sh
export ROBOTWIN_BENCH_TASK="bench" # bench if you want to work with benchmark tasks, anything else if you are working with the original robotwin tasks
```
---
Run commands in the same way you would run them for the original Robotwin. Refer to this [usage guide](https://robotwin-platform.github.io/doc/usage/index.html).

Run the following script to make sure everything is set up properly:
```bash
python script/bench_script/visualize_task_scene.py mouse_on_pad bench_demo_clean --rollout --seed 0
```

## For Creating New Benchmark Tasks
These are required steps to follow in addiion to creating a new task file.
- Add the desired evaluation step limit to _eval_step_lim.yml in benchmark/bench_task_config.yml
- Create a task description template json and put it in benchmark/bench_description/task_instructions.

Refer to [this page](https://robotwin-platform.github.io/doc/usage/description.html) for how to create new task description templates using an LLM API.

Refer to the following alternative instructions if no LLM API is available:

Prompt an LLM directly with the following template with the placeholders filled in with the task-specific info. Refer to any existing task template json in customized_robotwin/description/task_instruction for examples on placeholder values.
```bash
Generate a list of 60 different alternative descriptions for the following: "{general task instruction}". Each description should not be longer than {max words} words

##Generic example: "Set {A} on the {B} mat with {a}."

##Criteria:
1. Use natural, action-oriented verbs like "grab", "slide", "set", "stick", "drop", "place", etc., instead of technical jargon.
2. Vary sentence structures (e.g., questions, commands, requests) and maintain a natural, conversational tone.
4. Avoid adding unnecessary ADJECTIVES or adverbs at the end of sentences!!!!!
5. Clearly or implicitly include all steps of the task in each instruction.

##Schema:
The object schema for you to abstract is "{object schema}"
1. Use placeholders in the format {X} for objects, where X is defined in a schema.
2. Use placeholders in the format {x} for arm placeholders, where x is defined in a schema.
2. Object placeholders ({A-Z}) are included in every instruction, but REFERENCE TO ARMS, INCLUDING arm placeholders ({a-z}) MUST be omitted in 50% of the instructions.
3. Make sure instructions flow naturally when placeholders are replaced with actual objects or arm notations.
```
### Additional Notes for Task Generation
- Run visualize_task_scene.py in customized_robotwin/script/bench_script for visualizing scenes

- Do not name a new task with the same name as any existing task, including the original Robotwin tasks.

Note: Current implementation supports Aloha-Agilex the best in terms of task set up. Attaching external objects to link is only defined in aloha yml. To do this for other embodiments:
- New link in robot yaml: extra_links, collision_link_names, self_collision_ignore
- Sphere placeholders in collision yaml: collision_spheres

