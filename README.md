# robotwin_bench

Benchmark for Robotwin

git submodule update --init --recursive

Follow installation instructions on [https://robotwin-platform.github.io/doc/usage/robotwin-install.html](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)  
in bash, run source set_env.sh

run visualize_task_scene.py for visualizing scenes

gotta install the additional objects_bench assets somehow. gotta get the updated camera config files, config.yml

For new envs:

create the tasks_instructions/{task_name}.json file using gen_task_instruction_templates.sh
can't name the same as robotwin name env

Note: Current implementation only works for Aloha-Agilex
    CuRobo obstacles to robot frame
    Attaching external objects to link is only defined in aloha yml


for task description json generation:
Generate a list of 60 different alternative descriptions for the following: "pick up the phone and put it on the phone stand". Each description should not be longer than 8 words

##Generic example: "Set {A} on the {B} mat with {a}."

##Criteria:
1. Use natural, action-oriented verbs like "grab", "slide", "set", "stick", "drop", "place", etc., instead of technical jargon.
2. Vary sentence structures (e.g., questions, commands, requests) and maintain a natural, conversational tone.
4. Avoid adding unnecessary ADJECTIVES or adverbs at the end of sentences!!!!!
5. Clearly or implicitly include all steps of the task in each instruction.

##Schema:
The object schema for you to abstract is "{A} notifies the phone, {B} notifies the phonestand. Arm use literal 'arm'"
1. Use placeholders in the format {X} for objects, where X is defined in a schema.
2. Use placeholders in the format {x} for arm placeholders, where x is defined in a schema.
2. Object placeholders ({A-Z}) are included in every instruction, but REFERENCE TO ARMS, INCLUDING arm placeholders ({a-z}) MUST be omitted in 50% of the instructions.
3. Make sure instructions flow naturally when placeholders are replaced with actual objects or arm notations.
