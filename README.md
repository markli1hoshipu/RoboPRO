# robotwin_bench

Benchmark for Robotwin

git submodule update --init --recursive

Follow installation instructions on [https://robotwin-platform.github.io/doc/usage/robotwin-install.html](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)  
in bash, run source set_env.sh

run visualize_task_scene.py for visualizing scenes

gotta install the additional objects_bench assets somehow. create shelf glb with convert_obj_glb.py file

For new envs:

create the tasks_instructions/{task_name}.json file

Note: Current implementation only works for Aloha-Agilex
    CuRobo obstacles to robot frame
    Attaching external objects to link is only defined in aloha yml