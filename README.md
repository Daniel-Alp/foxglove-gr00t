# Downloading the Dataset
```
git clone --filter=blob:none --no-checkout https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
git sparse-checkout init --cone
git sparse-checkout set single_panda_gripper.CloseDoubleDoor
git checkout main
```
This script works for any dataset in the [Robot Arm Kitchen Manipulation](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim#robot-arm-kitchen-manipulation-72k-trajectories) folder.
# Converting to MCAP
Clone the repository and `cd` into it
```
git clone https://github.com/Daniel-Alp/foxglove-gr00t
cd foxglove-gr00t
```
Install virtualenv, create a virtual environment, and activate it
```
python3 -m pip install virtualenv
virtualenv venv 
source venv/bin/activate
```
Install requirements
```
python3 -m pip install -r requirements.txt
```
Run the script
```
python3 foxglove-gr00t/main.py data_root chunk episode
```
For example
```
python3 foxglove-gr00t/main.py ~/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CloseDoubleDoor 000 000000
```