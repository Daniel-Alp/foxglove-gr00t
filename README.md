# Downloading the Dataset
```
git clone --filter=blob:none --no-checkout https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
git sparse-checkout init --cone
git sparse-checkout set single_panda_gripper.CloseDoubleDoor
git checkout main
```
This script works for any dataset in the [Robot Arm Kitchen Manipulation](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim#robot-arm-kitchen-manipulation-72k-trajectories) folder.
# Converting State to MCAP
Clone this repository and `cd` into it
```
git clone https://github.com/Daniel-Alp/foxglove-gr00t
cd foxglove-gr00t
```
Install virtualenv, create a virtual environment, and activate it </br>
on MacOS
```
python3 -m pip install virtualenv
```
on Linux
```
sudo apt install python3-virtualenv
```
```
virtualenv venv 
source venv/bin/activate
```
Install requirements
```
python3 -m pip install -r requirements.txt
```
example of converting state
```
python3 foxglove-gr00t/state.py ~/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CloseDoubleDoor 000 000000
```
# Converting Camera to MCAP
In addition to the dependencies in requriements.txt to convert camera to MCAP install ffmpeg
</br>
on MacOS
```
brew install pkg-config ffmpeg
```
on Linux
```
sudo apt-get update
sudo apt-get install -qq --no-install-recommends \
  pkg-config \
  ffmpeg \
  libavutil-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libswresample-dev \
  libavfilter-dev \
  libavdevice-dev
```
example of converting camera
```
python3 foxglove-gr00t/camera.py ~/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CloseDoubleDoor 000 000000
```