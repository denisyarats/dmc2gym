# OpenAI Gym wrapper for the DeepMind Control Suite

A lightweight wrapper around the DeepMind Control Suite that provides the standard OpenAI Gym interface. The wrapper allows to specify the following:

* Reliable random seed initialization that will ensure deterministic behaviour.
* Setting ```from_pixels=True``` converts proprioceptive observations into image-based. In additional, you can choose the image dimensions, by setting ```height``` and ```width```.
* Action space normalization bound each action's coordinate into the ```[-1, 1]``` range.
* Setting ```frame_skip``` argument lets to perform action repeat.

## Installation

```shell
pip install git+git://github.com/kmdanielduan/dmc2gym.git
```

## Usage

```python
import gym
from dm_control import suite
from dmc2gym import register_suite, dmc_task2str

register_suite(suite, tag='easy')  # register all tasks with tag 'easy' in the suite to gym registry
env_id = dmc_task2str('point_mass', 'easy')  # convert to "point_mass-easy-v0"
env = gym.make(env_id, task_kwargs=dict(random=42))  # make environments directly

done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```
