# An OpenAI Gym like wrapper for the DeepMind Control Suite.

### Instalation
```
git clone git@github.com:1nadequacy/dmc2gym.git
cd dmc2gym
pip install .
```

### Usage
```python
import dmc2gym

env = dmc2gym.make(domain_name='point_mass', task_name='easy', seed=1)

done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```
