from gym import core, spaces
from dm_control import suite
from dm_control.rl import specs
import numpy as np


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.ArraySpec:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArraySpec:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(core.Env):
    def __init__(self,
                 domain_name,
                 task_name,
                 task_kwargs=None,
                 visualize_reward=True):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward)
        self._action_space = _spec_to_box([self._env.action_spec()])
        self._observation_space = _spec_to_box(
            self._env.observation_spec().values())

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def seed(self):
        # seed is set during the env creation
        pass

    def step(self, action):
        assert self._action_space.contains(action)
        time_step = self._env.step(action)
        obs = _flatten_obs(time_step.observation)
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': time_step.discount}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        return _flatten_obs(time_step.observation)

    def render(self, mode='rgb_array', height=64, width=64, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id)
