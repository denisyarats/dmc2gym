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
                 visualize_reward=True,
                 from_pixels=False,
                 height=84,
                 width=84,
                 camera_id=0):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward)
        self._action_space = _spec_to_box([self._env.action_spec()])
        if from_pixels:
            self._observation_space = spaces.Box(
                low=0, high=1, shape=[3, height, width], dtype=np.float32)
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values())

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id)
            obs = obs.transpose(2, 0, 1) / 255.
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def seed(self, seed):
        self._action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._action_space.contains(action)
        time_step = self._env.step(action)
        obs = self._get_obs(time_step)
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': time_step.discount}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=64, width=64, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id)
