import gym
from gym.envs.registration import register


def make(domain_name, task_name, seed=1, visualize_reward=True, from_pixels=False):
    env_id = 'dmc_%s_%s-v1' % (domain_name, task_name)
    
    if from_pixels:
        assert not visualize_reward, 'cannot use visualize reward when learning from pixels'
        
    register(
        id=env_id,
        entry_point='dmc2gym.wrappers:DMCWrapper',
        kwargs={
            'domain_name': domain_name,
            'task_name': task_name,
            'task_kwargs': {
                'random': seed
            },
            'visualize_reward': visualize_reward,
            'from_pixels': from_pixels
        },
        max_episode_steps=1000)
    return gym.make(env_id)
