# MODIFIED BY Yawen Duan (https://github.com/kmdanielduan)
# moving the whole register process to registration.py

import gym
from dmc2gym.wrappers import DMCWrapper
from dmc2gym.registration import register_suite, register_env

__all__ = ["DMCWrapper", "register_suite", "register_env", "make"]


def make(
        domain_name, 
        task_name,
        **kwargs
    ):
    env_id = 'DMC-%s-%s-v0' % (domain_name, task_name)
    return gym.make(env_id, **kwargs)