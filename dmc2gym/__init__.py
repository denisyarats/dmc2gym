# MODIFIED BY Yawen Duan (https://github.com/kmdanielduan)
# moving the whole register process to registration.py

from dmc2gym.wrappers import DMCWrapper
from dmc2gym.registration import register_suite, register_env

__all__ = ["DMCWrapper", "register_suite", "register_env"]