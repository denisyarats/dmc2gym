"""Helper functions for dmc2gym wrappers"""


def dmc_task2str(domain_name, task_name):
    """Convert domain_name and task_name to a string suitable for environment_kwargs"""
    return "%s-%s-v0" % (domain_name, task_name)
