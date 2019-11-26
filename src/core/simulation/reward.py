import numpy as np

# Reward functions based on the change_in_cash from one state to the next state


def RewardId(change_in_cash, scale=1):
    """Reward awarded is identical to the change in cash"""
    return change_in_cash * scale

