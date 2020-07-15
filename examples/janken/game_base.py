import numpy as np


def battle_once(agent1, agent2, compare):
    actions1 = agent1.actions()
    actions2 = agent2.actions()

    result = 0
    for a1, a2 in zip(actions1, actions2):
        result += compare(a1, a2)
    if result > len(actions1)/2:
        return 1
    elif result < len(actions1)/2:
        return 0
    else:
        return 0.5


def battles(agent1, agent2, compare, n_battles):
    results = [battle_once(agent1, agent2, compare) for _ in range(n_battles)]
    return np.mean(results)
