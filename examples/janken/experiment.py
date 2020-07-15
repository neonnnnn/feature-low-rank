from agent_jyanken import Agent
from rule_janken import compare, map_num_to_hand
from game_base import battle_once, battles
import numpy as np
import argparse
from scipy.special import comb
from sklearn.datasets import dump_svmlight_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Janken Cards Battle Experiments"
    )
    parser.add_argument("--n-cards", type=int, default=20)
    parser.add_argument("--n-agents", type=int, default=200)
    parser.add_argument("--n-battles", type=int, default=100)
    parser.add_argument("--rng", type=int, default=0)
    args = parser.parse_args()
    n_cards = args.n_cards
    n_agents = args.n_agents
    n_battles = args.n_battles
    rng = np.random.RandomState(args.rng)
    if comb(n_cards+3-1, n_cards) < n_agents:
        raise ValueError("Too many agents (bigger than the number of"
                         "all combinations of cards).")
    n = 0
    agents = []
    agents_id = []
    while n < n_agents:
        agent = Agent(n_cards=n_cards, random_state=rng)
        agent.create_cards()
        if agent.id_ not in agents_id:
            n += 1
            agents.append(agent)
            agents_id.append(agent.id_)
    X = []
    y = []
    X_id = []
    for a in range(n_agents):
        for b in range(a+1, n_agents):
            win_rate = battles(agents[a], agents[b], compare, n_battles)
            a_vec = agents[a].feature_vector_
            b_vec = agents[b].feature_vector_
            X.append(np.append(a_vec, b_vec))
            y.append(win_rate)
            X_id.append(np.array([agents_id[a], agents_id[b]]))
            print(len(y))
    X = np.vstack(X)
    X_id = np.vstack(X_id)
    y = np.array(y)
    print(X.shape, X_id.shape, y.shape)
    filename = "janken_"+str(n_cards)+"_"+str(n_agents)+"_"+str(n_battles)
    # feature (bag-of-cards) encoding
    dump_svmlight_file(X, y, filename+'.svm')
    # one-hot encoding
    dump_svmlight_file(X_id, y, filename+'_id.svm')