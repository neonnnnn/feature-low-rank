import numpy as np
from sklearn.utils import check_random_state


class Agent(object):
    def __init__(self, n_cards=1, random_state=None):
        self.n_cards = n_cards
        self.random_state =  check_random_state(random_state)

    def create_cards(self):
        self.cards_ = self.random_state.randint(0, 3, size=self.n_cards)
        unique, counts = np.unique(self.cards_, return_counts=True)
        features = np.zeros(3)
        features[unique] = counts
        self.feature_vector_ = features
        self.id_ = np.dot(np.array([self.n_cards]) ** np.arange(3)[::-1], self.feature_vector_)

    def actions(self):
        self.random_state.shuffle(self.cards_)
        return self.cards_
