

class MonteCarloFirst:
    def __init__(self, actions):
        self.actions = actions
        self.V = {}

    def update(self, state, G):
        if state not in self.V:
            self.V[state] = 0.5
        self.V[state] = self.V[state] + 0.09 * (G - self.V[state])

