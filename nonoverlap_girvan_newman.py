from networkx.algorithms.community import girvan_newman


class GirvanNewman:
    def __init__(self, G):
        self._G = G

    def excuate(self):
        communities_generator = girvan_newman(self._G)
        communities = next(communities_generator)
        return communities
