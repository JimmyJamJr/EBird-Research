class RankGrid:
    eps_list: list
    teps_list: list

    def __init__(self, eps_list: list, teps_list: list):
        self.eps_list = eps_list
        self.teps_list = teps_list

    def generate_using_dbscan(self, save_to_file=True):
        pass