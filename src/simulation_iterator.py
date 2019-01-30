import pandas as pd


class SimulationIterator:
    def __init__(self, files_list, test=False):
        self.files = sorted(files_list)
        self.test = test
        if test:
            self.cf_files = [f for f in self.files if f.endswith('_cf.csv')]
            self.files = [f for f in self.files if not f.endswith('_cf.csv')]

    def __iter__(self):
        self.curr = 0
        return self

    @property
    def df(self):
        print('curr:', self.curr)
        print('file:', self.files[self.curr])
        return pd.read_csv(self.files[self.curr])

    @property
    def cf(self):
        if self.test:
            return pd.read_csv(self.cf_files[self.curr])

    def __next__(self):
        self.curr += 1
        if self.curr == len(self.files):
            raise StopIteration
