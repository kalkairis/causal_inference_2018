import pandas as pd

from src.config import BaseConfig
from os.path import join


def join_data_frames():
    metaphlan_abundances = pd.read_pickle(join(BaseConfig.data_dir, 'MPA.dat'))
    metaphlan_abundances.transpose(inplace=True)
    metaphlan_abundances['FD'] = tmp['FD'] = list(map(lambda x: x.split('_')[0], metaphlan_abundances.index.values))

    stool_metaphlan_df = pd.read_pickle(join(BaseConfig.data_dir, 'StoolMetadataDF.dat'))
    connections_df = pd.read_pickle(join(BaseConfig.data_dir, 'ConnectionsDF.dat'))
    print('bla')


if __name__ == "__main__":
    join_data_frames()
