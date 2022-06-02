from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
import os
import os.path as osp
import numpy as np
import torch
import pandas as pd

class TaobaoDataset(InMemoryDataset):
    def __init__(self, root: str, preprocess: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        assert self.preprocess in [None, 'metapath2vec', 'transe']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_classes(self) -> int:
        return int(self.data['paper'].y.max()) + 1

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'mag', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'mag', 'processed')

    def download(self):
        pass

    @property
    def processed_file_names(self) -> str:
        if self.preprocess is not None:
            return f'data_{self.preprocess}.pt'
        else:
            return 'data.pt'

    def __repr__(self) -> str:
        return 'taobao'

    def process(self):
        import pandas as pd
        data = HeteroData()

        user_info = pd.read_csv('{}/user_info_format1.csv'.format(root))

        data['user'].x = torch.ByteTensor(user_info[["age_range","gender"]].to_numpy())
        data['user'].id = torch.LongTensor(user_info[["user_id"]].to_numpy()) 

        # 有一些user没有user info

        user_log = pd.read_csv('{}/user_log_format1.csv'.format(root))

        num_users, user_count = numpy.unique(user_log["user_id"].to_numpy(),return_counts=True)
        num_max_user = user_log["user_id"].to_numpy().max()
        num_min_user = user_log["user_id"].to_numpy().min()

        num_sellers = numpy.unique(user_log["seller_id"].to_numpy())
        num_max_sellers = user_log["seller_id"].to_numpy().max()
        num_min_sellers = user_log["seller_id"].to_numpy().min()

        k_user = user_log[['user_id','seller_id','time_stamp']].to_numpy()
        bb = k_user[:,0]*100000000+k_user[:,1]*10000+k_user[:,2]
        _,c=np.unique(bb,return_counts=True)
        print(c.max()) # 6919
        user_feature = torch.zeros(data['user'].id.shape[0],c.max(),dtype=torch.long)-1