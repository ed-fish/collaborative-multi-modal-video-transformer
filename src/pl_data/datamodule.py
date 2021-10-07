import random
from typing import Optional, Sequence

import hydra
import pandas as pd
import pickle
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.common.utils import PROJECT_ROOT


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        mixing: DictConfig,

    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.mixing = mixing

        # self.train_dataset: Optional[Dataset] = None
        # self.val_datasets: Optional[Dataset] = None
        # self.test_datasets: Optional[Dataset] = None

    def prepare_data(self) -> None:
        # download only
        pass

    def clean_data(self, data_frame):
        target_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                        'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Science Fiction', 'Thriller',  'War']

        print("cleaning data")
        print(data_frame.describe())

        longest_seq = 0
        for i in range(len(data_frame)):
            data = data_frame.at[i, "scenes"]
            label = data_frame.at[i, "label"]
            n_labels = 0
            for l in label[0]:
                if l not in target_names:
                    n_labels += 1
            if n_labels == 6:
                data_frame = data_frame.drop(i)
                continue
            data_chunk = list(data.values())
            if len(data_chunk) > longest_seq:
                longest_seq = len(data_chunk)
            if len(data_chunk) < 5:
                data_frame = data_frame.drop(i)
                continue
            if not self.find_keys(data_chunk, self.datasets.train[0]["experts"]):
                print("removing index", i)
                data_frame = data_frame.drop(i)
                continue

        data_frame = data_frame.reset_index(drop=True)
        return data_frame

    def find_keys(self, data_chunk, experts):
        for p in range(len(data_chunk)):
            for e in experts:
                for k, v in data_chunk[p].items():
                    if e not in list(v.keys()):
                        print(e)
                        print(list(v.keys()))
                        return False
        return True

    def load_data(self, db):
        data = []
        with open(db, "rb") as pkly:
            while 1:
                try:
                    # append if data serialised with open file
                    data.append(pickle.load(pkly))
                    # else data not streamed
                    # data = pickle.load(pkly)
                except EOFError:
                    break

        data_frame = pd.DataFrame(data)
        print("data loaded")
        print("length", len(data_frame))
        # data_frame = data_frame.head(1000)
        return data_frame

    def setup(self, stage: Optional[str] = None):
        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if stage is None or stage == "fit":

            self.train_data = self.load_data(self.datasets.train[0]["path"])
            self.train_data = self.clean_data(self.train_data)
            self.val_data = self.load_data(self.datasets.val[0]["path"])
            self.val_data = self.clean_data(self.val_data)

            self.train_dataset = hydra.utils.instantiate(
                self.datasets.train[0], self.train_data, mixing=self.mixing)
            self.val_dataset = hydra.utils.instantiate(
                self.datasets.val[0], self.val_data, mixing=self.mixing)

        if stage is None or stage == "test":

            self.test_data = self.load_data(self.datasets.val[0]["path"])
            self.test_data = self.clean_data(self.test_data)
            self.test_dataset = hydra.utils.instantiate(
                self.datasets.test[0], self.test_data, mixing=self.mixing)

    def custom_collater(self, batch):
        return {
            'label': [x['label'] for x in batch],
            'experts': [x['experts'] for x in batch]
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
            collate_fn=self.custom_collater,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            worker_init_fn=worker_init_fn,
            collate_fn=self.custom_collater,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.test,
            num_workers=self.num_workers.test,
            collate_fn=self.custom_collater,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


@ hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )


if __name__ == "__main__":
    main()
