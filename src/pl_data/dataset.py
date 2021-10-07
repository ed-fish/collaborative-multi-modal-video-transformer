from typing import Dict, Tuple, Union

import hydra
import numpy as np
import omegaconf
import torch.functional as F
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
from src.common.utils import PROJECT_ROOT


class MyDataset(Dataset):
    def __init__(self, data,
                 seq_len: int,
                 experts: list,
                 name: str,
                 path: str,
                 mixing: DictConfig):
        super().__init__()

        self.data_frame = data
        self.experts = experts
        self.mixing = mixing
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data_frame)

    def collect_labels(self, label):

        target_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                        'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Science Fiction', 'Thriller',  'War']
        label_list = np.zeros(15)

        for i, genre in enumerate(target_names):
            if genre == "Sci-Fi" or genre == "ScienceFiction":
                genre = "Science Fiction"
            if genre in label:
                label_list[i] = 1
        if np.sum(label_list) == 0:
            label_list[5] = 1

        return label_list

    def __getitem__(self, idx) -> Dict:
      # retrieve labels
        label = self.data_frame.at[idx, "label"]
        if len(label) == 2:
            label = self.collect_labels(label[0])
        else:
            label = self.collect_labels(label)
        label = torch.tensor(label).unsqueeze(0)    # Covert label to tensor
        scenes = self.data_frame.at[idx, "scenes"]
        expert_list = []

        # iterate through the scenes for the trailer

        for i, d in enumerate(scenes.values()):
            try:
                if len(expert_list) < self.seq_len:
                    expert_tensor_list = []
                    # if there are multiple experts find out the mixing method
                    if len(self.experts) > 1:
                        for expert in self.experts:
                            t = self.retrieve_tensors(d, expert)
                            # Retrieve the tensors for each expert.
                            expert_tensor_list.append(t)
                        if self.mixing.concat:
                            # concat experts for pre model
                            cat_experts = torch.cat(expert_tensor_list, dim=-1)
                            # expert_list.append(cat_experts)
                            if self.mixing.concat_norm:
                                cat_experts = F.normalize(
                                    cat_experts, p=2, dim=-1)
                            if self.mixing.concat_softmax:
                                cat_experts = F.softmax(cat_experts, dim=-1)
                            expert_list.append(cat_experts)
                        else:
                            expert_list.append(torch.stack(
                                expert_tensor_list))
                    else:
                        # otherwise return one expert
                        expert_list.append(self.retrieve_tensors(
                            d, self.experts[0]))
            except IndexError:
                continue
        if self.mixing.collab:
            while len(expert_list) < self.seq_len:
                pad_list = []
                for i in range(len(self.experts)):
                    pad_list.append(torch.zeros_like(expert_list[0][0]))
                expert_list.append(torch.stack(pad_list))
            if self.mixing.collab:
                expert_list = torch.stack(expert_list)
            expert_list = expert_list.squeeze()
        else:
            while len(expert_list) < self.seq_len:
                expert_list.append(torch.zeros_like(expert_list[0]))

            expert_list = torch.cat(expert_list, dim=0)  # scenes
            expert_list = expert_list.unsqueeze(0)

        return {"label": label, "experts": expert_list}

    def load_tensor(self, tensor) -> torch.Tensor:
        tensor = torch.load(tensor, map_location=torch.device('cpu'))
        return tensor

    def return_expert_path(self, path, expert):
        return path[list(path.keys())[0]][expert]

    def retrieve_tensors(self, path, expert):
        tensor_paths = self.return_expert_path(path, expert)
        if expert == "test-img-embeddings" or expert == "test-location-embeddings":
            tensor_paths = tensor_paths[0]
        t = self.load_tensor(tensor_paths)
        if expert == "audio-embeddings":
            t = t.unsqueeze(0)
        # elif self.mixing.frame_agg == "pool":
        #     pool_list = [self.load_tensor(x) for x in tensor_paths]
        #     pool_list = torch.stack(pool_list, dim=-1)
        #     pool_list = pool_list.unsqueeze(0)
        #     pooled_tensor = F.adaptive_avg_pool2d(
        #         pool_list, (1, self.config["input_shape"]), dim=-1)
        #     t = pooled_tensor.squeeze(0)
        if self.mixing.collab:
            if t.shape[-1] != 2048:
                # zero pad dimensions.
                t = nn.ConstantPad1d((0, 2048 - t.shape[-1]), 0)(t)
        return t

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: MyDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )


if __name__ == "__main__":
    main()
