from typing import Any, Dict, Sequence, Tuple, Union
import math
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from omegaconf import DictConfig
from torch.optim import Optimizer

from src.common.utils import PROJECT_ROOT


class MyModel(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!

        self.save_hyperparameters()

        self.criterion = nn.BCELoss()
        input_size = self.hparams.ninp  # shape of input vector
        token_embedding = self.hparams.token_embedding  # embedding size of head output
        dropout = self.hparams.dropout
        self.bs = self.hparams.batch_size
        n_classes = self.hparams.ntoken  # number of classes (tokens)

        self.pos_encoder = PositionalEncoding(
            input_size // 2, dropout, max_len=self.hparams.seq_len)
        encoder_layers = TransformerEncoderLayer(
            input_size // 2, self.hparams.nhead, self.hparams.nhid, self.hparams.dropout)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, self.hparams.nlayers)
        self.encoder = nn.Linear(input_size, input_size // 2)
        self.decoder = nn.Linear(input_size // 2, token_embedding)
        post_encoder_layer = TransformerEncoderLayer(
            token_embedding, self.hparams.nhead, self.hparams.nhid, self.hparams.dropout)
        self.post_transformer_encoder = TransformerEncoder(
            post_encoder_layer, self.hparams.nlayers)
        self.classifier = nn.Sequential(
            nn.Linear(token_embedding, token_embedding // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(token_embedding // 2, token_embedding // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(token_embedding // 2, n_classes))
        self.classifier_2 = nn.Sequential(
            nn.Linear(token_embedding * self.hparams.seq_len, token_embedding)
        )

        self.cat_classifier = nn.Sequential(
            nn.Linear(token_embedding * len(self.hparams.experts),
                      self.hparams.ntoken)
        )
        self.running_labels = []
        self.running_labels = []
        self.running_logits = []

    def forward(self, src, src_mask):
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        #src = self.encoder(src) * math.sqrt(self.hparams.ninp)
        #src_mask = self.generate_square_subsequent_mask(self.seq_len)
        src = self.encoder(src)
        src = self.pos_encoder(src)
        src_mask = src_mask.to(self.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        output = torch.sigmoid(output)
        # output = F.softmax(output)
        # Do not include softmax if nn.crossentropy as softmax included via NLLoss
        return output

    def post_transformer(self, data):
        data = torch.stack(data)
        data = data.transpose(0, 2)
        collab_array = []
        for x in range(len(self.hparams.experts)):
            d = data[x, :, :, :]
            d = self.step(d)
            collab_array.append(d)
        stacked_array = torch.stack(collab_array)
        stacked_array = self.pos_encoder(stacked_array)
        #stacked_array = stacked_array.transpose(0, 1)
        src_mask = self.generate_square_subsequent_mask(stacked_array.size(0))
        src_mask = src_mask.to(self.device)
        data = self.post_transformer_encoder(stacked_array, src_mask)
        data = data.transpose(0, 1)
        data = data.reshape(self.bs, -1)
        #output = self.decoder(data)

        transform_t = self.cat_classifier(data)
        pooled_result = transform_t.squeeze(0)
        pooled_result = torch.sigmoid(pooled_result)

        return pooled_result

    def format_target(self, target):
        target = torch.cat(target, dim=0)
        target = target.squeeze()
        return target

    def step(self, data):
        # data = torch.cat(data, dim=0)
        # if data.shape[-1] != 2048:
        #     data = self.pad(data)
        # reshape for transformer output (B, S, E) -> (S, B, E)

        data = data.permute(1, 0, 2)
        src_mask = self.generate_square_subsequent_mask(data.size(0))
        src_mask = src_mask.to(self.device)

        # FORWARD
        output = self(data, src_mask)
        ##print("output step 1:", output.shape)

        # reshape back to original (S, B, E) -> (B, S, E)
        transform_t = output.permute(1, 0, 2)

        #print("output_reshape", output.shape)

        # flatten sequence embeddings (S, B, E) -> (B, S * E)
        transform_t = transform_t.reshape(self.bs, -1)

        #print("output_reshape 2", output.shape)
        transform_t = transform_t.unsqueeze(0)

        # Pooling before classification?
        if self.hparams.pooling == "avg":
            transform_t = F.adaptive_avg_pool1d(
                transform_t, self.config["token_embedding"])
            transform_t = transform_t.squeeze(0)
            pooled_result = self.classifier(transform_t)
        elif self.hparams.pooling == "max":
            transform_t = F.adaptive_max_pool1d(
                transform_t, self.config["token_embedding"])
            transform_t = transform_t.squeeze(0)
            pooled_result = self.classifier(transform_t)
        elif self.hparams.pooling == "total":
            transform_t = F.adaptive_max_pool1d(
                transform_t, self.config["ntokens"])
            pooled_result = transform_t.squeeze(0)
        elif self.hparams.pooling == "none":
            #transform_t = transform_t.squeeze()
            transform_t = self.classifier_2(transform_t)
            pooled_result = transform_t.squeeze(0)
            pooled_result = torch.sigmoid(pooled_result)

        return pooled_result

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        data = batch["experts"]
        target = batch["label"]

        data = self.post_transformer(data)

        target = self.format_target(target)
        target = target.float()

        loss = self.criterion(data, target)
        #acc_preds = self.preds_acc(data)

        # gradient clipping for stability
        # torch.nn.utils.clip_grad_norm(self.parameters(), 0.5)

        self.log_dict(
            {"train_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        data = batch["experts"]
        target = batch["label"]

        data = self.post_transformer(data)
        target = self.format_target(target)
        target = target.float()

        loss = self.criterion(data, target)

        self.log_dict(
            {"val_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        data = batch["experts"]
        target = batch["label"]
        data = self.post_transformer(data)
        target = target.float()
        loss = self.criterion(data, target)
        self.log_dict(
            {"test_loss": loss},
        )
        return loss

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return [opt], [scheduler]


class PositionalEncoding(pl.LightningModule):
    def __init__(self, d_model, dropout=0.1, max_len=5):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(1000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.mean = 0.06
        self.std = 0.2

    def forward(self, x):
        # x = (x - self.mean) / self.std
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


@ hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
