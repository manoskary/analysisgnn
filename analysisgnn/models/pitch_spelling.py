import torch.nn as nn
import numpy as np
import torch_geometric.nn as gnn
from pytorch_lightning import LightningModule
import torch
from torch.optim import Adam
from graphmuse.nn.models import MetricalGNN
from torchmetrics import Accuracy, F1Score
from analysisgnn.utils.music import PitchEncoder, KeySignatureEncoder
from torch_geometric.utils import trim_to_layer


class HierarchicalHeteroGraphSage(torch.nn.Module):
    def __init__(self, edge_types, input_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        conv = gnn.HeteroConv(
            {
                edge_type: gnn.SAGEConv(input_channels, hidden_channels)
                for edge_type in edge_types
            }, aggr='sum')
        self.convs.append(conv)
        for _ in range(num_layers-1):
            conv = gnn.HeteroConv(
                {
                    edge_type: gnn.SAGEConv(hidden_channels, hidden_channels)
                    for edge_type in edge_types
                }, aggr='sum')
            self.convs.append(conv)

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, num_sampled_edges_dict,
                num_sampled_nodes_dict):
        for i, conv in enumerate(self.convs):
            if num_sampled_edges_dict is not None:
                x_dict, edge_index_dict, _ = trim_to_layer(
                    layer=i,
                    num_sampled_nodes_per_hop=num_sampled_nodes_dict,
                    num_sampled_edges_per_hop=num_sampled_edges_dict,
                    x=x_dict,
                    edge_index=edge_index_dict,
                )
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict["note"])


class PKSpell(nn.Module):
    """
    PKSpell state of the art algorithm for pitch spelling ISMIR 2021.
    PKSPell is based on an RNN architecture.

    CITE:
    ------
    PKSpell: Data-Driven Pitch Spelling and Key Signature Estimation
    Francesco Foscarin (CNAM), Nicolas Audebert, Raphaël Fournier-S'Niehotta
    """
    def __init__(self, in_feats, n_hidden1, n_hidden2, out_feats, rnn_depth=1,
                 dropout=0.5, cell_type="GRU", bidirectional=True, mode="both"):
        super(PKSpell, self).__init__()

        if n_hidden1 % 2 != 0:
            raise ValueError("Hidden_dim must be an even integer")
        if n_hidden2 % 2 != 0:
            raise ValueError("Hidden_dim2 must be an even integer")
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2

        if cell_type == "GRU":
            rnn_cell = nn.GRU
        elif cell_type == "LSTM":
            rnn_cell = nn.LSTM
        else:
            raise ValueError(f"Unknown RNN cell type: {cell_type}")

        # RNN layer.
        self.rnn = rnn_cell(
            input_size=in_feats,
            hidden_size=n_hidden1 // 2 if bidirectional else n_hidden1,
            bidirectional=bidirectional,
            num_layers=rnn_depth,
            batch_first=True,
        )
        self.rnn2 = rnn_cell(
            input_size=n_hidden1,
            hidden_size=n_hidden2 // 2 if bidirectional else n_hidden2,
            bidirectional=bidirectional,
            num_layers=rnn_depth,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        # Output layers.
        self.top_layer_pitch = nn.Linear(n_hidden1, out_feats)
        self.top_layer_ks = nn.Linear(n_hidden2, 15)

        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(reduction="mean", ignore_index=out_feats)
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=15)
        self.mode = mode

    def forward(self, sentences, sentences_len):
        sentences = torch.split(sentences, sentences_len)
        sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True)
        rnn_out, _ = self.rnn(sentences)
        rnn_out = self.dropout(rnn_out)
        out_pitch = self.top_layer_pitch(rnn_out)

        # pass the ks information into the second rnn
        rnn_out, _ = self.rnn2(rnn_out)
        rnn_out = self.dropout(rnn_out)
        out_ks = self.top_layer_ks(rnn_out)

        return out_pitch, out_ks

    def predict(self, sentences, sentences_len):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.forward(sentences, sentences_len)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted_pitch = scores_pitch.argmax(dim=2)
        predicted_ks = scores_ks.argmax(dim=2)
        return (
            [
                predicted_pitch[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
            [
                predicted_ks[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
        )

    def loss_computation(self, pred_pitch, pred_ks, true_pitch, true_ks):
        # Flatten the outputs and the gold-standard labels, to compute the loss.
        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
        scores_pitch = pred_pitch.view(-1, self.n_out_pitch)
        scores_ks = pred_ks.view(-1, self.n_out_ks)
        pitches = true_pitch.view(-1)
        keysignatures = true_ks.view(-1)
        if self.mode == "both":
            loss = self.loss_pitch(scores_pitch, pitches) + self.loss_ks(
                scores_ks, keysignatures
            )
        elif self.mode == "ks":
            loss = self.loss_ks(scores_ks, keysignatures)
        else:
            loss = self.loss_pitch(scores_pitch, pitches)
        return loss


class PitchSpellingGNN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats_enc, out_feats_pc, out_feats_ks, n_layers, metadata, dropout=0.5, add_seq=False):
        super(PitchSpellingGNN, self).__init__()
        self.gnn_enc = MetricalGNN(in_feats, n_hidden, out_feats_enc, n_layers, metadata, dropout=dropout)
        self.normalize = gnn.GraphNorm(out_feats_enc)
        self.add_seq = add_seq
        if add_seq:
            self.rnn = nn.GRU(
                input_size=in_feats,
                hidden_size=n_hidden // 2,
                bidirectional=True,
                num_layers=1,
                batch_first=True,
            )
            self.rnn_norm = nn.LayerNorm(n_hidden)
            self.rnn_project = nn.Linear(n_hidden, out_feats_enc)
            self.cat_lin = nn.Linear(out_feats_enc * 2, out_feats_enc)
            self.rnn_ks = nn.GRU(
                input_size=out_feats_enc + out_feats_pc,
                hidden_size=n_hidden // 2,
                bidirectional=True,
                num_layers=1,
                batch_first=True,
            )
            self.rnn_norm_ks = nn.LayerNorm(n_hidden)
            self.rnn_project_ks = nn.Linear(n_hidden, out_feats_enc + out_feats_pc)


        self.mlp_clf_pc = nn.Sequential(
            nn.Linear(out_feats_enc, out_feats_enc // 2),
            nn.ReLU(),
            nn.LayerNorm(out_feats_enc // 2),
            nn.Dropout(dropout),
            nn.Linear(out_feats_enc // 2, out_feats_pc),
        )
        self.mlp_clf_ks = nn.Sequential(
            nn.Linear(out_feats_enc + out_feats_pc, out_feats_enc // 2),
            nn.ReLU(),
            nn.LayerNorm(out_feats_enc // 2),
            nn.Dropout(dropout),
            nn.Linear(out_feats_enc // 2, out_feats_ks),
        )

    def sequential_forward(self, note, neighbor_mask_node, batch):
        z = note[neighbor_mask_node["note"] == 0]
        lengths = torch.bincount(batch)
        z = z.split(lengths.tolist())
        z = nn.utils.rnn.pad_sequence(z, batch_first=True)
        z, _ = self.rnn(z)
        z = self.rnn_norm(z)
        z = self.rnn_project(z)
        z = nn.utils.rnn.unpad_sequence(z, lengths, batch_first=True)
        z = torch.cat(z, dim=0)
        return z

    def sequential_ks_forward(self, x, batch):
        lengths = torch.bincount(batch)
        x = x.split(lengths.tolist())
        x = nn.utils.rnn.pad_sequence(x, batch_first=True)
        x, _ = self.rnn_ks(x)
        x = self.rnn_norm_ks(x)
        x = self.rnn_project_ks(x)
        x = nn.utils.rnn.unpad_sequence(x, lengths, batch_first=True)
        x = torch.cat(x, dim=0)
        return x

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node=None, neighbor_mask_edge=None, batch=None):
        x = self.gnn_enc(x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge)
        x = self.normalize(x, batch=batch)
        if self.add_seq:
            z = self.sequential_forward(x_dict["note"], neighbor_mask_node, batch)
            x = torch.cat([x, z], dim=-1)
            x = self.cat_lin(x)
            out_pc = self.mlp_clf_pc(x)
            x = torch.cat([x, out_pc], dim=-1)
            x = self.sequential_ks_forward(x, batch)
            out_ks = self.mlp_clf_ks(x)
            return out_pc, out_ks

        out_pc = self.mlp_clf_pc(x)
        x = torch.cat([x, out_pc], dim=-1)
        out_ks = self.mlp_clf_ks(x)
        return out_pc, out_ks


class PitchSpellingNeighborGNN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats_enc, out_feats_pc, out_feats_ks, n_layers, metadata, dropout=0.5):
        super(PitchSpellingNeighborGNN, self).__init__()
        self.gnn_enc = HierarchicalHeteroGraphSage(metadata[1], in_feats, n_hidden, out_feats_enc, n_layers)
        self.normalize = nn.BatchNorm1d(out_feats_enc)
        self.mlp_clf_pc = nn.Sequential(
            nn.Linear(out_feats_enc, out_feats_enc // 2),
            nn.ReLU(),
            nn.LayerNorm(out_feats_enc // 2),
            nn.Dropout(dropout),
            nn.Linear(out_feats_enc // 2, out_feats_pc),
        )
        self.mlp_clf_ks = nn.Sequential(
            nn.Linear(out_feats_enc + out_feats_pc, out_feats_enc // 2),
            nn.ReLU(),
            nn.LayerNorm(out_feats_enc // 2),
            nn.Dropout(dropout),
            nn.Linear(out_feats_enc // 2, out_feats_ks),
        )

    def forward(self, x_dict, edge_index_dict, num_sampled_edges_dict=None, num_sampled_nodes_dict=None):
        x = self.gnn_enc(x_dict, edge_index_dict, num_sampled_edges_dict, num_sampled_nodes_dict)
        # x = self.normalize(x, batch=batch)
        x = self.normalize(x)
        out_pc = self.mlp_clf_pc(x)
        x = torch.cat([x, out_pc], dim=-1)
        out_ks = self.mlp_clf_ks(x)
        return out_pc, out_ks


class PitchSpellingModel(LightningModule):
    def __init__(self,
                 in_feats,
                 n_hidden1,
                 n_hidden2,
                 lr=0.001,
                 weight_decay=5e-4,
        ):
        super(PitchSpellingModel, self).__init__()
        self.save_hyperparameters()
        self.pitch_encoder = PitchEncoder()
        self.key_encoder = KeySignatureEncoder()
        self.module = PKSpell(in_feats, n_hidden1, n_hidden2, self.pitch_encoder.encode_dim, rnn_depth=1)
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_ps = nn.CrossEntropyLoss(reduction="mean")
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean")
        self.acc_ps = Accuracy(task="multiclass", num_classes=self.pitch_encoder.encode_dim)
        self.f1_ps = F1Score(num_classes=self.pitch_encoder.encode_dim, average="macro", task="multiclass")
        self.acc_ks = Accuracy(task="multiclass", num_classes=self.key_encoder.encode_dim)
        self.f1_ks = F1Score(num_classes=self.key_encoder.encode_dim, average="macro", task="multiclass")

    def _common_step(self, batch, batch_idx):
        x = batch["note"].x
        lengths = [data["note"].x.shape[0] for data in batch.to_data_list()]
        labels_ps = batch["note"].pitch_spelling
        labels_ks = batch["note"].key_signature
        preds_pitch, preds_key = self.module(x, lengths)
        # unpad the predictions of pitch
        lengths = torch.tensor(lengths, device=preds_pitch.device, dtype=torch.long)
        preds_pitch = nn.utils.rnn.unpad_sequence(preds_pitch, lengths, batch_first=True)
        preds_pitch = torch.cat(preds_pitch, dim=0)
        preds_key = nn.utils.rnn.unpad_sequence(preds_key, lengths, batch_first=True)
        preds_key = torch.cat(preds_key, dim=0)
        return preds_pitch, labels_ps, preds_key, labels_ks

    def training_step(self, batch, batch_idx):
        p_ps, t_ps, p_ks, t_ks = self._common_step(batch, batch_idx)
        loss_ps = self.loss_ps(p_ps, t_ps.view(-1))
        loss_ks = self.loss_ks(p_ks, t_ks.view(-1))
        loss = loss_ps + loss_ks
        batch_size = t_ps.shape[0]
        self.log("train_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=batch_size, on_step=False)
        self.log("train_acc_ps", self.acc_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size, on_step=False)
        self.log("train_acc_ks", self.acc_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        p_ps, t_ps, p_ks, t_ks = self._common_step(batch, batch_idx)
        loss_ps = self.loss_ps(p_ps, t_ps.view(-1))
        loss_ks = self.loss_ks(p_ks, t_ks.view(-1))
        loss = loss_ps + loss_ks
        batch_size = t_ps.shape[0]
        self.log("val_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("val_acc_ps", self.acc_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("val_f1_ps", self.f1_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("val_acc_ks", self.acc_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("val_f1_ks", self.f1_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        p_ps, t_ps, p_ks, t_ks = self._common_step(batch, batch_idx)
        loss_ps = self.loss_ps(p_ps, t_ps.view(-1))
        loss_ks = self.loss_ks(p_ks, t_ks.view(-1))
        loss = loss_ps + loss_ks
        batch_size = t_ps.shape[0]
        self.log("test_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("test_acc_ps", self.acc_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("test_f1_ps", self.f1_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("test_acc_ks", self.acc_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("test_f1_ks", self.f1_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

class PitchSpellingGNNPL(LightningModule):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 metadata,
                 dropout=0.5,
                 lr=0.001,
                 weight_decay=5e-4,
                 add_seq=False
                 ):
        super(PitchSpellingGNNPL, self).__init__()
        self.save_hyperparameters()
        self.pitch_encoder = PitchEncoder()
        self.key_encoder = KeySignatureEncoder()
        self.module = PitchSpellingGNN(in_feats, n_hidden, 64, self.pitch_encoder.encode_dim,
                                       self.key_encoder.encode_dim, n_layers, metadata, dropout=dropout, add_seq=add_seq)
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_ps = nn.CrossEntropyLoss(reduction="mean")
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean")
        self.acc_ps = Accuracy(task="multiclass", num_classes=self.pitch_encoder.encode_dim)
        self.f1_ps = F1Score(num_classes=self.pitch_encoder.encode_dim, average="macro", task="multiclass")
        self.acc_ks = Accuracy(task="multiclass", num_classes=self.key_encoder.encode_dim)
        self.f1_ks = F1Score(num_classes=self.key_encoder.encode_dim, average="macro", task="multiclass")

    def _common_step(self, batch, batch_idx):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        neighbor_mask_node = {k: batch[k].neighbor_mask for k in batch.node_types}
        neighbor_mask_edge = {k: batch[k].neighbor_mask for k in batch.edge_types}
        # Trim labels to targets
        label_mask = neighbor_mask_node["note"] == 0
        labels_ps = batch["note"].pitch_spelling[label_mask]
        labels_ks = batch["note"].key_signature[label_mask]
        batch_note = batch["note"].batch[label_mask]
        # Forward pass
        pred_ps, pred_ks = self.module(x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge, batch_note)
        return pred_ps, labels_ps, pred_ks, labels_ks

    def training_step(self, batch, batch_idx):
        p_ps, t_ps, p_ks, t_ks = self._common_step(batch, batch_idx)
        loss_ps = self.loss_ps(p_ps, t_ps.view(-1))
        loss_ks = self.loss_ks(p_ks, t_ks.view(-1))
        loss = loss_ps + loss_ks
        batch_size = t_ps.shape[0]
        self.log("train_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=batch_size, on_step=False)
        self.log("train_acc_ps", self.acc_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size, on_step=False)
        self.log("train_acc_ks", self.acc_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        p_ps, t_ps, p_ks, t_ks = self._common_step(batch, batch_idx)
        loss_ps = self.loss_ps(p_ps, t_ps.view(-1))
        loss_ks = self.loss_ks(p_ks, t_ks.view(-1))
        loss = loss_ps + loss_ks
        batch_size = t_ps.shape[0]
        self.log("val_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("val_acc_ps", self.acc_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("val_f1_ps", self.f1_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("val_acc_ks", self.acc_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("val_f1_ks", self.f1_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        p_ps, t_ps, p_ks, t_ks = self._common_step(batch, batch_idx)
        loss_ps = self.loss_ps(p_ps, t_ps.view(-1))
        loss_ks = self.loss_ks(p_ks, t_ks.view(-1))
        loss = loss_ps + loss_ks
        batch_size = t_ps.shape[0]
        self.log("test_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("test_acc_ps", self.acc_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("test_f1_ps", self.f1_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("test_acc_ks", self.acc_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("test_f1_ks", self.f1_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class PitchSpellingNeighborGNNPL(LightningModule):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 metadata,
                 dropout=0.5,
                 lr=0.001,
                 weight_decay=5e-4,
                 ):
        super(PitchSpellingNeighborGNNPL, self).__init__()
        self.save_hyperparameters()
        self.pitch_encoder = PitchEncoder()
        self.key_encoder = KeySignatureEncoder()
        self.module = PitchSpellingNeighborGNN(in_feats, n_hidden, 64, self.pitch_encoder.encode_dim,
                                       self.key_encoder.encode_dim, n_layers, metadata, dropout=dropout)
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_ps = nn.CrossEntropyLoss(reduction="mean")
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean")
        self.acc_ps = Accuracy(task="multiclass", num_classes=self.pitch_encoder.encode_dim)
        self.f1_ps = F1Score(num_classes=self.pitch_encoder.encode_dim, average="macro", task="multiclass")
        self.acc_ks = Accuracy(task="multiclass", num_classes=self.key_encoder.encode_dim)
        self.f1_ks = F1Score(num_classes=self.key_encoder.encode_dim, average="macro", task="multiclass")

    def _common_step(self, batch, batch_idx):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        neighbor_mask_node = {k: batch[k].neighbor_mask for k in batch.node_types}

        # Trim labels to targets
        label_mask = neighbor_mask_node["note"] == 0
        labels_ps = batch["note"].pitch_spelling[label_mask]
        labels_ks = batch["note"].key_signature[label_mask]
        # batch_note = batch["note"].batch # [label_mask]
        # Forward pass
        pred_ps, pred_ks = self.module(x_dict, edge_index_dict)
        pred_ps = pred_ps[label_mask]
        pred_ks = pred_ks[label_mask]
        return pred_ps, labels_ps, pred_ks, labels_ks

    def training_step(self, batch, batch_idx):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        num_sampled_edges_dict = batch.num_sampled_edges_dict
        num_sampled_nodes_dict = batch.num_sampled_nodes_dict
        batch_size = batch["note"].batch_size
        # batch_note = batch["note"].batch[:batch_size]
        p_ps, p_ks = self.module(x_dict, edge_index_dict, num_sampled_edges_dict, num_sampled_nodes_dict)
        t_ps = batch["note"].pitch_spelling[:batch_size]
        t_ks = batch["note"].key_signature[:batch_size]
        p_ps = p_ps[:batch_size]
        p_ks = p_ks[:batch_size]
        loss_ps = self.loss_ps(p_ps, t_ps.view(-1))
        loss_ks = self.loss_ks(p_ks, t_ks.view(-1))
        loss = loss_ps + loss_ks
        batch_size = t_ps.shape[0]
        self.log("train_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=batch_size, on_step=False)
        self.log("train_acc_ps", self.acc_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size, on_step=False)
        self.log("train_acc_ks", self.acc_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        p_ps, t_ps, p_ks, t_ks = self._common_step(batch, batch_idx)
        loss_ps = self.loss_ps(p_ps, t_ps.view(-1))
        loss_ks = self.loss_ks(p_ks, t_ks.view(-1))
        loss = loss_ps + loss_ks
        batch_size = t_ps.shape[0]
        self.log("val_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("val_acc_ps", self.acc_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("val_f1_ps", self.f1_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("val_acc_ks", self.acc_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("val_f1_ks", self.f1_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        p_ps, t_ps, p_ks, t_ks = self._common_step(batch, batch_idx)
        loss_ps = self.loss_ps(p_ps, t_ps.view(-1))
        loss_ks = self.loss_ks(p_ks, t_ks.view(-1))
        loss = loss_ps + loss_ks
        batch_size = t_ps.shape[0]
        self.log("test_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("test_acc_ps", self.acc_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("test_f1_ps", self.f1_ps(p_ps, t_ps.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("test_acc_ks", self.acc_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)
        self.log("test_f1_ks", self.f1_ks(p_ks, t_ks.view(-1)), prog_bar=True, on_epoch=True,
                    batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
