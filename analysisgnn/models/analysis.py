import torch_scatter
from torchmetrics import Accuracy, F1Score
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
from copy import deepcopy
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from graphmuse.nn.models.metrical_gnn import HybridGNN, HybridHGT, MetricalGNN
from pytorch_lightning import LightningModule
# Removed: from analysisgnn.models.vocsep.pl_models import isin_pairwise
from analysisgnn.models.cadence import SMOTE
from typing import List, Union, Dict, Any
from analysisgnn.models.chord import MultiTaskLoss
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
import math
import warnings
from analysisgnn.utils.chord_representations import available_representations


def isin_pairwise(element,test_elements, assume_unique=True):
    """Like isin function of torch, but every element in the sequence is a pair of integers.
    # TODO: check if this solution can be better https://stackoverflow.com/questions/71708091/is-there-an-equivalent-numpy-function-to-isin-that-works-row-based
    
    Args:
        element (torch.Tensor): Tensor of shape (2, N) where N is the number of elements.
        test_elements (torch.Tensor): Tensor of shape (2, M) where M is the number of elements to test.
        assume_unique (bool, optional): If True, the input arrays are both assumed to be unique, which can speed up the calculation. Defaults to True.
        
        Returns:
            torch.Tensor: Tensor of shape (M,) with boolean values indicating whether the element is in the test_elements.
                        
    """
    def cantor_pairing(x, y):
        return (x + y) * (x + y + 1) // 2 + y

    element_cantor_proj = cantor_pairing(element[0], element[1])
    test_elements_cantor_proj = cantor_pairing(test_elements[0], test_elements[1])
    return torch.isin(element_cantor_proj, test_elements_cantor_proj, assume_unique=assume_unique)


class LinearWarmupCosineAnnealingLR(LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a cosine annealing schedule between
    base_lr and eta_min.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Maximum number of steps for linear warmup
            max_epochs (int): Maximum number of epochs
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.current_step = 0

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        # Handle warmup based on steps
        if self.current_step < self.warmup_steps:
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (self.current_step / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        
        # After warmup, use cosine annealing schedule based on epochs
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps / self.steps_per_epoch) / 
                         (self.max_epochs - self.warmup_steps / self.steps_per_epoch)))
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        # Increment step counter
        self.current_step += 1
        
        # Calculate steps_per_epoch if not already set
        if not hasattr(self, 'steps_per_epoch') and self.last_epoch > 0:
            self.steps_per_epoch = self.current_step / self.last_epoch
        
        return super().step(epoch)

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.current_step < self.warmup_steps:
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (self.current_step / self.warmup_steps)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps / self.steps_per_epoch) / 
                         (self.max_epochs - self.warmup_steps / self.steps_per_epoch)))
            for base_lr in self.base_lrs
        ]


class LinearWarmupExponentialDecayLR(LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by an exponential decay schedule between
    base_lr and eta_min.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        decay_steps: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        gamma: float = 0.999,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Maximum number of steps for linear warmup
            decay_steps (int): Number of steps for exponential decay
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            gamma (float): Multiplicative factor of learning rate decay. Default: 0.95.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.gamma = gamma
        self.current_step = 0

        super(LinearWarmupExponentialDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        # Handle warmup based on steps
        if self.current_step < self.warmup_steps:
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (self.current_step / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        
        # After warmup, use exponential decay
        decay_step = self.current_step - self.warmup_steps
        decay_factor = self.gamma ** (decay_step / self.decay_steps)
        
        return [
            max(self.eta_min, base_lr * decay_factor)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        # Increment step counter
        self.current_step += 1
        return super().step(epoch)

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.current_step < self.warmup_steps:
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (self.current_step / self.warmup_steps)
                for base_lr in self.base_lrs
            ]

        decay_step = self.current_step - self.warmup_steps
        decay_factor = self.gamma ** (decay_step / self.decay_steps)
        
        return [
            max(self.eta_min, base_lr * decay_factor)
            for base_lr in self.base_lrs
        ]


class FAMO:
    """
    Fast Adaptive Multitask Optimization.

    This class implements the FAMO algorithm for multitask learning.
    Re-implementation of the algorithm described in the paper:
    "FAMO: Fast Adaptive Multitask Optimization"
    taken from the repository:
    https://github.com/Cranial-XIX/FAMO/blob/main/famo.py
    """

    def __init__(
            self,
            task_dict: dict,
            device: torch.device,
            gamma: float = 0.01,  # the regularization coefficient
            w_lr: float = 0.025,  # the learning rate of the task logits
            max_norm: float = 1.0,  # the maximum gradient norm
    ):
        n_tasks = len(task_dict.keys())
        self.task_dict = {task: i for i, task in enumerate(task_dict.keys())}
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.prev_loss = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = n_tasks
        self.device = device

    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, loss_dict):
        mask = torch.zeros_like(self.min_losses).bool()
        losses = torch.zeros_like(self.min_losses)
        for task, loss in loss_dict.items():
            mask[self.task_dict[task]] = True
            losses[self.task_dict[task]] = loss
        self.prev_loss[mask] = losses
        z = F.softmax(self.w[mask], -1)
        D = losses - self.min_losses[mask] + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()

    def backward(
            self,
            loss_dict: torch.Tensor,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
    ) -> Union[torch.Tensor, None]:
        """

        Parameters
        ----------
        loss_dict :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        Returns
        -------
        Loss, extra outputs
        """
        loss = self.get_weighted_loss(loss_dict=loss_dict)
        loss.backward()
        if self.max_norm > 0 and shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return loss


class PreEncoder(nn.Module):
    def __init__(self, metadata, in_channels, out_channels, num_layers, heads, dropout=0.5, jk=True):
        super().__init__()
        self.encoder = HybridHGT(metadata, in_channels, out_channels, num_layers,
                                 heads=heads, dropout=dropout, jk=jk)
        self.pitch_spelling_classes = 35
        self.fifths_classes = 15
        self.staff_clf = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels),
        )
        self.voice_clf = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels),
        )
        self.fifths_clf = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, self.fifths_classes),
        )
        self.spelling_clf = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, self.pitch_spelling_classes),
        )

    def forward(self, x_dict, edge_index_dict, batch_dict, batch_size, neighbor_mask_node,
                neighbor_mask_edge, staff_candidate_edges, voice_candidate_edges, return_embedding=False):
        x = self.encoder(x_dict, edge_index_dict, batch_dict, batch_size, neighbor_mask_node,
                neighbor_mask_edge)
        # staff_x = torch.cat([x[staff_candidate_edges[0]], x[staff_candidateEdges[1]]], dim=1)
        staff_x = self.staff_clf(x)
        voice_x = self.voice_clf(x)
        # voice_x = torch.cat([x[voice_candidate_edges[0]], x[voice_candidate_edges[1]]], dim=1)
        staff_logits = (staff_x[staff_candidate_edges[0]] * staff_x[staff_candidate_edges[1]]).sum(-1)
        voice_logits = (voice_x[voice_candidate_edges[0]] * voice_x[voice_candidate_edges[1]]).sum(-1)
        fifths_logits = self.fifths_clf(x)
        spelling_logits = self.spelling_clf(x)
        if return_embedding:
            return staff_logits, voice_logits, fifths_logits, spelling_logits, x
        return staff_logits, voice_logits, fifths_logits, spelling_logits


class CrossTaskTransformer(nn.Module):
    def __init__(self, proj_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)
        
    def forward(self, task_projections):
        # task_projections: (batch_size, num_tasks, proj_dim)
        attended, _ = self.multihead_attn(task_projections, task_projections, task_projections)
        return self.norm(task_projections + attended)
    

class TorchAnalysisGNN(nn.Module):
    def __init__(self, metadata, in_channels, hidden_channels, out_channels, task_dict, num_layers, dropout=0.5, use_jk=True, logit_fusion=True, use_rnn=False, encoder_type="hybridgnn"):
        super().__init__()
        self.pitch_embedding = nn.Embedding(35, 64)
        self.key_embedding = nn.Embedding(15, 64)
        self.logit_fusion = logit_fusion
        self.use_rnn = use_rnn
        self.hidden_channels = hidden_channels        
        self.project_dict = nn.ModuleDict({
            k: (nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.LayerNorm(hidden_channels),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            ) if k != "note" else nn.Sequential(
                nn.Linear(in_channels+128, hidden_channels),
                nn.ReLU(),
                nn.LayerNorm(hidden_channels),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )) for k in metadata[0]
        })
        if encoder_type == "hgt":
            self.encoder = HybridHGT(
                metadata=metadata,
                input_channels=hidden_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                heads=4,
                dropout=dropout,
                use_jk=use_jk
            )
        elif encoder_type == "hybridgnn":
            self.encoder = HybridGNN(
                metadata=metadata,
                input_channels=hidden_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
                use_jk=use_jk
            )
        elif encoder_type == "metricalgnn":
            self.encoder = MetricalGNN(
                metadata=metadata,
                input_channels=hidden_channels,
                hidden_channels=hidden_channels,
                output_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
                use_jk=use_jk,
                fast=True
            )
        self.project_enc = nn.Sequential(
            nn.LayerNorm(2*hidden_channels),
            nn.Linear(2*hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )
        self.clf_dict = nn.ModuleDict(
            {
                task_name: nn.Sequential(
                    nn.Linear(out_channels, out_channels // 2),
                    nn.ReLU(),
                    nn.LayerNorm(out_channels // 2),
                    nn.Linear(out_channels // 2, num_classes),
                )
                for task_name, num_classes in task_dict.items()
            }
        )
        if logit_fusion:
            # Each task’s classifier logits are projected to a common space.
            self.clf_proj_layers = nn.ModuleDict({
                task: nn.Sequential(
                    nn.Linear(out_dim, out_channels // 2),
                    nn.ReLU(),
                    nn.LayerNorm(out_channels // 2),                    
                    ) for task, out_dim in task_dict.items()
            })
            self.cross_task_transformer = CrossTaskTransformer(out_channels // 2, num_heads=4, dropout=dropout)

            # Fusion layers combine each task's own projected logits with the aggregated projections from other tasks.
            self.fusion_layers = nn.ModuleDict({
                task: nn.Linear(out_channels // 2, out_dim) for task, out_dim in task_dict.items()
            })
        if use_rnn:
            self.rnn = nn.GRU(out_channels, out_channels, num_layers=2, batch_first=True, bidirectional=True)
            self.rnn_norm = nn.LayerNorm(out_channels)
            self.rnn_mlp = nn.Sequential(
                nn.Linear(out_channels, out_channels // 2),
                nn.ReLU(),
                nn.LayerNorm(out_channels),
                nn.Dropout(dropout),
                nn.Linear(out_channels, out_channels),
            )
        else:
            self.rnn = nn.Identity()
            self.rnn_norm = nn.Identity()
            self.rnn_mlp = nn.Identity()        

    def rnn_forward(self, x, batch):
        # NOTE optimize sampling to order sequences by length
        lengths = torch.bincount(batch)
        x = x.split(lengths.tolist())
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0)
        x, _ = self.rnn(x)
        x = self.rnn_norm(x)
        x = self.rnn_mlp(x)
        x = nn.utils.rnn.unpad_sequence(x, batch_first=True, lengths=lengths)
        x = torch.cat(x, dim=0)
        return x

    def forward(self, pitch_spelling, key_signature, x_dict, edge_index_dict, batch_dict, batch_size, neighbor_mask_node,
                neighbor_mask_edge):
        x = self.encode(
            pitch_spelling, key_signature, x_dict, edge_index_dict, batch_dict, batch_size, neighbor_mask_node, neighbor_mask_edge)
        logits_dict = self.forward_clf(x)
        return logits_dict

    def forward_clf(self, x, tasks=None):
        tasks = self.clf_dict.keys() if tasks is None else tasks
        raw_logits = {task_name: self.clf_dict[task_name](x) for task_name in tasks}

        if self.logit_fusion:
            refined_logits = {}
            proj_logits = {task_name: self.clf_proj_layers[task_name](raw_logits[task_name]) for task_name in raw_logits.keys()}
            
            # Stack all projections
            task_names = list(proj_logits.keys())
            proj_stack = torch.stack([proj_logits[task] for task in task_names], dim=1)  # (batch_size, num_tasks, proj_dim)
            
            # Apply cross-task attention
            enhanced_projs = self.cross_task_transformer(proj_stack)  # (batch_size, num_tasks, proj_dim)
            
            # Generate refined logits for each task
            for i, task in enumerate(task_names):
                if task in tasks:
                    refined = self.fusion_layers[task](enhanced_projs[:, i])
                    refined_logits[task] = refined
                
            return refined_logits

        return raw_logits

    def encode(self, pitch_spelling, key_signature, x_dict, edge_index_dict, batch_dict, batch_size, neighbor_mask_node, neighbor_mask_edge):
        # initialize all values of x_dict with zeros and size self.hidden_channels except from notes
        z_dict = {k: v.clone() for k, v in x_dict.items()}
        z_dict["note"] = torch.cat([z_dict["note"], self.pitch_embedding(pitch_spelling), self.key_embedding(key_signature)], dim=-1)
        h_dict = {k: self.project_dict[k](z_dict[k]) for k in self.project_dict.keys()}
        x = self.encoder(
            x_dict=h_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict,
            batch_size=batch_size, neighbor_mask_node=neighbor_mask_node,
            neighbor_mask_edge=neighbor_mask_edge, return_edge_index=False, edge_attr_dict=None)
        onset_edges = edge_index_dict[("note", "onset", "note")]
        onset_edge_mask = torch.logical_and(onset_edges[0] < batch_size, onset_edges[1] < batch_size)
        onset_edges = onset_edges[:, onset_edge_mask]
        # remove self loops
        onset_edges = onset_edges[:, onset_edges[0] != onset_edges[1]]
        # torch scatter mean
        x_pool = torch_scatter.scatter_mean(x[onset_edges[1]], onset_edges[0], dim=0, dim_size=x.size(0), out=x.clone())
        x = torch.cat([x, x_pool], dim=-1)
        x = self.project_enc(x)
        if self.use_rnn:
            x = self.rnn_forward(x, batch_dict["note"][:batch_size])
        return x

    def clf_task(self, x, task_name):
        return self.clf_dict[task_name](x)

    def predict(self, x_dict, edge_index_dict, staff_candidate_edges, voice_candidate_edges):
        logits_dict = self.forward(x_dict, edge_index_dict, staff_candidate_edges, voice_candidate_edges)
        preds_dict = {
            task_name: F.softmax(logits, dim=-1)
            for task_name, logits in logits_dict.items()
        }
        return preds_dict


class AnalysisGNN(LightningModule):
    def __init__(
        self,
        metadata,
        encoder_in_channels,
        encoder_hidden_channels,
        encoder_out_channels,
        classifier_hidden_channels,
        classifier_out_channels,
        task_dict,
        encoder_layers,
        clf_layers,
        dropout=0.5,
        lr = 0.001,
        weight_decay = 0.0005,
        ):
        super().__init__()
        self.encoder = PreEncoder(encoder_in_channels, encoder_hidden_channels, encoder_out_channels, encoder_layers, dropout)
        self.clf = TorchAnalysisGNN(encoder_out_channels, classifier_hidden_channels, classifier_out_channels, task_dict, encoder_out_channels, clf_layers, dropout)
        self.task_dict = task_dict
        self.losses = {k: nn.CrossEntropyLoss() for k in task_dict.keys()}
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x_dict = batch.x_dict
        labels_dict = {k: batch["note"][k] for k in self.task_dict.keys() if k in batch["note"].keys()}
        edge_index_dict = batch.edge_index_dict
        batch_dict = batch.batch_dict
        batch_size_enc = batch["note"].batch_size
        batch_size_clf = batch["note"].batch_size
        num_sampled_edges_dict = batch.num_sampled_edges_dict
        num_sampled_nodes_dict = batch.num_sampled_nodes_dict
        voice_candidate_edges = edge_index_dict.pop(("note", "voice_cand", "note"))
        edge_index_dict.pop(("note", "voice", "note"))
        edge_index_dict.pop(("note", "staff", "note"))
        staff_candidate_edges = edge_index_dict.pop(("note", "staff_cand", "note"))
        staff_logits, voice_logits, fifths_logits, spelling_logits, x = self.encoder(x_dict, edge_index_dict, batch_dict, batch_size_enc, num_sampled_nodes_dict, num_sampled_edges_dict, staff_candidate_edges, voice_candidate_edges, return_embedding=True)
        x_dict = {"note": x}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k[0] == "note" and k[-1] == "note"}
        edge_index_dict[("note", "voice", "note")] = voice_candidate_edges[voice_logits > 0.5]
        edge_index_dict[("note", "staff", "note")] = staff_candidate_edges[staff_logits > 0.5]
        logits_dict = self.clf(x_dict, edge_index_dict, batch_dict, batch_size_clf, num_sampled_nodes_dict, num_sampled_edges_dict)
        total_loss = 0
        for task_name, labels in labels_dict.items():
            loss = self.losses[task_name](logits_dict[task_name], labels)
            self.log(f"{task_name}_loss", loss)
            total_loss += loss

        self.log("total_loss", total_loss)
        return total_loss


class PreEncoderPL(LightningModule):
    def __init__(
        self,
        metadata,
        in_channels,
        hidden_channels,
        num_layers,
        heads=4,
        dropout=0.5,
        lr=0.001,
        weight_decay=0.0005,
        warmup_steps=500,
        multi_task_weight_strategy="famo",
        devices=1,
    ):
        super().__init__()
        self.model = PreEncoder(metadata, in_channels, hidden_channels, num_layers, heads=heads, dropout=dropout)
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_warmup_steps = warmup_steps
        self.metadata = metadata
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.staff_loss = nn.BCEWithLogitsLoss()
        self.voice_loss = nn.BCEWithLogitsLoss()
        self.fifths_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.spelling_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.spelling_acc = Accuracy(task="multiclass", num_classes=self.model.pitch_spelling_classes)
        self.fifths_acc = Accuracy(task="multiclass", num_classes=self.model.fifths_classes)
        self.staff_f1 = F1Score(task="binary", num_classes=1, average="macro")
        self.voice_f1 = F1Score(task="binary", num_classes=1, average="macro")
        dev = devices[0] if isinstance(devices, list) else devices
        self.multitask_weight_stategy = FAMO(n_tasks=4, device=dev) if multi_task_weight_strategy == "famo" else None
        self.automatic_optimization = multi_task_weight_strategy != "famo"
        self.save_hyperparameters()

    def _common_step(self, batch, batch_idx, mode="train"):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        batch_dict = batch.batch_dict
        batch_size = batch["note"].batch_size
        num_sampled_edges_dict = batch.num_sampled_edges_dict
        num_sampled_nodes_dict = batch.num_sampled_nodes_dict
        staff_candidate_edges = torch.cat(
            (edge_index_dict[("note", "onset", "note")], edge_index_dict[("note", "consecutive", "note")]), dim=1)
        # sort the src nodes of the staff edge index
        staff_candidate_edges = staff_candidate_edges[:, staff_candidate_edges[0].argsort()]
        voice_candidate_edges = edge_index_dict[("note", "consecutive", "note")]
        staff_true_edges = edge_index_dict.pop(("note", "staff", "note"))
        voice_true_edges = edge_index_dict.pop(("note", "voice", "note"))
        # Filter edges to batch_size
        staff_candidate_edges = staff_candidate_edges[:,
                                torch.logical_and(staff_candidate_edges[0] < batch_size, staff_candidate_edges[1] < batch_size)]
        voice_candidate_edges = voice_candidate_edges[:,
                                torch.logical_and(voice_candidate_edges[0] < batch_size, voice_candidate_edges[1] < batch_size)]
        staff_true_edges = staff_true_edges[:, torch.logical_and(staff_true_edges[0] < batch_size, staff_true_edges[1] < batch_size)]
        voice_true_edges = voice_true_edges[:, torch.logical_and(voice_true_edges[0] < batch_size, voice_true_edges[1] < batch_size)]
        staff_labels = isin_pairwise(staff_candidate_edges, staff_true_edges, assume_unique=True)
        voice_labels = isin_pairwise(voice_candidate_edges, voice_true_edges, assume_unique=True)
        fifths_labels = batch["note"].key_signature[:batch_size]
        spelling_labels = batch["note"].pitch_spelling[:batch_size]
        staff_logits, voice_logits, fifths_logits, spelling_logits = self.model(x_dict, edge_index_dict, batch_dict,
                                                                                batch_size, num_sampled_nodes_dict,
                                                                                num_sampled_edges_dict,
                                                                                staff_candidate_edges,
                                                                                voice_candidate_edges)
        staff_loss = self.staff_loss(staff_logits.squeeze(), staff_labels.squeeze().float())
        voice_loss = self.voice_loss(voice_logits.squeeze(), voice_labels.squeeze().float())
        fifths_loss = self.fifths_loss(fifths_logits, fifths_labels)
        spelling_loss = self.spelling_loss(spelling_logits, spelling_labels)
        total_loss = staff_loss + voice_loss + fifths_loss + spelling_loss
        if torch.isnan(total_loss):
            return
        self.log(f"{mode}_staff_loss", staff_loss, batch_size=batch_size)
        self.log(f"{mode}_voice_loss", voice_loss, batch_size=batch_size)
        self.log(f"{mode}_fifths_loss", fifths_loss, batch_size=batch_size)
        self.log(f"{mode}_spelling_loss", spelling_loss, batch_size=batch_size)
        self.log(f"{mode}_total_loss", total_loss, batch_size=batch_size, prog_bar=True)
        if mode != "train":
            self.log(f"{mode}_staff_f1", self.staff_f1(staff_logits.squeeze(), staff_labels.squeeze().float()), batch_size=batch_size)
            self.log(f"{mode}_voice_f1", self.voice_f1(voice_logits.squeeze(), voice_labels.squeeze().float()), batch_size=batch_size)
            self.log(f"{mode}_fifths_acc", self.fifths_acc(fifths_logits, fifths_labels), batch_size=batch_size)
            self.log(f"{mode}_spelling_acc", self.spelling_acc(spelling_logits, spelling_labels), batch_size=batch_size)
        return total_loss

    def training_step(self, batch, batch_idx):
        if self.automatic_optimization:
            loss = self._common_step(batch, batch_idx, mode="train")
        else:
            opt = self.optimizers()
            opt.zero_grad()
            loss = self._common_step(batch, batch_idx, mode="train")
            if loss is None:
                return
            self.multitask_weight_stategy.backward(loss)
            if self.trainer.global_step < self.lr_warmup_steps:
                lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
                for pg in opt.param_groups:
                    pg["lr"] = lr_scale * self.lr
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
        return loss

    def validation_step(self, batch, batch_idx):
        total_loss = self._common_step(batch, batch_idx, mode="validation")

    def on_validation_epoch_end(self) -> None:
        if not self.automatic_optimization and self.trainer.global_step > self.lr_warmup_steps:
            sched = self.lr_schedulers()
            sched.step()

    def test_step(self, batch, batch_idx):
        total_loss = self._common_step(batch, batch_idx, mode="test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5, last_epoch=-1),
            'name': 'learning_rate',
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "total_loss"}

    # learning rate warm-up for automatic optimization
    # def optimizer_step(
    #         self,
    #         epoch,
    #         batch_idx,
    #         optimizer,
    #         optimizer_idx,
    #         optimizer_closure,
    #         on_tpu=False,
    #         using_native_amp=False,
    #         using_lbfgs=False,
    # ):
    #     # skip the first 500 steps
    #     if self.trainer.global_step < self.lr_warmup_steps:
    #         lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
    #         for pg in optimizer.param_groups:
    #             pg["lr"] = lr_scale * self.hparams.learning_rate
    #
    #     # update params
    #     optimizer.step(closure=optimizer_closure)


class EdgeDecoder(nn.Module):
    def __init__(self, channels, edge_types, dropout=0.5):
        super().__init__()
        
        self.embed = nn.ModuleDict()
        for edge_type in edge_types:
            self.embed[edge_type] = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.LayerNorm(channels),
                nn.Dropout(dropout)
            )        
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
            nn.Linear(channels, 2)  # Binary classification for edge existence
        )

    def forward(self, edge_dict, x):
        logit_dict = {}
        for edge_type, (src, dst) in edge_dict.items():
            if edge_type[1] not in self.embed:
                raise ValueError(f"Edge type {edge_type} not found in embed dictionary.")
            src_embed = self.embed[edge_type[1]](x[src])
            dst_embed = self.embed[edge_type[1]](x[dst])
            # edge_features should be cosine similarity between src and dst embeddings so that edge_features is N x channels
            edge_features = src_embed * dst_embed
            logit_dict[edge_type] = self.fc(edge_features)
        return logit_dict


class ContinualAnalysisGNN(LightningModule):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()
        encoder_type = hparams.get("model", "hybridgnn").lower()
        # save hparams as attributes
        self.model = TorchAnalysisGNN(
            metadata=hparams["metadata"],
            in_channels=hparams["in_channels"],
            hidden_channels=hparams["hidden_channels"],
            out_channels=hparams["out_channels"],
            task_dict=hparams["task_dict"],
            num_layers=hparams["num_layers"],
            dropout=hparams["dropout"],
            logit_fusion=hparams.get("logit_fusion", False),
            use_rnn = hparams.get("use_rnn", False),
            use_jk = hparams.get("use_jk", True),
            encoder_type=encoder_type
        )
        self.use_edge_loss = hparams.get("use_edge_loss", False)
        if self.use_edge_loss:
            self.edge_clf = EdgeDecoder(
                channels=hparams["out_channels"],
                edge_types=list(set([v[1] for v in hparams["metadata"][1] if v[0] == "note" and v[2] == "note"])),
                dropout=hparams.get("dropout", 0.5)
            )
            self.edge_loss = nn.CrossEntropyLoss(
                ignore_index=-1,
                label_smoothing=0.1,
            )
        self.main_tasks = hparams.get("main_tasks", ["rna", "cadence", "all"])
        self.total_epochs = hparams["num_epochs"]
        self.epochs_per_task = hparams.get("epochs_per_task", [self.total_epochs// len(self.main_tasks)] * len(self.main_tasks))
        self.lr = hparams["lr"]
        self.weight_decay = hparams["weight_decay"]
        self.use_ewc = hparams.get("use_ewc", False)
        self.fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self._means = {}
        self.has_memories = hparams.get("has_memories", False)
        self.mt_strategy = hparams.get("mt_strategy", None)
        self.lambda_ewc = hparams.get("lambda_ewc", 2.0)
        self.cl_training = hparams.get("cl_training", False)
        self.task_dict = hparams["task_dict"]        
        loss_dict = nn.ModuleDict(
            {
                task: nn.CrossEntropyLoss(
                    ignore_index=-1, 
                    label_smoothing=0.1,                    
                    ) for task in self.task_dict.keys()
                }
            )
        
        self.accuracy_dict = nn.ModuleDict({k: Accuracy(task="multiclass", num_classes=v) for k, v in self.task_dict.items()})
        self.f1_dict = nn.ModuleDict({k: F1Score(task="multiclass", num_classes=v, average="macro") for k, v in self.task_dict.items()})
        self.dctn_loss_dict = nn.ModuleDict({k: nn.KLDivLoss(reduction="batchmean") for k in self.task_dict.keys()})
        self.use_smote = hparams.get("use_smote", False)
        self.smote = SMOTE(dims=hparams["out_channels"], distance="euclidean", k=3)
        if self.mt_strategy == 'famo':
            self.automatic_optimization = False
            self.clf_loss = nn.ModuleDict({task: nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1) for task in self.task_dict.keys()})
            self.famo = FAMO(n_tasks=len(self.task_dict.keys()), device=self.device)
        elif self.mt_strategy == 'wloss':
            self.clf_loss = MultiTaskLoss(
                tasks=list(self.task_dict.keys()),
                loss_ft=loss_dict,
                requires_grad=True)
        else:
            self.clf_loss = MultiTaskLoss(
                tasks=list(self.task_dict.keys()),
                loss_ft=loss_dict,
                requires_grad=False)
        self.lambda_dctn = hparams.get("lambda_dctn", 0.5)
        self.lambda_featl = hparams.get("lambda_featl", 0.1)
        self.previous_tasks = []
        
        
        
        self.current_task = self.main_tasks[0] if self.cl_training else self.main_tasks
        self.current_val_tasks = [self.main_tasks[0]] if self.cl_training else self.main_tasks
        self.save_hyperparameters(hparams)
        if self.lambda_dctn > 0 and len(self.main_tasks) > 1:
            self.memory_model = TorchAnalysisGNN(
                metadata=hparams["metadata"],
                in_channels=hparams["in_channels"],
                hidden_channels=hparams["hidden_channels"],
                out_channels=hparams["out_channels"],
                task_dict=hparams["task_dict"],
                num_layers=hparams["num_layers"],
                dropout=hparams["dropout"],
                logit_fusion=hparams.get("logit_fusion", False),
                use_rnn=hparams.get("use_rnn", False),
                use_jk=hparams.get("use_jk", True),
                encoder_type=encoder_type
            )
            self.update_memory_model()


    def create_mask_dict(self, labels_dict, batch, batch_size):
        mask_dict = {k: torch.ones_like(v).bool() for k, v in labels_dict.items()}
        if "valid_cadence_label" in batch["note"].keys():
            mask_dict["cadence"] = batch["note"]["valid_cadence_label"][:batch_size].bool()
        if "has_phrase" in batch["note"].keys():
            mask_dict["phrase"] = batch["note"]["has_phrase"][:batch_size].bool()
        if "valid_organ_point_label" in batch["note"].keys():
            mask_dict["pedal"] = batch["note"]["valid_organ_point_label"][:batch_size].bool()
        if "valid_section_start_label" in batch["note"].keys():
            mask_dict["section"] = batch["note"]["valid_section_start_label"][:batch_size].bool()
        return mask_dict

    def common_step(self, batch):
        x_dict = batch.x_dict
        batch_size = batch["note"].batch_size
        labels_dict = {k: batch["note"][k][:batch_size] for k in self.task_dict.keys() if k in batch["note"].keys()}
        pitch_spelling = batch["note"].pitch_spelling
        key_signature = batch["note"].key_signature        
        # limit labels to the num_labels values of the self.task_dict and set the rest to 0
        labels_dict = {
            k: torch.where(labels_dict[k] < self.task_dict[k], labels_dict[k], torch.zeros_like(labels_dict[k])) for k
            in labels_dict.keys()}
        edge_index_dict = batch.edge_index_dict
        batch_dict = batch.batch_dict

        num_sampled_edges_dict = batch.num_sampled_edges_dict
        num_sampled_nodes_dict = batch.num_sampled_nodes_dict

        mask_dict = self.create_mask_dict(labels_dict, batch, batch_size)

        # NOTE: mask to remove invalid labels
        if "valid_label" not in batch["note"].keys():
            valid_label_mask = torch.ones_like(batch["note"]["pitch_spelling"][:batch_size]).bool()
        else:
            valid_label_mask = batch["note"]["valid_label"][:batch_size].bool()

        labels_dict = {k: v[valid_label_mask] for k, v in labels_dict.items()}
        mask_dict = {k: v[valid_label_mask] for k, v in mask_dict.items()}
        labels_dict = {k: v[mask_dict[k]] for k, v in labels_dict.items()}

        x = self.model.encode(
            pitch_spelling=pitch_spelling,
            key_signature=key_signature,
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            batch_dict=batch_dict,
            batch_size=batch_size,
            neighbor_mask_node=num_sampled_nodes_dict, neighbor_mask_edge=num_sampled_edges_dict)
        # feature loss verifies that the features are not too different
        feature_loss = x.pow(2).mean()

        # NOTE: EDGE_BASED_LABELING
        edge_loss = torch.tensor(0.0, device=x.device)
        rna_keys = ["quality", "inversion", "degree1", "degree2", "localkey"]
        if self.use_edge_loss and all(rna_key in labels_dict.keys() for rna_key in rna_keys):
            # target_edge_index_dict 
            target_edge_index_dict = {k: v[:, (v[0] < batch_size) & (v[1] < batch_size)] for k, v in edge_index_dict.items() if k[0] == "note" and k[-1] == "note"}
            # randomly select the same number of edges for each key as the batch size
            for k, v in target_edge_index_dict.items():
                if v.size(1) > batch_size:
                    target_edge_index_dict[k] = v[:, torch.randperm(v.size(1))[:batch_size]]
                else:
                    target_edge_index_dict[k] = v
            
            # edges share the same label when rna_keys are the same
            
            ground_truth_same_label_edge_dict = {
                k: torch.zeros(v.shape[-1], device=x.device) for k, v in target_edge_index_dict.items()
            }
            # for every edge type if all the source and target nodes have the same label, then the edge label is True else False
            for k, v in target_edge_index_dict.items():
                if k[0] == "note" and k[-1] == "note":
                    src_labels = torch.stack([labels_dict[rna_key][v[0]] for rna_key in rna_keys], dim=1)
                    tgt_labels = torch.stack([labels_dict[rna_key][v[1]] for rna_key in rna_keys], dim=1)
                    same_label_mask = (src_labels == tgt_labels).all(dim=1)
                    ground_truth_same_label_edge_dict[k] = same_label_mask.long()
            # compute logits for the edges
            edge_logits_dict = self.edge_clf(target_edge_index_dict, x)
            # compute edge loss by stacking the loss in a tensor
            edge_loss = torch.tensor(0.0, device=self.device)
            for k, v in edge_logits_dict.items():
                if k in ground_truth_same_label_edge_dict.keys():
                    edge_loss += self.edge_loss(v, ground_truth_same_label_edge_dict[k])

            edge_loss /= len(edge_logits_dict.keys())                                        

        x = x[valid_label_mask]

        if "cadence" in labels_dict.keys() and len(labels_dict.keys()) == 1 and self.use_smote:
            y = labels_dict["cadence"]
            x_over, y_over = self.smote.fit_generate(x, y)
            labels_dict["cadence"] = y_over
            mask_dict["cadence"] = torch.ones_like(y_over).bool()
            feature_loss = self.update_feature_loss(feature_loss, x_over, y_over, x, y, batch_size)
            x = x_over

        logits_dict = self.model.forward_clf(x)
        logits_dict = {k: logits_dict[k][mask_dict[k]] for k in labels_dict.keys()}
        # TODO remove labels and logits based on has_cadence and has_phrase masks here

        loss_dict = self.clf_loss(logits_dict, labels_dict)
        # pop the total loss and remove it from the dict
        total_loss = loss_dict.pop("total") / len(labels_dict.keys())

        memory_loss = 0
        if len(self.previous_tasks) > 0:
            if self.lambda_dctn > 0:
                x = self.memory_model.encode(
                    pitch_spelling=pitch_spelling,
                    key_signature=key_signature,
                    x_dict=x_dict,
                    edge_index_dict=edge_index_dict,
                    batch_dict=batch_dict,
                    batch_size=batch_size,
                    neighbor_mask_node=num_sampled_nodes_dict, neighbor_mask_edge=num_sampled_edges_dict
                )
                logits_dict = self.model.forward_clf(x, self.previous_tasks)
                memory_dict_logits = self.memory_model.forward_clf(x, self.previous_tasks)
                temp = 2.0
                teacher_probs = {k: F.softmax(v / temp, 1) for k, v in memory_dict_logits.items()}
                student_log_probs  = {k: F.log_softmax(v / temp, 1) for k, v in logits_dict.items()}
                # logits_dict = {k: (self.model.clf_task(x, k) if k not in labels_dict.keys() else logits_dict[k]) for k in self.previous_tasks}
                # memory_dict_logits = {k: self.memory_model.clf_task(x, k) for k in self.previous_tasks}
                memory_loss_dict = {
                    k: F.kl_div(student_log_probs[k] , teacher_probs[k], reduction='batchmean') * (temp**2) for k
                    in self.previous_tasks}
                memory_loss = torch.stack(list(memory_loss_dict.values())).mean()
                self.log("train/memory_loss", memory_loss, prog_bar=True)
                memory_loss = self.lambda_dctn * memory_loss
            if self.has_memories:
                self.memory_replay()
                ewc_loss = self.get_ewc_loss()
                self.log("train/ewc_loss", ewc_loss.item(), prog_bar=True)
                memory_loss += self.lambda_ewc * ewc_loss

        # Add edge loss to total loss with a weighting factor
        lambda_edge = self.hparams.get("lambda_edge", 0.05)  # Add this to your config
        total_loss += memory_loss + feature_loss * self.lambda_featl + edge_loss * lambda_edge
        
        self.log("train/total_loss", total_loss.item(), prog_bar=True)
        self.log("train/feature_loss", feature_loss.item(), prog_bar=True)

        for k in loss_dict.keys():
            self.log(f"train/{k}_loss", loss_dict[k].item())

        return total_loss

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            combined_batch = batch
            loss = 0
            for gtask_key, batch in combined_batch.items():
                if batch is None:
                    continue

                loss += self.common_step(batch)

            return loss / len(combined_batch.keys())
        else:
            loss = self.common_step(batch)
            return loss

    def validation_step(self, combined_batch, batch_idx):
        # combined_batch = combined_batch if isinstance(combined_batch, dict) else {self.current_task: combined_batch}
        for k, batch in combined_batch.items():
            if k not in self.current_val_tasks:
                continue
            if batch is None:
                continue
            x_dict = batch.x_dict
            batch_size = batch["note"].batch_size
            labels_dict = {k: batch["note"][k][:batch_size] for k in self.task_dict.keys() if k in batch["note"].keys()}
            pitch_spelling = batch["note"].pitch_spelling
            key_signature = batch["note"].key_signature
            # limit labels to the num_labels values of the self.task_dict and set the rest to 0
            labels_dict = {
                k: torch.where(labels_dict[k] < self.task_dict[k], labels_dict[k], torch.zeros_like(labels_dict[k])) for
                k in labels_dict.keys()}
            edge_index_dict = batch.edge_index_dict
            batch_dict = batch.batch_dict

            num_sampled_edges_dict = batch.num_sampled_edges_dict
            num_sampled_nodes_dict = batch.num_sampled_nodes_dict
            mask_dict = self.create_mask_dict(labels_dict, batch, batch_size)

            # NOTE: mask to remove invalid labels
            if "valid_label" not in batch["note"].keys():
                valid_label_mask = torch.ones_like(batch["note"]["pitch_spelling"][:batch_size]).bool()
            else:
                valid_label_mask = batch["note"]["valid_label"][:batch_size].bool()

            labels_dict = {k: v[valid_label_mask] for k, v in labels_dict.items()}
            mask_dict = {k: v[valid_label_mask] for k, v in mask_dict.items()}
            labels_dict = {k: v[mask_dict[k]] for k, v in labels_dict.items()}

            logits_dict = self.model(
                    pitch_spelling=pitch_spelling,
                    key_signature=key_signature,
                    x_dict=x_dict,
                    edge_index_dict=edge_index_dict,
                    batch_dict=batch_dict,
                    batch_size=batch_size,
                    neighbor_mask_node=num_sampled_nodes_dict, neighbor_mask_edge=num_sampled_edges_dict
                )
            logits_dict = {k: v[valid_label_mask] for k, v in logits_dict.items()}
            logits_dict = {k: (v[mask_dict[k]] if k in mask_dict.keys() else v) for k, v in logits_dict.items()}
            loss_dict = self.clf_loss(logits_dict, labels_dict)
            total_loss = loss_dict.pop("total") / len(labels_dict.keys())
            accuracy_dict = {k: self.accuracy_dict[k](logits_dict[k], labels_dict[k]) for k in labels_dict.keys()}
            f1_dict = {k: self.f1_dict[k](logits_dict[k], labels_dict[k]) for k in labels_dict.keys()}
            self.log("val/total_loss", total_loss.item(), batch_size=batch_size, prog_bar=True)

            for k in loss_dict.keys():
                self.log(f"val/{k}_loss", loss_dict[k].item(), batch_size=batch_size)
                self.log(f"val/{k}_acc", accuracy_dict[k], batch_size=batch_size)
                self.log(f"val/{k}_f1", f1_dict[k], batch_size=batch_size)

            # RNA accuracy calculation based on in_label notes only
            if "tpc_in_label" in logits_dict.keys():
                rna_keys = ["quality", "inversion", "degree1", "degree2", "localkey"]
                mask = logits_dict["tpc_in_label"].argmax(-1).bool()
                rna_acc_total = {}
                if all([k in labels_dict.keys() for k in rna_keys]):
                    for k in rna_keys:                        
                        rna_acc = self.accuracy_dict[k](logits_dict[k][mask], labels_dict[k][mask])
                        rna_acc_total[k] = logits_dict[k][mask].argmax(-1).eq(labels_dict[k][mask])
                        self.log(f"val/NCT_{k}_acc", rna_acc, batch_size=batch_size)
                    # stack and get accuracy over all rna keys
                    rna_acc_total = torch.stack(list(rna_acc_total.values())).all(dim=0).float().mean()
                    self.log("val/total_rna_acc", rna_acc_total, batch_size=batch_size)

    def on_validation_epoch_end(self):
        # if the epoch % self.total_epochs // 3 == 0, change the task
        if self.cl_training:
            i = self.main_tasks.index(self.current_task)
            if self.current_epoch == sum(self.epochs_per_task[:i+1]):
                if i == len(self.main_tasks) - 1:
                    next_task = None
                else:
                    next_task = self.main_tasks[i + 1]

                self.trainer.save_checkpoint(f"{self.trainer.checkpoint_callback.dirpath}/{self.current_task}_model_epoch={self.trainer.current_epoch}.ckpt")
                if next_task:
                    self.set_task(next_task)
                    self.current_val_tasks.append(next_task)
                    print(f"Changing Task to {next_task} \n")
                else:
                    print("All Tasks have been processed")

    def test_step(self, combined_batch, batch_idx) -> STEP_OUTPUT:
        for gtask_key, batch in combined_batch.items():
            if batch is None:
                print("Batch is None")
                continue
            x_dict = batch.x_dict
            labels_dict = {k: batch["note"][k] for k in self.task_dict.keys() if k in batch["note"].keys()}
            pitch_spelling = batch["note"].pitch_spelling
            key_signature = batch["note"].key_signature
            # limit labels to the num_labels values of the self.task_dict and set the rest to 0
            labels_dict = {
                k: torch.where(labels_dict[k] < self.task_dict[k], labels_dict[k], torch.zeros_like(labels_dict[k])) for
                k in labels_dict.keys()}
            edge_index_dict = batch.edge_index_dict
            batch_dict = batch.batch_dict
            batch_size = batch["note"].batch_size
            num_sampled_edges_dict = batch.num_sampled_edges_dict
            num_sampled_nodes_dict = batch.num_sampled_nodes_dict
            # NOTE: mask to remove invalid labels
            if "valid_label" not in batch["note"].keys():
                valid_label_mask = torch.ones_like(batch["note"]["pitch_spelling"][:batch_size]).bool()
            else:
                valid_label_mask = batch["note"]["valid_label"][:batch_size].bool()
            labels_dict = {k: v[:batch_size][valid_label_mask] for k, v in labels_dict.items()}
            logits_dict = self.model(
                pitch_spelling=pitch_spelling,
                key_signature=key_signature,
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                batch_dict=batch_dict,
                batch_size=batch_size,
                neighbor_mask_node=num_sampled_nodes_dict, neighbor_mask_edge=num_sampled_edges_dict
            )
            logits_softmax_dict = {k: v.softmax(-1) for k, v in logits_dict.items()}
            logits_dict = {k: v[valid_label_mask] for k, v in logits_dict.items()}
            loss_dict = self.clf_loss(logits_dict, labels_dict)
            total_loss = loss_dict.pop("total") / len(labels_dict.keys())
            accuracy_dict = {k: self.accuracy_dict[k](logits_dict[k], labels_dict[k]) for k in labels_dict.keys()}
            f1_dict = {k: self.f1_dict[k](logits_dict[k], labels_dict[k]) for k in labels_dict.keys()}

            self.log("test/total_loss", total_loss.item(), add_dataloader_idx=True, batch_size=batch_size, prog_bar=True)
            # RNA calculation Onsetwise
            rna_keys = ["quality", "inversion", "degree1", "degree2"] # ["localkey", "quality", "inversion", "degree1", "degree2"]
            if all([k in labels_dict.keys() for k in rna_keys]):
                # NOTE: Aggregate per onset
                onset_edges = edge_index_dict["note", "onset", "note"]
                onset_edge_mask_src = onset_edges[0] < batch_size
                onset_edge_mask_dst = onset_edges[1] < batch_size
                onset_edges = onset_edges[:, torch.logical_and(onset_edge_mask_src, onset_edge_mask_dst)]
                # remove self loops
                onset_edges = onset_edges[:, onset_edges[0] != onset_edges[1]]
                # aggregate the logit predictions based on the onset edges
                aggregate_logit_dict = {}
                for k, v in logits_softmax_dict.items():
                    if k in rna_keys:
                        aggregate_logit_dict[k] = torch_scatter.scatter_mean(v[onset_edges[0]], onset_edges[1], dim=0, out=v).softmax(-1)
                # keep valid labels
                aggregate_logit_dict = {k: v[valid_label_mask] for k, v in aggregate_logit_dict.items()}
                onsets = batch["note"].onset_div[:batch_size][valid_label_mask]
                onsets = onsets - onsets.min()
                batch_id = batch["note"].batch[:batch_size][valid_label_mask]
                # map tuple to unique int
                cantor_pair = (onsets + batch_id) * (onsets + batch_id + 1) // 2 + batch_id
                # Find unique onsets and their inverse indices
                unique, inverse = torch.unique(cantor_pair, sorted=True, return_inverse=True)
                # Create a range of indices with the same size as inverse
                perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
                # Reverse the order of inverse and perm
                inverse, perm = inverse.flip([0]), perm.flip([0])
                # Scatter perm into a new tensor of the same size as unique using inverse as indices
                perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
                unique_logit_map = perm.sort().values
                onsetwise_logit_dict = {k: v[unique_logit_map] for k, v in aggregate_logit_dict.items()}
                onsetwise_label_dict = {k: v[unique_logit_map] for k, v in labels_dict.items()}
                # RNA calculation
                rna_labels = {k: onsetwise_label_dict[k] for k in rna_keys}
                rna_preds = {k: onsetwise_logit_dict[k].argmax(-1) for k in rna_keys}
                # rna_accuracy is the logical and from the comparison of all the rna keys
                rna_accuracy = (torch.stack([rna_preds[k] == rna_labels[k] for k in rna_keys]).t().float().mean(
                    -1) == 1).float().mean()
                self.log(f"test/RN(Onset)_{gtask_key}_accuracy", rna_accuracy.item(), add_dataloader_idx=True, batch_size=batch_size)


            for k in labels_dict.keys():
                self.log(f"test/{k}_{gtask_key}_acc", accuracy_dict[k], add_dataloader_idx=True, batch_size=batch_size)
                self.log(f"test/{k}_{gtask_key}_f1", f1_dict[k], add_dataloader_idx=True, batch_size=batch_size)

            # RNA accuracy calculation based on in_label notes only
            if "tpc_in_label" in logits_dict.keys():
                rna_keys = ["quality", "inversion", "degree1", "degree2", "localkey"]
                mask = logits_dict["tpc_in_label"].argmax(-1).bool()
                if all([k in labels_dict.keys() for k in rna_keys]):
                    for k in rna_keys:
                        rna_acc = self.accuracy_dict[k](logits_dict[k][mask], labels_dict[k][mask])
                        self.log(f"test/NCT_{k}_{gtask_key}_acc", rna_acc, batch_size=batch_size)

                    rna_accuracy = (torch.stack([logits_dict[k][mask].argmax(-1) == labels_dict[k][mask] for k in rna_keys]).t().float().mean(
                        -1) == 1).float().mean()
                    self.log(f"test/RN(NCT)_{gtask_key}_accuracy", rna_accuracy.item(), add_dataloader_idx=True, batch_size=batch_size)

    def predict_step(self, batch, batch_idx):
        x_dict = batch.x_dict
        pitch_spelling = batch["note"].pitch_spelling
        key_signature = batch["note"].key_signature
        edge_index_dict = batch.edge_index_dict
        batch_dict = batch.batch_dict
        batch_size = batch["note"].batch_size
        num_sampled_edges_dict = batch.num_sampled_edges_dict
        num_sampled_nodes_dict = batch.num_sampled_nodes_dict
        logits_dict = self.model(
            pitch_spelling=pitch_spelling,
            key_signature=key_signature,
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            batch_dict=batch_dict,
            batch_size=batch_size,
            neighbor_mask_node=num_sampled_nodes_dict, neighbor_mask_edge=num_sampled_edges_dict
        )
        logits_softmax_dict = {k: v.softmax(-1) for k, v in logits_dict.items()}
        preds = {task: logits_softmax_dict[task].argmax(-1) for task in self.task_dict.keys()}
        decoded_labels = {task: available_representations[task].decode(preds[task].reshape(-1, 1).cpu().numpy()) for task in preds.keys() if task in available_representations.keys()}
        return preds, decoded_labels

    def set_task(self, task):
        if self.has_memories:
            self.get_optimal_params()
            self.memory_replay()

        if self.current_task != task and self.current_task is not None:
            if self.current_task in self.task_dict.keys():
                self.previous_tasks.append(self.current_task)
            elif self.current_task == "rna":
                for t in ["localkey", "tonkey", "quality", "root", "bass", "inversion", "degree1", "degree2"]:
                    self.previous_tasks.append(t)
            elif self.current_task == "all":
                for t in self.task_dict.keys():
                    self.previous_tasks.append(t)

        self.current_task = task
        if self.lambda_dctn > 0:
            self.update_memory_model()

    @torch.enable_grad()
    def memory_replay(self):
        self.model.train()
        dataloaders = self.trainer.val_dataloaders
        count_dataloaders = len(self.current_val_tasks)
        for task in self.current_val_tasks:
            if task == self.current_task:
                continue
            dataloader = dataloaders[task]
            length_dataloader = len(dataloader)
            batch = next(iter(dataloader))
            self.model.zero_grad()
            self.zero_grad()
            batch = batch.to(self.device)
            x_dict = batch.x_dict
            labels_dict = {k: batch["note"][k] for k in self.task_dict.keys() if k in batch["note"].keys()}
            pitch_spelling = batch["note"].pitch_spelling
            key_signature = batch["note"].key_signature
            # limit labels to the num_labels values of the self.task_dict and set the rest to 0
            labels_dict = {
                k: torch.where(labels_dict[k] < self.task_dict[k], labels_dict[k], torch.zeros_like(labels_dict[k]))
                for
                k in labels_dict.keys()}
            edge_index_dict = batch.edge_index_dict
            batch_dict = batch.batch_dict
            batch_size = batch["note"].batch_size
            num_sampled_edges_dict = batch.num_sampled_edges_dict
            num_sampled_nodes_dict = batch.num_sampled_nodes_dict
            labels_dict = {k: v[:batch_size] for k, v in labels_dict.items()}
            logits_dict = self.model(
                pitch_spelling=pitch_spelling,
                key_signature=key_signature,
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                batch_dict=batch_dict,
                batch_size=batch_size,
                neighbor_mask_node=num_sampled_nodes_dict, neighbor_mask_edge=num_sampled_edges_dict
            )
            loss_dict = self.clf_loss(logits_dict, labels_dict)
            loss = loss_dict["total"]
            loss.backward()
            self.compute_fisher(count_dataloaders)
        self.zero_grad()

    def update_memory_model(self):
        state_dict = deepcopy(self.model.state_dict())
        self.memory_model.load_state_dict(state_dict)
        self.freeze_memory_model()
        self.memory_model.eval()

    def freeze_memory_model(self):
        for param in self.memory_model.parameters():
            param.requires_grad = False

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Calculate total training steps for better scheduler configuration
        if hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            # Fallback calculation
            total_steps = self.total_epochs * 5000  # Rough estimate, adjust based on your data
        
        warmup_steps = min(500, total_steps // 20)  # 5% of total steps or 500, whichever is smaller
        
        # Use the LinearWarmupCosineAnnealingLR for better performance
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, 
            warmup_steps=warmup_steps, 
            max_epochs=self.total_epochs,
            eta_min=self.lr * 0.01,  # Lower minimum for better convergence
            last_epoch=-1
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val/total_loss",  # Monitor validation loss instead of training
                "strict": False,
            }
        }

    def update_feature_loss(self, feature_loss, x_over, y_over, x, y, batch_size=100):
        """
        Penalize when distance is too large between original and synthetic samples of the same class
        Calculate Euclidean distance between synthetic and original samples
        """
        unique_cls_labels = y_over.unique()
        threshold = 1.0  # Set your own threshold
        bs = batch_size // len(unique_cls_labels)
        if isinstance(bs, torch.Tensor):
            bs = bs.item()
        for class_label in unique_cls_labels:
            mask = y_over == class_label
            x_over_class = x_over[mask]
            x_class = x[y == class_label]
            # Sample a few points from x_class and x_over to reduce computational cost
            if len(x_class) > bs:
                perm = torch.randperm(len(x_class)).to(x.device)
                x_class = x_class[perm[:bs]]
            if len(x_over_class) > bs:
                perm = torch.randperm(len(x_over_class)).to(x.device)
                x_over_class = x_over_class[perm[:bs]]
            distances = torch.cdist(x_over_class, x_class)
            min_distances, _ = torch.min(distances, dim=1)
            # Add penalty if distance is too large
            penalties = torch.clamp(min_distances - threshold, min=0)
            feature_loss += penalties.mean()
        return feature_loss

    def compute_fisher(self, len_dataloader):
        """
        Computes an approximation of the Fisher Information matrix.

        Args:
            model (torch.nn.Module): The model trained on the previous task.
            dataloader (torch.utils.data.DataLoader): DataLoader for the previous task.
            criterion: Loss function used in training.

        Returns:
            dict: A dictionary mapping parameter names to their Fisher information.
        """
        # Accumulate squared gradients
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                self.fisher[n] += p.grad.data.clone().pow(2) / len_dataloader

    def _init_fisher(self):
        self.fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

    def get_optimal_params(self):
        """
        Stores a copy of the current model parameters.

        Args:
            model (torch.nn.Module): The trained model.

        Returns:
            dict: A dictionary mapping parameter names to their current values.
        """

        optimal_params = {}
        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        for n, p in deepcopy(params).items():
            optimal_params[n] = Variable(p.data)
        self._means = optimal_params
        self._init_fisher()


    def get_ewc_loss(self):
        """
        Computes the total loss with the EWC penalty.

        Args:
            fisher (dict): Fisher information for each parameter.
            opt_params (dict): The optimal parameters from the previous task.

        Returns:
            torch.Tensor: Total loss with the EWC regularization term.
        """
        ewc_penalty = 0
        for n, p in self.model.named_parameters():
            # Accumulate the penalty for each parameter
            ewc_penalty += (self.fisher[n] * (p - self._means[n]).pow(2)).sum()

        return ewc_penalty

    def predict(self, score):
        """Predict analysis for a musical score.
        
        Args:
            score: Path to score file or partitura Score object
            
        Returns:
            Dictionary of predictions for each task
        """
        import tempfile
        import os
        import partitura as pt
        from analysisgnn.descriptors import select_features
        from graphmuse import create_score_graph
        from analysisgnn.utils.music import PitchEncoder, KeySignatureEncoder
        import numpy as np
        
        # Handle both score objects and file paths
        if isinstance(score, str):
            # It's a file path, load it
            score_obj = pt.load_score(score)
            score_name = os.path.splitext(os.path.basename(score))[0]
        else:
            # It's already a partitura score object
            score_obj = score
            score_name = "unknown_score"
        
        # Process the score directly without saving to file
        try:
            # Get the note array with all required features
            note_array = score_obj.note_array(
                include_time_signature=True, 
                include_pitch_spelling=True,
                include_key_signature=True, 
                include_staff=True, 
                include_metrical_position=True
            )
            note_array = np.sort(note_array, order=["onset_div", "pitch"])
            
            # Get measures and part
            measures = score_obj[-1].measures
            part = score_obj[-1]
            
            # Select features (using default "voice" feature type)
            note_features = select_features(note_array, "voice")
            
            # Create graph data
            data = create_score_graph(note_features, note_array, measures=measures, add_beats=True, labels=None)
            
            # Add pitch spelling and key signature encodings
            pitch_encoder = PitchEncoder()
            ks_encoder = KeySignatureEncoder()
            labels_ps = pitch_encoder.encode(note_array)
            labels_ks = ks_encoder.encode(note_array)
            
            data["note"].pitch_spelling = torch.from_numpy(labels_ps).long()
            data["note"].key_signature = torch.from_numpy(labels_ks).long()
            data["note"].voice = torch.from_numpy(note_array["voice"]).long()
            data["note"].staff = torch.from_numpy(note_array["staff"]).long()
            
            # Add batch information for single score (all nodes belong to batch 0)
            batch_size = data["note"].x.size(0)
            data["note"].batch = torch.zeros(batch_size, dtype=torch.long)
            
            # Convert to the format expected by the model
            x_dict = data.x_dict
            edge_index_dict = data.edge_index_dict
            batch_dict = data.batch_dict
            pitch_spelling = data["note"].pitch_spelling
            key_signature = data["note"].key_signature
            
            # Create dummy masks for single score prediction
            # For prediction, we don't use sampling, so set to None
            num_sampled_nodes_dict = None
            num_sampled_edges_dict = None
            
            # Get predictions from the model
            logits_dict = self.model(
                pitch_spelling=pitch_spelling,
                key_signature=key_signature,
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                batch_dict=batch_dict,
                batch_size=batch_size,
                neighbor_mask_node=num_sampled_nodes_dict,
                neighbor_mask_edge=num_sampled_edges_dict
            )
            
            # Convert logits to probabilities
            predictions = {k: torch.softmax(v, dim=-1) for k, v in logits_dict.items()}
            
            return predictions
            
        except Exception as e:
            raise ValueError(f"Failed to process score: {str(e)}")

