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
from typing import List, Union, Dict, Any, Optional
from analysisgnn.models.chord import MultiTaskLoss
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
import math
import warnings
from analysisgnn.utils.chord_representations import available_representations
from analysisgnn.utils.masked_conditioning import MaskedConditioningSpec
from analysisgnn.utils.node_masking import (
    create_node_mask,
    sample_context_indices,
    split_nodes_by_mask,
    clamp_logits_to_labels,
)


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


class PCGrad:
    """Projected Conflicting Gradient (PCGrad) optimizer wrapper."""

    def __init__(
        self,
        optimizer: Optional[Optimizer] = None,
        parameters: Optional[List[torch.nn.Parameter]] = None,
    ) -> None:
        self.optimizer = optimizer
        self.parameters = parameters

    def _params(self):
        if self.parameters is not None:
            return [p for p in self.parameters if p.requires_grad]
        if self.optimizer is None:
            return []
        params = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)
        return params

    def pc_backward(self, losses: List[torch.Tensor], retain_graph: bool = False) -> None:
        params = self._params()
        if not params or not losses:
            return
        grads: List[List[Optional[torch.Tensor]]] = []
        for i, loss in enumerate(losses):
            keep_graph = retain_graph or (i < len(losses) - 1)
            grad = torch.autograd.grad(
                loss,
                params,
                retain_graph=keep_graph,
                allow_unused=True,
            )
            grads.append([g for g in grad])

        # Project conflicting gradients.
        for i in range(len(grads)):
            for j in range(len(grads)):
                if i == j:
                    continue
                gij = 0.0
                for gi, gj in zip(grads[i], grads[j]):
                    if gi is None or gj is None:
                        continue
                    gij = gij + (gi * gj).sum()
                if gij < 0:
                    gj_norm_sq = 0.0
                    for gj in grads[j]:
                        if gj is None:
                            continue
                        gj_norm_sq = gj_norm_sq + (gj ** 2).sum()
                    gj_norm_sq = gj_norm_sq + 1e-12
                    coeff = gij / gj_norm_sq
                    projected = []
                    for gi, gj in zip(grads[i], grads[j]):
                        if gi is None:
                            projected.append(None)
                        elif gj is None:
                            projected.append(gi)
                        else:
                            projected.append(gi - coeff * gj)
                    grads[i] = projected

        final_grads = []
        for k in range(len(params)):
            grad_k = None
            for task_grads in grads:
                g = task_grads[k]
                if g is None:
                    continue
                grad_k = g if grad_k is None else (grad_k + g)
            final_grads.append(grad_k)

        for p, g in zip(params, final_grads):
            if g is None:
                continue
            if p.grad is None:
                p.grad = g.detach()
            else:
                p.grad = p.grad + g.detach()


def onsetwise_logit_aggregation(logits_softmax_dict, graph, edge_index_dict=None, batch_size=None, valid_label_mask=None, rna_keys=["cadence", "phrase", "root", "localkey", "quality", "inversion", "degree1", "degree2", "romanNumeral", "section"]):        
    if all([k in logits_softmax_dict.keys() for k in rna_keys]) and rna_keys:
        batch_size = len(graph["note"].x) if batch_size is None else batch_size
        edge_index_dict = graph.edge_index_dict if edge_index_dict is None else edge_index_dict
        valid_label_mask = torch.ones(batch_size, dtype=torch.bool).to(graph["note"].x.device) if valid_label_mask is None else valid_label_mask
        # NOTE: Aggregate per onset
        onset_edges = edge_index_dict["note", "onset", "note"]
        onset_edge_mask_src = onset_edges[0] < batch_size
        onset_edge_mask_dst = onset_edges[1] < batch_size
        onset_edges = onset_edges[:, torch.logical_and(onset_edge_mask_src, onset_edge_mask_dst)]
        # remove self loops
        onset_edges = onset_edges[:, onset_edges[0] != onset_edges[1]]
        # If tpc_in_label is in logits_softmax_dict make a mask out of argmax
        if "tpc_in_label" in logits_softmax_dict:
            tpc_in_label_mask = logits_softmax_dict["tpc_in_label"].argmax(-1).bool()
            onset_edges = onset_edges[:, tpc_in_label_mask[onset_edges[0]] & tpc_in_label_mask[onset_edges[1]]]
        else:
            tpc_in_label_mask = None
        # aggregate the logit predictions based on the onset edges
        aggregate_logit_dict = {}
        for k, v in logits_softmax_dict.items():
            if k in rna_keys:
                aggregate_logit_dict[k] = torch_scatter.scatter_mean(v[onset_edges[0]], onset_edges[1], dim=0, out=v).softmax(-1)
        # keep valid labels
        aggregate_logit_dict = {k: v[valid_label_mask].softmax(-1) for k, v in aggregate_logit_dict.items()}
        logits_softmax_dict.update(aggregate_logit_dict)
        batch_id = graph["note"].batch[:batch_size][valid_label_mask]
        if torch.all(batch_id == batch_id[0]):                            
            onsets = graph["note"].onset_div[:batch_size][valid_label_mask]
            onsets = onsets - onsets.min()
            if tpc_in_label_mask is not None:
                onsets_filtered = onsets[tpc_in_label_mask]
                aggregate_logit_dict = {k: v[tpc_in_label_mask] for k, v in aggregate_logit_dict.items()}
            else:
                onsets_filtered = onsets
            unique_onset_values, un_onset_indices = torch.unique(onsets_filtered, return_inverse=True)
            unique_logit_map = (un_onset_indices[1:] != un_onset_indices[:-1]).nonzero(as_tuple=True)[0] + 1
            unique_logit_map = torch.cat([torch.tensor([0], device=unique_logit_map.device), unique_logit_map])
            onsetwise_logit_dict = {k: v[unique_logit_map] for k, v in aggregate_logit_dict.items()}
            # RNA calculation
            rna_preds = {k: onsetwise_logit_dict[k].argmax(-1) for k in rna_keys}            
            # find unique onsets where the predictions change
            for k in rna_preds.keys():
                x = rna_preds[k]
                # assume that x is in order. Find in which i+1 != i
                change_points = (x[1:] != x[:-1]).nonzero(as_tuple=True)[0] + 1
                # Add 0 as the first change point
                change_points = torch.cat([torch.tensor([0], device=change_points.device), change_points])
                # Update the logits_softmax_dict with the new logits
                onsets_value_on_change = unique_onset_values[change_points]
                x_on_change = x[change_points]
                x_logits_on_change = onsetwise_logit_dict[k][change_points]
                # find indices of onsets that are between change points and assign them to the same logits
                for i in range(len(change_points) - 1):
                    onset_mask = (onsets_value_on_change[i] <= onsets) & (onsets < onsets_value_on_change[i + 1])
                    logits_softmax_dict[k][onset_mask] = x_logits_on_change[i]

    return logits_softmax_dict


def beatwise_logit_aggregation(logits_softmax_dict, graph, edge_index_dict=None, batch_size=None, valid_label_mask=None, rna_keys=["root", "localkey", "quality", "inversion", "degree1", "degree2", "romanNumeral", "cadence", "phrase", "section"]):        
    if all([k in logits_softmax_dict.keys() for k in rna_keys]) and rna_keys:
        batch_size = len(graph["note"].x) if batch_size is None else batch_size
        edge_index_dict = graph.edge_index_dict if edge_index_dict is None else edge_index_dict
        valid_label_mask = torch.ones(batch_size, dtype=torch.bool).to(graph["note"].x.device) if valid_label_mask is None else valid_label_mask
        # NOTE: Aggregate per beat
        beat_edges_out = edge_index_dict["beat", "connects", "note"]
        beat_edges_in = edge_index_dict["note", "connects", "beat"]
        # find number of beats from beat_edges_in and beat_edges_out
        num_beats = max(beat_edges_out[0].max(), beat_edges_in[1].max()) + 1
        beat_edge_mask_src = beat_edges_out[1] < batch_size
        beat_edge_mask_dst = beat_edges_out[0] < batch_size
        beat_edges_out = beat_edges_out[:, beat_edge_mask_dst]
        beat_edges_in = beat_edges_in[:, beat_edge_mask_src]
        # If tpc_in_label is in logits_softmax_dict make a mask out of argmax
        if "tpc_in_label" in logits_softmax_dict:
            tpc_in_label_mask = logits_softmax_dict["tpc_in_label"].argmax(-1).bool()
            beat_edges_out = beat_edges_out[:, tpc_in_label_mask[beat_edges_out[1]]]
            beat_edges_in = beat_edges_in[:, tpc_in_label_mask[beat_edges_in[0]]]
        else:
            tpc_in_label_mask = None
        # aggregate the logit predictions based on the onset edges
        aggregate_logit_dict = {}
        for k, v in logits_softmax_dict.items():
            if k in rna_keys:
                # create a tensor of size (num_beats, num_classes) to store the aggregated logits
                beat_logits = torch.zeros((num_beats, v.size(-1)), device=v.device)
                # aggregate logits from notes to beats
                beat_logits = torch_scatter.scatter_mean(v[beat_edges_out[1]], beat_edges_out[0], dim=0, dim_size=num_beats, out=beat_logits)
                # distribute back to notes
                aggregate_logit_dict[k] = torch_scatter.scatter_mean(beat_logits[beat_edges_in[1]], beat_edges_in[0], dim=0, out=v).softmax(-1)
    return logits_softmax_dict


def measurewise_logit_aggregation(logits_softmax_dict, graph, edge_index_dict=None, batch_size=None, valid_label_mask=None, rna_keys=["localkey"]):
    if all([k in logits_softmax_dict.keys() for k in rna_keys]) and rna_keys:
        batch_size = len(graph["note"].x) if batch_size is None else batch_size
        edge_index_dict = graph.edge_index_dict if edge_index_dict is None else edge_index_dict
        valid_label_mask = torch.ones(batch_size, dtype=torch.bool).to(graph["note"].x.device) if valid_label_mask is None else valid_label_mask
        # NOTE: Aggregate per measure
        measure_edges_out = edge_index_dict["measure", "connects", "note"]
        measure_edges_in = edge_index_dict["note", "connects", "measure"]
        # find number of measures from measure_edges_in and measure_edges_out
        num_measures = max(measure_edges_out[0].max(), measure_edges_in[1].max()) + 1
        measure_edge_mask_src = measure_edges_out[1] < batch_size
        measure_edge_mask_dst = measure_edges_out[0] < batch_size
        measure_edges_out = measure_edges_out[:, measure_edge_mask_dst]
        measure_edges_in = measure_edges_in[:, measure_edge_mask_src]
        # If tpc_in_label is in logits_softmax_dict make a mask out of argmax
        if "tpc_in_label" in logits_softmax_dict:
            tpc_in_label_mask = logits_softmax_dict["tpc_in_label"].argmax(-1).bool()
            measure_edges_out = measure_edges_out[:, tpc_in_label_mask[measure_edges_out[1]]]
            measure_edges_in = measure_edges_in[:, tpc_in_label_mask[measure_edges_in[0]]]
        else:
            tpc_in_label_mask = None
        # aggregate the logit predictions based on the onset edges
        aggregate_logit_dict = {}
        for k, v in logits_softmax_dict.items():
            if k in rna_keys:
                # create a tensor of size (num_measures, num_classes) to store the aggregated logits
                measure_logits = torch.zeros((num_measures, v.size(-1)), device=v.device)
                # aggregate logits from notes to measures
                measure_logits = torch_scatter.scatter_mean(v[measure_edges_out[1]], measure_edges_out[0], dim=0, dim_size=num_measures, out=measure_logits)                
                # distribute back to notes
                aggregate_logit_dict[k] = torch_scatter.scatter_mean(measure_logits[measure_edges_in[1]], measure_edges_in[0], dim=0, out=v).softmax(-1)
        return logits_softmax_dict


        # # keep valid labels
        # aggregate_logit_dict = {k: v[valid_label_mask].softmax(-1) for k, v in aggregate_logit_dict.items()}
        # logits_softmax_dict.update(aggregate_logit_dict)
        # batch_id = graph["note"].batch[:batch_size][valid_label_mask]
        # if torch.all(batch_id == batch_id[0]):                            
        #     onsets = graph["note"].onset_div[:batch_size][valid_label_mask]
        #     onsets = onsets - onsets.min()
        #     if tpc_in_label_mask is not None:
        #         onsets_filtered = onsets[tpc_in_label_mask]
        #         aggregate_logit_dict = {k: v[tpc_in_label_mask] for k, v in aggregate_logit_dict.items()}
        #     else:
        #         onsets_filtered = onsets
        #     unique_onset_values, un_onset_indices = torch.unique(onsets_filtered, return_inverse=True)
        #     unique_logit_map = (un_onset_indices[1:] != un_onset_indices[:-1]).nonzero(as_tuple=True)[0] + 1
        #     unique_logit_map = torch.cat([torch.tensor([0], device=unique_logit_map.device), unique_logit_map])
        #     onsetwise_logit_dict = {k: v[unique_logit_map] for k, v in aggregate_logit_dict.items()}
        #     # RNA calculation
        #     rna_preds = {k: onsetwise_logit_dict[k].argmax(-1) for k in rna_keys}            
        #     # find unique onsets where the predictions change
        #     for k in rna_preds.keys():
        #         x = rna_preds[k]
        #         # assume that x is in order. Find in which i+1 != i
        #         change_points = (x[1:] != x[:-1]).nonzero(as_tuple=True)[0] + 1
        #         # Add 0 as the first change point
        #         change_points = torch.cat([torch.tensor([0], device=change_points.device), change_points])
        #         # Update the logits_softmax_dict with the new logits
        #         onsets_value_on_change = unique_onset_values[change_points]
        #         x_on_change = x[change_points]
        #         x_logits_on_change = onsetwise_logit_dict[k][change_points]
        #         # find indices of onsets that are between change points and assign them to the same logits
        #         for i in range(len(change_points) - 1):
        #             onset_mask = (onsets_value_on_change[i] <= onsets) & (onsets < onsets_value_on_change[i + 1])
        #             logits_softmax_dict[k][onset_mask] = x_logits_on_change[i]
    


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
        total_steps: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Maximum number of steps for linear warmup
            total_steps (int): Total number of optimization steps.
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = min(max(0, int(warmup_steps)), self.total_steps - 1)
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

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

        current_step = min(max(self.last_epoch, 0), self.total_steps)

        if current_step < self.warmup_steps:
            warmup_progress = current_step / max(1, self.warmup_steps)
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_progress
                for base_lr in self.base_lrs
            ]

        cosine_steps = max(1, self.total_steps - self.warmup_steps)
        cosine_progress = min(max((current_step - self.warmup_steps) / cosine_steps, 0.0), 1.0)
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + math.cos(math.pi * cosine_progress))
            for base_lr in self.base_lrs
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        current_step = min(max(self.last_epoch, 0), self.total_steps)
        if current_step < self.warmup_steps:
            warmup_progress = current_step / max(1, self.warmup_steps)
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_progress
                for base_lr in self.base_lrs
            ]

        cosine_steps = max(1, self.total_steps - self.warmup_steps)
        cosine_progress = min(max((current_step - self.warmup_steps) / cosine_steps, 0.0), 1.0)
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + math.cos(math.pi * cosine_progress))
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
    def __init__(
        self,
        metadata,
        in_channels,
        hidden_channels,
        out_channels,
        task_dict,
        num_layers,
        dropout=0.5,
        use_jk=True,
        logit_fusion=True,
        use_rnn=False,
        encoder_type="hybridgnn",
        use_graph_encoder=True,
    ):
        super().__init__()
        self.pitch_embedding = nn.Embedding(35, 64)
        self.key_embedding = nn.Embedding(15, 64)
        self.logit_fusion = logit_fusion
        self.use_rnn = use_rnn
        self.use_graph_encoder = use_graph_encoder
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
        # In heads-only mode we still benefit from a learned projection block
        # before onset pooling / classification.
        self.no_gnn_note_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
        )
        if self.use_graph_encoder:
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
        else:
            self.encoder = None
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

    def forward(
        self,
        pitch_spelling,
        key_signature,
        x_dict,
        edge_index_dict,
        batch_dict,
        batch_size,
        neighbor_mask_node,
        neighbor_mask_edge,
        label_context=None,
    ):
        x = self.encode(
            pitch_spelling,
            key_signature,
            x_dict,
            edge_index_dict,
            batch_dict,
            batch_size,
            neighbor_mask_node,
            neighbor_mask_edge,
            label_context=label_context,
        )
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

    def encode(
        self,
        pitch_spelling,
        key_signature,
        x_dict,
        edge_index_dict,
        batch_dict,
        batch_size,
        neighbor_mask_node,
        neighbor_mask_edge,
        label_context=None,
    ):
        # initialize all values of x_dict with zeros and size self.hidden_channels except from notes
        z_dict = {k: v.clone() for k, v in x_dict.items()}
        z_dict["note"] = torch.cat([z_dict["note"], self.pitch_embedding(pitch_spelling), self.key_embedding(key_signature)], dim=-1)
        h_dict = {k: self.project_dict[k](z_dict[k]) for k in self.project_dict.keys()}
        if isinstance(label_context, dict) and "note_bias" in label_context:
            note_bias = label_context["note_bias"]
            if note_bias is not None:
                if note_bias.shape[0] != h_dict["note"].shape[0]:
                    raise ValueError(
                        "Label conditioning note_bias node count mismatch: "
                        f"{note_bias.shape[0]} vs {h_dict['note'].shape[0]}"
                    )
                if note_bias.dtype != h_dict["note"].dtype:
                    note_bias = note_bias.to(h_dict["note"].dtype)
                h_dict["note"] = h_dict["note"] + note_bias
        if self.use_graph_encoder:
            x = self.encoder(
                x_dict=h_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict,
                batch_size=batch_size, neighbor_mask_node=neighbor_mask_node,
                neighbor_mask_edge=neighbor_mask_edge, return_edge_index=False, edge_attr_dict=None)
        else:
            # Heads-only mode: use projected note features directly (no message passing).
            x = h_dict["note"][:batch_size]
            x = self.no_gnn_note_mlp(x)
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
    def __init__(self, hparams: Dict[str, Any], note_encoder: Optional[nn.Module] = None):
        super().__init__()
        encoder_type = hparams.get("model", "hybridgnn").lower()
        use_graph_encoder = not hparams.get("disable_graph_encoder", False)
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
            encoder_type=encoder_type,
            use_graph_encoder=use_graph_encoder,
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
        self.note_encoder = note_encoder
        self.musicbert_use_cached_embeddings = hparams.get("musicbert_use_cached_embeddings", False)
        self.musicbert_fusion = hparams.get("musicbert_fusion", "replace")
        self.base_in_channels = hparams.get("base_in_channels", hparams.get("in_channels"))
        self.musicbert_hidden_size = hparams.get("musicbert_hidden_size", None)
        if self.note_encoder is not None and self.musicbert_hidden_size is None:
            try:
                self.musicbert_hidden_size = self.note_encoder.backbone.model.config.hidden_size
            except Exception:
                self.musicbert_hidden_size = None

        self.musicbert_proj = None
        self.musicbert_gate = None
        if self.note_encoder is not None and self.musicbert_fusion == "gate":
            if self.musicbert_hidden_size is None or self.base_in_channels is None:
                raise ValueError("Gate fusion requires base_in_channels and musicbert hidden size.")
            self.musicbert_proj = nn.Linear(self.musicbert_hidden_size, self.base_in_channels)
            self.musicbert_gate = nn.Sequential(
                nn.Linear(self.base_in_channels * 2, self.base_in_channels),
                nn.Sigmoid(),
            )

        self.mt_conflict_method = hparams.get("mt_conflict_method", "none")
        self.gradnorm_alpha = hparams.get("gradnorm_alpha", 1.5)
        self.grad_clip_val = float(hparams.get("grad_clip_val", 0.0))
        self.optimizer_stats_log_every_n_steps = int(hparams.get("optimizer_stats_log_every_n_steps", 50))
        self.monitor_metric = hparams.get("monitor_metric", "val/total_loss")
        self.monitor_mode = hparams.get("monitor_mode", "min")
        self.scheduler_type = hparams.get("scheduler_type", "cosine_warmup")
        self.warmup_ratio = float(hparams.get("warmup_ratio", 0.05))
        self.min_lr_ratio = float(hparams.get("min_lr_ratio", 0.02))
        self.plateau_factor = float(hparams.get("plateau_factor", 0.5))
        self.plateau_patience = int(hparams.get("plateau_patience", 6))
        self.plateau_min_lr = float(hparams.get("plateau_min_lr", 1e-6))
        self.task_list = list(self.task_dict.keys())
        self.task_to_idx = {t: i for i, t in enumerate(self.task_list)}
        self.gradnorm_weights = None
        self.initial_task_losses = {}
        self._pcgrad_shared_params_cache: Optional[List[torch.nn.Parameter]] = None
        self._pcgrad_head_params_cache: Optional[List[torch.nn.Parameter]] = None
        if self.mt_conflict_method in {"pcgrad", "gradnorm"} and self.mt_strategy != "famo":
            self.automatic_optimization = False
            if self.mt_conflict_method == "gradnorm":
                self.gradnorm_weights = nn.Parameter(torch.ones(len(self.task_list)))
        
        # Semi-supervised node masking parameters
        self.train_with_masking = hparams.get("train_with_masking", False)
        self.mask_ratio = float(hparams.get("mask_ratio", 0.15))
        # Label-conditioned masked prediction parameters.
        self.masked_prediction_train = bool(hparams.get("masked_prediction_train", False))
        self.masked_tasks = [t for t in hparams.get("masked_tasks", []) if t in self.task_dict]
        self.known_ratio = float(hparams.get("known_ratio", self.mask_ratio))
        self.mask_sampling_policy = str(hparams.get("mask_sampling_policy", "hybrid")).lower()
        self.mask_span_min_onsets = int(hparams.get("mask_span_min_onsets", 2))
        self.mask_span_max_onsets = int(hparams.get("mask_span_max_onsets", 8))
        self.constraint_mode = str(hparams.get("constraint_mode", "hard")).lower()
        self.feedback_mode = str(hparams.get("feedback_mode", "single_pass")).lower()
        self.label_condition_dropout = float(hparams.get("dropout", 0.0))
        if self.feedback_mode != "single_pass":
            warnings.warn(
                f"feedback_mode='{self.feedback_mode}' is not implemented; falling back to 'single_pass'.",
                RuntimeWarning,
            )
            self.feedback_mode = "single_pass"
        if self.constraint_mode not in {"hard", "soft"}:
            warnings.warn(
                f"constraint_mode='{self.constraint_mode}' is invalid; falling back to 'hard'.",
                RuntimeWarning,
            )
            self.constraint_mode = "hard"
        if self.masked_prediction_train and len(self.masked_tasks) == 0:
            raise ValueError("masked_prediction_train requires a non-empty masked_tasks list.")
        self.label_condition_embeddings = nn.ModuleDict()
        self.label_condition_fusion = None
        if self.masked_tasks:
            self._init_label_conditioning_modules(self.masked_tasks)
        
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
                encoder_type=encoder_type,
                use_graph_encoder=use_graph_encoder,
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

    def create_random_node_mask(self, batch_size, device):
        """
        Create random node masks for semi-supervised training.
        
        This implements BERT-style random masking where:
        - mask_ratio of nodes become context (C) nodes with down-weighted loss
        - Remaining nodes are target (T) nodes with full loss
        
        Args:
            batch_size: Number of nodes in the batch
            device: Device to create the mask on
            
        Returns:
            node_mask: Tensor of shape [batch_size] with values:
                - 1.0 for target nodes
                - 0.1 for context nodes (down-weighted)
        """
        # Random selection of context nodes
        num_context = int(batch_size * self.mask_ratio)
        all_indices = torch.randperm(batch_size, device=device)
        context_indices = all_indices[:num_context]
        target_indices = all_indices[num_context:]
        
        # Create mask
        node_mask = create_node_mask(
            num_nodes=batch_size,
            target_indices=target_indices,
            context_indices=context_indices,
            context_weight=0.1,  # Context contributes 10% of target loss
            device=device
        )
        
        return node_mask

    def _init_label_conditioning_modules(self, tasks: Optional[List[str]] = None) -> None:
        tasks = list(self.masked_tasks if tasks is None else tasks)
        hidden_size = int(self.model.hidden_channels)
        if self.label_condition_fusion is None:
            self.label_condition_fusion = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(self.label_condition_dropout),
                nn.Linear(hidden_size, hidden_size),
            )
        for task in tasks:
            if task not in self.task_dict:
                continue
            if task in self.label_condition_embeddings:
                continue
            self.label_condition_embeddings[task] = nn.Embedding(
                self.task_dict[task] + 1, hidden_size
            )

    def _create_masked_prediction_node_mask(
        self,
        batch,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        onset_div = None
        if hasattr(batch["note"], "onset_div"):
            onset_div = batch["note"].onset_div[:batch_size]
        context_indices = sample_context_indices(
            num_nodes=batch_size,
            known_ratio=self.known_ratio,
            policy=self.mask_sampling_policy,
            onset_div=onset_div,
            span_min_onsets=self.mask_span_min_onsets,
            span_max_onsets=self.mask_span_max_onsets,
            device=device,
        )
        if context_indices.numel() > 0:
            target_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            target_mask[context_indices] = False
            target_indices = torch.where(target_mask)[0]
        else:
            target_indices = torch.arange(batch_size, device=device, dtype=torch.long)
        return create_node_mask(
            num_nodes=batch_size,
            target_indices=target_indices,
            context_indices=context_indices if context_indices.numel() > 0 else None,
            context_weight=0.1,
            device=device,
        )

    def _get_node_mask_for_batch(
        self,
        batch,
        batch_size: int,
        device: torch.device,
        allow_sampling: bool = True,
    ) -> Optional[torch.Tensor]:
        node_mask_from_batch = (
            batch["note"].node_mask[:batch_size]
            if hasattr(batch["note"], "node_mask") and batch["note"].node_mask is not None
            else None
        )
        if node_mask_from_batch is not None:
            if not isinstance(node_mask_from_batch, torch.Tensor):
                node_mask_from_batch = torch.tensor(node_mask_from_batch, device=device)
            return node_mask_from_batch.to(device=device, dtype=torch.float32)

        if not allow_sampling:
            return None
        if self.masked_prediction_train:
            return self._create_masked_prediction_node_mask(batch, batch_size, device)
        if self.train_with_masking and self.training:
            return self.create_random_node_mask(batch_size=batch_size, device=device)
        return None

    def _build_batch_masked_conditioning(
        self,
        labels_dict: Dict[str, torch.Tensor],
        node_mask: Optional[torch.Tensor],
        batch_size: int,
        total_nodes: int,
        device: torch.device,
    ) -> Optional[MaskedConditioningSpec]:
        if not self.masked_prediction_train:
            return None
        if not self.masked_tasks:
            return None
        full_node_mask = None
        if node_mask is not None:
            full_node_mask = torch.zeros(total_nodes, dtype=torch.float32, device=device)
            full_node_mask[:batch_size] = node_mask[:batch_size].to(device=device, dtype=torch.float32)

        if node_mask is None:
            context_indices = torch.zeros(0, dtype=torch.long, device=device)
        else:
            _, context_indices, _ = split_nodes_by_mask(node_mask[:batch_size])

        known_labels_by_task: Dict[str, torch.Tensor] = {}
        known_indices_by_task: Dict[str, torch.Tensor] = {}
        effective_tasks = []
        for task in self.masked_tasks:
            if task not in labels_dict:
                continue
            labels = labels_dict[task][:batch_size].to(device=device, dtype=torch.long)
            num_classes = self.task_dict[task]
            valid = (labels >= 0) & (labels < num_classes)
            known = torch.full((total_nodes,), -1, dtype=torch.long, device=device)
            if context_indices.numel() > 0:
                task_context = context_indices[valid[context_indices]]
                if task_context.numel() > 0:
                    known[task_context] = labels[task_context]
                    known_indices_by_task[task] = task_context
                else:
                    known_indices_by_task[task] = torch.zeros(0, dtype=torch.long, device=device)
            else:
                known_indices_by_task[task] = torch.zeros(0, dtype=torch.long, device=device)
            known_labels_by_task[task] = known
            effective_tasks.append(task)

        if not effective_tasks:
            return None
        return MaskedConditioningSpec(
            node_mask=full_node_mask,
            known_labels_by_task=known_labels_by_task,
            known_indices_by_task=known_indices_by_task,
            masked_tasks=effective_tasks,
            constraint_mode=self.constraint_mode,
            feedback_mode=self.feedback_mode,
        )

    def _build_model_label_context(
        self,
        conditioning: Optional[MaskedConditioningSpec],
        num_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if conditioning is None:
            return None
        if conditioning.feedback_mode != "single_pass":
            return None
        masked_tasks = [t for t in conditioning.masked_tasks if t in self.task_dict]
        if not masked_tasks:
            return None
        self._init_label_conditioning_modules(masked_tasks)

        note_bias = None
        used = 0
        for task in masked_tasks:
            if task not in conditioning.known_labels_by_task:
                continue
            if task not in self.label_condition_embeddings:
                continue
            known = conditioning.known_labels_by_task[task]
            if known.numel() != num_nodes:
                raise ValueError(
                    f"Known-label tensor size mismatch for task '{task}': {known.numel()} vs {num_nodes}"
                )
            known = known.to(device=device, dtype=torch.long)
            mask_id = self.task_dict[task]
            tokens = torch.where(
                (known >= 0) & (known < mask_id),
                known,
                torch.full_like(known, mask_id),
            )
            emb = self.label_condition_embeddings[task](tokens)
            note_bias = emb if note_bias is None else (note_bias + emb)
            used += 1
        if note_bias is None or used == 0:
            return None
        note_bias = note_bias / float(used)
        if self.label_condition_fusion is not None:
            note_bias = self.label_condition_fusion(note_bias)
        note_bias = note_bias.to(device=device, dtype=dtype)
        return {"note_bias": note_bias}

    def _apply_known_label_constraints(
        self,
        logits_dict: Dict[str, torch.Tensor],
        conditioning: Optional[MaskedConditioningSpec],
        valid_tasks: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if conditioning is None:
            return logits_dict
        if conditioning.constraint_mode != "hard":
            return logits_dict
        tasks = valid_tasks if valid_tasks is not None else list(logits_dict.keys())
        out = dict(logits_dict)
        for task in tasks:
            if task not in out:
                continue
            if task not in conditioning.known_indices_by_task:
                continue
            indices = conditioning.known_indices_by_task[task]
            if indices is None or indices.numel() == 0:
                continue
            labels = conditioning.known_labels_by_task[task]
            indices = indices.to(device=out[task].device, dtype=torch.long)
            labels = labels.to(device=out[task].device, dtype=torch.long)
            out[task] = clamp_logits_to_labels(
                out[task],
                labels,
                indices,
                num_classes=self.task_dict.get(task),
            )
        return out

    def _encode_notes_with_musicbert(self, batch):
        note_store = batch["note"]
        cached_embeddings = (
            getattr(note_store, "musicbert_note_embeddings", None)
            if self.musicbert_use_cached_embeddings
            else None
        )
        if cached_embeddings is not None:
            if not isinstance(cached_embeddings, torch.Tensor):
                cached_embeddings = torch.tensor(cached_embeddings)
            cached_embeddings = cached_embeddings.to(device=self.device)
            if cached_embeddings.ndim != 2:
                raise ValueError(
                    "Cached MusicBERT embeddings must be 2D [num_nodes, hidden_dim], "
                    f"got shape {tuple(cached_embeddings.shape)}"
                )
            expected_nodes = int(note_store.num_nodes)
            if cached_embeddings.shape[0] != expected_nodes:
                raise ValueError(
                    "Cached MusicBERT embeddings do not match sampled node count: "
                    f"{cached_embeddings.shape[0]} vs {expected_nodes}"
                )
            return cached_embeddings

        if self.note_encoder is None:
            if self.musicbert_use_cached_embeddings:
                raise ValueError(
                    "MusicBERT cached embeddings are enabled but missing on the batch. "
                    "Ensure `--musicbert_cached_embeddings_dir` is set and contains .npz files for all graphs."
                )
            raise ValueError("MusicBERT note encoder is not available.")

        required_fields = ["input_ids", "attention_mask", "token2note", "num_notes"]
        missing_fields = [field for field in required_fields if not hasattr(batch, field)]
        if missing_fields:
            raise ValueError(
                "MusicBERT note encoder requires batch fields: "
                f"{', '.join(missing_fields)}"
            )

        input_ids, attention_mask, token2note, num_notes = self._normalize_musicbert_inputs(batch)

        note_embeddings, _ = self.note_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token2note=token2note,
            num_notes=num_notes,
        )

        note_list = [note_embeddings[i, :num_notes[i]] for i in range(len(num_notes))]
        note_embeddings = torch.cat(note_list, dim=0)
        if hasattr(note_store, "note_idx"):
            note_idx = note_store.note_idx
            if not isinstance(note_idx, torch.Tensor):
                note_idx = torch.tensor(note_idx, device=note_embeddings.device)
            note_idx = note_idx.to(device=note_embeddings.device, dtype=torch.long)

            batch_ids = getattr(note_store, "batch", None)
            if batch_ids is not None:
                if not isinstance(batch_ids, torch.Tensor):
                    batch_ids = torch.tensor(batch_ids, device=note_embeddings.device)
                batch_ids = batch_ids.to(device=note_embeddings.device, dtype=torch.long)

                offsets = torch.zeros(len(num_notes), dtype=torch.long, device=note_embeddings.device)
                if len(num_notes) > 1:
                    offsets[1:] = torch.cumsum(
                        torch.tensor(num_notes[:-1], dtype=torch.long, device=note_embeddings.device),
                        dim=0,
                    )
                global_idx = note_idx + offsets[batch_ids]
            else:
                global_idx = note_idx

            if global_idx.numel() and global_idx.max().item() >= note_embeddings.shape[0]:
                raise ValueError(
                    "MusicBERT note indices exceed available embeddings: "
                    f"{global_idx.max().item()} >= {note_embeddings.shape[0]}"
                )
            note_embeddings = note_embeddings[global_idx]

        expected_nodes = int(note_store.num_nodes)
        if note_embeddings.shape[0] != expected_nodes:
            raise ValueError(
                "MusicBERT note embeddings do not match sampled node count: "
                f"{note_embeddings.shape[0]} vs {expected_nodes}. "
                "Ensure note_idx is attached before subgraph sampling."
            )

        return note_embeddings

    def _fuse_note_features(self, note_features: Optional[torch.Tensor], note_embeddings: torch.Tensor) -> torch.Tensor:
        if note_features is not None and note_embeddings.dtype != note_features.dtype:
            note_embeddings = note_embeddings.to(note_features.dtype)
        elif note_features is None:
            note_dtype = self.model.project_dict["note"][0].weight.dtype
            if note_embeddings.dtype != note_dtype:
                note_embeddings = note_embeddings.to(note_dtype)
        if note_features is None or self.musicbert_fusion == "replace":
            return note_embeddings
        if self.musicbert_fusion == "concat":
            return torch.cat([note_features, note_embeddings], dim=-1)
        if self.musicbert_fusion == "gate":
            if self.musicbert_proj is None or self.musicbert_gate is None:
                return note_embeddings
            projected = self.musicbert_proj(note_embeddings)
            gate = self.musicbert_gate(torch.cat([note_features, projected], dim=-1))
            return gate * projected + (1.0 - gate) * note_features
        return note_embeddings

    def _normalize_musicbert_inputs(self, batch):
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        token2note = batch.token2note
        num_notes = batch.num_notes

        if isinstance(num_notes, torch.Tensor):
            num_notes = num_notes.tolist()

        if isinstance(input_ids, torch.Tensor):
            input_ids_tensor = input_ids
        else:
            input_ids_list = [
                torch.tensor(seq, dtype=torch.long, device=self.device) for seq in input_ids
            ]
            input_ids_tensor = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=0
            )

        if isinstance(attention_mask, torch.Tensor):
            attention_mask_tensor = attention_mask
        else:
            attention_list = [
                torch.tensor(seq, dtype=torch.long, device=self.device) for seq in attention_mask
            ]
            attention_mask_tensor = torch.nn.utils.rnn.pad_sequence(
                attention_list, batch_first=True, padding_value=0
            )

        if isinstance(token2note, torch.Tensor):
            token2note_list = [token2note]
        else:
            token2note_list = [
                torch.tensor(edges, dtype=torch.float32, device=self.device) for edges in token2note
            ]

        return input_ids_tensor, attention_mask_tensor, token2note_list, num_notes

    def _compute_task_losses(self, batch):
        x_dict = self._maybe_encode_x_dict(batch, batch.x_dict)
        batch_size = batch["note"].batch_size
        total_nodes = int(batch["note"].x.size(0))
        labels_dict = {k: batch["note"][k][:batch_size] for k in self.task_dict.keys() if k in batch["note"].keys()}
        pitch_spelling = batch["note"].pitch_spelling
        key_signature = batch["note"].key_signature
        labels_dict = {
            k: torch.where(
                (labels_dict[k] >= 0) & (labels_dict[k] < self.task_dict[k]),
                labels_dict[k],
                torch.full_like(labels_dict[k], -1),
            )
            for k in labels_dict.keys()
        }
        edge_index_dict = batch.edge_index_dict
        batch_dict = batch.batch_dict
        num_sampled_edges_dict = batch.num_sampled_edges_dict
        num_sampled_nodes_dict = batch.num_sampled_nodes_dict

        device = labels_dict[list(labels_dict.keys())[0]].device if labels_dict else batch["note"].x.device
        mask_dict = self.create_mask_dict(labels_dict, batch, batch_size)
        node_mask = self._get_node_mask_for_batch(
            batch=batch,
            batch_size=batch_size,
            device=device,
            allow_sampling=True,
        )
        batch_conditioning = self._build_batch_masked_conditioning(
            labels_dict=labels_dict,
            node_mask=node_mask,
            batch_size=batch_size,
            total_nodes=total_nodes,
            device=device,
        )
        label_context = self._build_model_label_context(
            conditioning=batch_conditioning,
            num_nodes=total_nodes,
            dtype=x_dict["note"].dtype,
            device=x_dict["note"].device,
        )

        if "valid_label" not in batch["note"].keys():
            valid_label_mask = torch.ones_like(batch["note"]["pitch_spelling"][:batch_size]).bool()
        else:
            valid_label_mask = batch["note"]["valid_label"][:batch_size].bool()

        labels_valid = {k: v[valid_label_mask] for k, v in labels_dict.items()}
        mask_valid = {k: v[valid_label_mask] for k, v in mask_dict.items()}
        node_mask_valid = node_mask[valid_label_mask] if node_mask is not None else None
        labels_dict = {}
        mask_dict = {}
        for task, values in labels_valid.items():
            if task not in mask_valid:
                continue
            task_values = values[mask_valid[task]]
            if task_values.numel() == 0:
                continue
            if not self._metric_safe_mask(task_values, task).any():
                continue
            labels_dict[task] = task_values
            mask_dict[task] = mask_valid[task]
        if not labels_dict:
            zero = torch.tensor(0.0, device=batch["note"].x.device)
            return {}, zero, zero, zero

        x = self.model.encode(
            pitch_spelling=pitch_spelling,
            key_signature=key_signature,
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            batch_dict=batch_dict,
            batch_size=batch_size,
            neighbor_mask_node=num_sampled_nodes_dict,
            neighbor_mask_edge=num_sampled_edges_dict,
            label_context=label_context,
        )
        feature_loss = x.pow(2).mean()

        edge_loss = torch.tensor(0.0, device=x.device)
        rna_keys = ["quality", "inversion", "degree1", "degree2", "localkey"]
        if self.use_edge_loss and all(rna_key in labels_dict.keys() for rna_key in rna_keys):
            target_edge_index_dict = {
                k: v[:, (v[0] < batch_size) & (v[1] < batch_size)]
                for k, v in edge_index_dict.items()
                if k[0] == "note" and k[-1] == "note"
            }
            for k, v in target_edge_index_dict.items():
                if v.size(1) > batch_size:
                    target_edge_index_dict[k] = v[:, torch.randperm(v.size(1))[:batch_size]]
                else:
                    target_edge_index_dict[k] = v

            ground_truth_same_label_edge_dict = {
                k: torch.zeros(v.shape[-1], device=x.device) for k, v in target_edge_index_dict.items()
            }
            for k, v in target_edge_index_dict.items():
                if k[0] == "note" and k[-1] == "note":
                    src_labels = torch.stack([labels_dict[rna_key][v[0]] for rna_key in rna_keys], dim=1)
                    tgt_labels = torch.stack([labels_dict[rna_key][v[1]] for rna_key in rna_keys], dim=1)
                    same_label_mask = (src_labels == tgt_labels).all(dim=1)
                    ground_truth_same_label_edge_dict[k] = same_label_mask.long()
            edge_logits_dict = self.edge_clf(target_edge_index_dict, x)
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
            if node_mask_valid is not None:
                node_mask_valid = torch.ones_like(y_over, dtype=torch.float32)

        logits_dict = self.model.forward_clf(x)
        logits_dict = {k: logits_dict[k][mask_dict[k]] for k in labels_dict.keys()}

        raw_task_node_masks = {}
        task_loss_masks = {}
        if node_mask_valid is not None:
            for task in labels_dict.keys():
                raw_task_mask = node_mask_valid[mask_dict[task]]
                raw_task_node_masks[task] = raw_task_mask
                if self.masked_prediction_train and task in self.masked_tasks:
                    task_loss_masks[task] = torch.where(
                        raw_task_mask > 0.9,
                        torch.ones_like(raw_task_mask),
                        torch.zeros_like(raw_task_mask),
                    )
                else:
                    task_loss_masks[task] = raw_task_mask
            if self.constraint_mode == "hard":
                for task in labels_dict.keys():
                    _, context_indices, _ = split_nodes_by_mask(raw_task_node_masks[task])
                    if context_indices.numel() == 0:
                        continue
                    valid_context = self._metric_safe_mask(labels_dict[task], task)[context_indices]
                    context_indices = context_indices[valid_context]
                    if context_indices.numel() == 0:
                        continue
                    logits_dict[task] = clamp_logits_to_labels(
                        logits_dict[task],
                        labels_dict[task],
                        context_indices,
                        num_classes=self.task_dict.get(task),
                    )

        node_mask_for_loss = task_loss_masks if task_loss_masks else None
        loss_dict = self.clf_loss(logits_dict, labels_dict, node_mask=node_mask_for_loss)
        task_losses = {k: loss_dict[k] for k in labels_dict.keys()}
        total_task_loss = loss_dict["total"] / len(labels_dict.keys())
        if self.masked_prediction_train:
            masked_losses = [task_losses[t] for t in self.masked_tasks if t in task_losses]
            if masked_losses:
                masked_target_total_loss = torch.stack(masked_losses).mean()
                self.log("train/masked_target_total_loss", masked_target_total_loss, prog_bar=False)
            for task in self.masked_tasks:
                if task not in labels_dict:
                    continue
                task_logits = logits_dict[task]
                task_labels = labels_dict[task]
                raw_mask = raw_task_node_masks.get(task)
                if raw_mask is None:
                    continue
                target_indices, context_indices, _ = split_nodes_by_mask(raw_mask)
                if target_indices.numel() > 0:
                    target_acc = (task_logits[target_indices].argmax(-1) == task_labels[target_indices]).float().mean()
                    self.log(f"train/{task}_target_acc", target_acc, prog_bar=False)
                if context_indices.numel() > 0:
                    valid_context = self._metric_safe_mask(task_labels, task)[context_indices]
                    context_indices = context_indices[valid_context]
                if context_indices.numel() > 0:
                    consistency = (task_logits[context_indices].argmax(-1) == task_labels[context_indices]).float().mean()
                    self.log(f"train/{task}_known_consistency", consistency, prog_bar=False)

        memory_loss = 0.0
        if len(self.previous_tasks) > 0:
            if self.lambda_dctn > 0:
                x_mem = self.memory_model.encode(
                    pitch_spelling=pitch_spelling,
                    key_signature=key_signature,
                    x_dict=x_dict,
                    edge_index_dict=edge_index_dict,
                    batch_dict=batch_dict,
                    batch_size=batch_size,
                    neighbor_mask_node=num_sampled_nodes_dict,
                    neighbor_mask_edge=num_sampled_edges_dict,
                    label_context=label_context,
                )
                logits_mem = self.model.forward_clf(x_mem, self.previous_tasks)
                memory_dict_logits = self.memory_model.forward_clf(x_mem, self.previous_tasks)
                temp = 2.0
                teacher_probs = {k: F.softmax(v / temp, 1) for k, v in memory_dict_logits.items()}
                student_log_probs = {k: F.log_softmax(v / temp, 1) for k, v in logits_mem.items()}
                memory_loss_dict = {
                    k: F.kl_div(student_log_probs[k], teacher_probs[k], reduction="batchmean") * (temp ** 2)
                    for k in self.previous_tasks
                }
                memory_loss = torch.stack(list(memory_loss_dict.values())).mean()
                self.log("train/memory_loss", memory_loss, prog_bar=True)
                memory_loss = self.lambda_dctn * memory_loss
            if self.has_memories:
                self.memory_replay()
                ewc_loss = self.get_ewc_loss()
                self.log("train/ewc_loss", ewc_loss.item(), prog_bar=True)
                memory_loss += self.lambda_ewc * ewc_loss

        lambda_edge = self.hparams.get("lambda_edge", 0.05)
        aux_loss = memory_loss + feature_loss * self.lambda_featl + edge_loss * lambda_edge

        return task_losses, total_task_loss, aux_loss, feature_loss

    def _metric_safe_mask(self, labels: torch.Tensor, task_name: str) -> torch.Tensor:
        num_classes = self.task_dict[task_name]
        return (labels >= 0) & (labels < num_classes)

    def _safe_metric_value(
        self,
        metric: nn.Module,
        logits: torch.Tensor,
        labels: torch.Tensor,
        task_name: str,
    ):
        valid_mask = self._metric_safe_mask(labels, task_name)
        if not torch.any(valid_mask):
            return None
        return metric(logits[valid_mask], labels[valid_mask])

    def _gradnorm_shared_params(self):
        if hasattr(self.model, "project_dict") and "note" in self.model.project_dict:
            params = [p for p in self.model.project_dict["note"].parameters() if p.requires_grad]
            if params:
                return params
        return [p for p in self.model.parameters() if p.requires_grad]

    def _pcgrad_param_groups(self):
        if self._pcgrad_shared_params_cache is not None and self._pcgrad_head_params_cache is not None:
            cached = self._pcgrad_shared_params_cache + self._pcgrad_head_params_cache
            if all(p.requires_grad for p in cached):
                return self._pcgrad_shared_params_cache, self._pcgrad_head_params_cache
            self._pcgrad_shared_params_cache = None
            self._pcgrad_head_params_cache = None

        head_prefixes = ("clf_dict.", "clf_proj_layers.", "cross_task_transformer.", "fusion_layers.")
        seen = set()
        shared_params: List[torch.nn.Parameter] = []
        head_params: List[torch.nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            if name.startswith(head_prefixes):
                head_params.append(param)
            else:
                shared_params.append(param)

        for module in (self.note_encoder, self.musicbert_proj, self.musicbert_gate):
            if module is None:
                continue
            for param in module.parameters():
                if not param.requires_grad:
                    continue
                pid = id(param)
                if pid in seen:
                    continue
                seen.add(pid)
                shared_params.append(param)

        # If no explicit heads were detected, keep previous behavior.
        if not head_params:
            shared_params = [p for p in self.parameters() if p.requires_grad]

        self._pcgrad_shared_params_cache = shared_params
        self._pcgrad_head_params_cache = head_params
        return shared_params, head_params

    @staticmethod
    def _accumulate_param_grads(
        params: List[torch.nn.Parameter],
        grads: List[Optional[torch.Tensor]],
    ) -> None:
        for param, grad in zip(params, grads):
            if grad is None:
                continue
            grad_detached = grad.detach()
            if param.grad is None:
                param.grad = grad_detached
            else:
                param.grad = param.grad + grad_detached

    def _should_step_optimizer(self) -> bool:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return True
        fit_loop = getattr(trainer, "fit_loop", None)
        if fit_loop is None:
            return True
        try:
            return not fit_loop._should_accumulate()
        except Exception:
            return True

    @staticmethod
    def _grad_norm(params: List[torch.nn.Parameter]) -> float:
        norms = [p.grad.detach().norm(2) for p in params if p.grad is not None]
        if not norms:
            return 0.0
        stacked = torch.stack(norms)
        return float(torch.norm(stacked, p=2).item())

    def _log_optimizer_stats(self, optimizer: Optimizer) -> None:
        if self.optimizer_stats_log_every_n_steps <= 0:
            return
        if int(self.global_step) % self.optimizer_stats_log_every_n_steps != 0:
            return
        lr = float(optimizer.param_groups[0]["lr"])
        self.log("train/lr", lr, on_step=True, on_epoch=False, logger=True)

        all_params = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    all_params.append(p)
        total_grad_norm = self._grad_norm(all_params)
        self.log("train/grad_norm_total", total_grad_norm, on_step=True, on_epoch=False, logger=True)

        note_proj_params = []
        if hasattr(self.model, "project_dict") and "note" in self.model.project_dict:
            note_proj_params = [
                p for p in self.model.project_dict["note"].parameters() if p.requires_grad
            ]
        if note_proj_params:
            note_proj_grad_norm = self._grad_norm(note_proj_params)
            self.log(
                "train/grad_norm_note_proj",
                note_proj_grad_norm,
                on_step=True,
                on_epoch=False,
                logger=True,
            )

    def _manual_clip_gradients(self, optimizer: Optimizer) -> None:
        if self.grad_clip_val <= 0:
            return
        params = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)
        if params:
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip_val)

    def _get_scheduler_obj(self):
        sched = self.lr_schedulers()
        if isinstance(sched, (list, tuple)):
            return sched[0] if sched else None
        return sched

    def _manual_scheduler_step(self, when: str, metric: Optional[float] = None) -> None:
        sched = self._get_scheduler_obj()
        if sched is None:
            return
        if self.scheduler_type == "plateau":
            if when == "epoch" and metric is not None:
                sched.step(metric)
            return
        if when == "step":
            sched.step()

    def _apply_pcgrad(self, task_losses: Union[Dict[str, torch.Tensor], List[torch.Tensor]], aux_loss: torch.Tensor):
        opt = self.optimizers()
        if opt is None:
            return
        should_step = self._should_step_optimizer()
        losses = list(task_losses.values()) if isinstance(task_losses, dict) else list(task_losses)
        losses = [loss for loss in losses if loss is not None]
        if not losses:
            return
        aux_requires_grad = aux_loss is not None and getattr(aux_loss, "requires_grad", False)

        shared_params, head_params = self._pcgrad_param_groups()
        head_requires_grad = len(head_params) > 0

        if not shared_params and not head_params:
            loss_mean = torch.stack(losses).mean()
            if aux_requires_grad:
                self.manual_backward(loss_mean + aux_loss)
            else:
                self.manual_backward(loss_mean)
        else:
            # Apply gradient surgery only on shared parameters.
            if shared_params:
                retain_for_heads = head_requires_grad or aux_requires_grad
                if len(losses) == 1:
                    grads = torch.autograd.grad(
                        losses[0],
                        shared_params,
                        retain_graph=retain_for_heads,
                        allow_unused=True,
                    )
                    self._accumulate_param_grads(shared_params, list(grads))
                else:
                    pcgrad = PCGrad(parameters=shared_params)
                    pcgrad.pc_backward(losses, retain_graph=retain_for_heads)

            # Backprop average task loss through task-specific heads normally.
            if head_params:
                loss_mean = torch.stack(losses).mean()
                head_grads = torch.autograd.grad(
                    loss_mean,
                    head_params,
                    retain_graph=aux_requires_grad,
                    allow_unused=True,
                )
                self._accumulate_param_grads(head_params, list(head_grads))

            # Aux terms (feature/edge/etc.) update all relevant parameters.
            if aux_requires_grad:
                self.manual_backward(aux_loss)

        if should_step:
            self._manual_clip_gradients(opt)
            self._log_optimizer_stats(opt)
            opt.step()
            opt.zero_grad()
            self._manual_scheduler_step("step")

    def _apply_gradnorm(self, task_losses: Dict[str, torch.Tensor], aux_loss: torch.Tensor):
        opt = self.optimizers()
        if opt is None:
            return
        should_step = self._should_step_optimizer()
        tasks = list(task_losses.keys())
        loss_vec = torch.stack([task_losses[t] for t in tasks])

        for t, loss in task_losses.items():
            if t not in self.initial_task_losses:
                self.initial_task_losses[t] = loss.detach()

        init_losses = torch.stack([self.initial_task_losses[t] for t in tasks]).to(loss_vec.device)
        indices = torch.tensor([self.task_to_idx[t] for t in tasks], device=loss_vec.device)
        weights = self.gradnorm_weights[indices]
        weighted_losses = weights * loss_vec
        total_task_loss = weighted_losses.sum()

        shared_params = self._gradnorm_shared_params()
        grad_norms = []
        for w, loss in zip(weights, loss_vec):
            grads = torch.autograd.grad(
                w * loss,
                shared_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            norms = [g.norm() for g in grads if g is not None]
            grad_norms.append(torch.norm(torch.stack(norms)) if norms else torch.tensor(0.0, device=loss_vec.device))
        grad_norms = torch.stack(grad_norms)

        loss_ratio = loss_vec.detach() / (init_losses + 1e-8)
        loss_ratio = loss_ratio / loss_ratio.mean()
        target = grad_norms.detach().mean() * (loss_ratio ** self.gradnorm_alpha)
        gradnorm_loss = torch.sum(torch.abs(grad_norms - target))

        total_loss = total_task_loss + aux_loss + gradnorm_loss

        self.manual_backward(total_loss)
        if should_step:
            self._manual_clip_gradients(opt)
            self._log_optimizer_stats(opt)
            opt.step()
            opt.zero_grad()
            with torch.no_grad():
                self.gradnorm_weights.clamp_(min=1e-3)
                self.gradnorm_weights.mul_(len(self.gradnorm_weights) / (self.gradnorm_weights.sum() + 1e-8))

            self._manual_scheduler_step("step")

    def common_step(self, batch):
        task_losses, total_task_loss, aux_loss, feature_loss = self._compute_task_losses(batch)
        if not task_losses:
            return torch.tensor(0.0, device=batch["note"].x.device)
        total_loss = total_task_loss + aux_loss
        self.log("train/total_loss", total_loss.item(), prog_bar=True)
        self.log("train/feature_loss", feature_loss.item(), prog_bar=True)
        for k, loss in task_losses.items():
            self.log(f"train/{k}_loss", loss.item())
        return total_loss

    def _maybe_encode_x_dict(self, batch, x_dict):
        if self.note_encoder is None and not self.musicbert_use_cached_embeddings:
            return x_dict
        note_embeddings = self._encode_notes_with_musicbert(batch)
        note_features = x_dict.get("note") if isinstance(x_dict, dict) else None
        fused = self._fuse_note_features(note_features, note_embeddings)
        x_dict = dict(x_dict)
        x_dict["note"] = fused
        return x_dict

    def training_step(self, batch, batch_idx):
        if self.mt_conflict_method == "none" or self.mt_strategy == "famo":
            if isinstance(batch, dict):
                combined_batch = batch
                loss = 0
                count = 0
                for _, bt in combined_batch.items():
                    if bt is None:
                        continue
                    loss += self.common_step(bt)
                    count += 1
                return loss / max(count, 1)
            return self.common_step(batch)

        if isinstance(batch, dict):
            if self.mt_conflict_method == "pcgrad":
                return self._training_step_conflict_combined_pcgrad(batch)
            losses = []
            for _, bt in batch.items():
                if bt is None:
                    continue
                losses.append(self._training_step_conflict(bt))
            if losses:
                return torch.stack([l.detach() for l in losses]).mean()
            return torch.tensor(0.0, device=self.device)
        return self._training_step_conflict(batch)

    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        if self.automatic_optimization:
            self._log_optimizer_stats(optimizer)

    def _training_step_conflict(self, batch):
        task_losses, total_task_loss, aux_loss, feature_loss = self._compute_task_losses(batch)
        if not task_losses:
            return torch.tensor(0.0, device=batch["note"].x.device)

        if self.mt_conflict_method == "pcgrad":
            self._apply_pcgrad(task_losses, aux_loss)
            total_loss = (sum(task_losses.values()) / len(task_losses)) + aux_loss
        elif self.mt_conflict_method == "gradnorm":
            self._apply_gradnorm(task_losses, aux_loss)
            total_loss = total_task_loss + aux_loss
        else:
            total_loss = total_task_loss + aux_loss

        self.log("train/total_loss", total_loss.item(), prog_bar=True)
        self.log("train/feature_loss", feature_loss.item(), prog_bar=True)
        for k, loss in task_losses.items():
            self.log(f"train/{k}_loss", loss.item())
        return total_loss

    def _training_step_conflict_combined_pcgrad(self, combined_batch):
        task_loss_terms: List[torch.Tensor] = []
        per_task_losses: Dict[str, List[torch.Tensor]] = {}
        feature_losses: List[torch.Tensor] = []
        aux_total: Optional[torch.Tensor] = None

        for _, batch in combined_batch.items():
            if batch is None:
                continue
            task_losses, _, aux_loss, feature_loss = self._compute_task_losses(batch)
            if not task_losses:
                continue

            feature_losses.append(feature_loss.detach())
            aux_total = aux_loss if aux_total is None else (aux_total + aux_loss)
            for task_name, loss in task_losses.items():
                task_loss_terms.append(loss)
                per_task_losses.setdefault(task_name, []).append(loss.detach())

        if not task_loss_terms:
            return torch.tensor(0.0, device=self.device)

        if aux_total is None:
            aux_total = torch.tensor(0.0, device=task_loss_terms[0].device)

        self._apply_pcgrad(task_loss_terms, aux_total)
        task_loss_mean = torch.stack(task_loss_terms).mean()
        total_loss = task_loss_mean + aux_total

        self.log("train/total_loss", total_loss.item(), prog_bar=True)
        if feature_losses:
            self.log("train/feature_loss", torch.stack(feature_losses).mean().item(), prog_bar=True)
        for task_name, losses in per_task_losses.items():
            self.log(f"train/{task_name}_loss", torch.stack(losses).mean().item())
        return total_loss

    def validation_step(self, combined_batch, batch_idx):
        # combined_batch = combined_batch if isinstance(combined_batch, dict) else {self.current_task: combined_batch}
        for k, batch in combined_batch.items():
            if k not in self.current_val_tasks:
                continue
            if batch is None:
                continue
            x_dict = self._maybe_encode_x_dict(batch, batch.x_dict)
            batch_size = batch["note"].batch_size
            total_nodes = int(batch["note"].x.size(0))
            labels_dict = {k: batch["note"][k][:batch_size] for k in self.task_dict.keys() if k in batch["note"].keys()}
            pitch_spelling = batch["note"].pitch_spelling
            key_signature = batch["note"].key_signature
            # Keep only valid class indices and map everything else to ignore_index (-1).
            labels_dict = {
                k: torch.where(
                    (labels_dict[k] >= 0) & (labels_dict[k] < self.task_dict[k]),
                    labels_dict[k],
                    torch.full_like(labels_dict[k], -1),
                )
                for k in labels_dict.keys()
            }
            edge_index_dict = batch.edge_index_dict
            batch_dict = batch.batch_dict

            num_sampled_edges_dict = batch.num_sampled_edges_dict
            num_sampled_nodes_dict = batch.num_sampled_nodes_dict
            mask_dict = self.create_mask_dict(labels_dict, batch, batch_size)
            device = labels_dict[list(labels_dict.keys())[0]].device if labels_dict else batch["note"].x.device
            node_mask = self._get_node_mask_for_batch(
                batch=batch,
                batch_size=batch_size,
                device=device,
                allow_sampling=self.masked_prediction_train,
            )
            batch_conditioning = self._build_batch_masked_conditioning(
                labels_dict=labels_dict,
                node_mask=node_mask,
                batch_size=batch_size,
                total_nodes=total_nodes,
                device=device,
            )
            label_context = self._build_model_label_context(
                conditioning=batch_conditioning,
                num_nodes=total_nodes,
                dtype=x_dict["note"].dtype,
                device=x_dict["note"].device,
            )

            # NOTE: mask to remove invalid labels
            if "valid_label" not in batch["note"].keys():
                valid_label_mask = torch.ones_like(batch["note"]["pitch_spelling"][:batch_size]).bool()
            else:
                valid_label_mask = batch["note"]["valid_label"][:batch_size].bool()

            labels_valid = {k: v[valid_label_mask] for k, v in labels_dict.items()}
            mask_valid = {k: v[valid_label_mask] for k, v in mask_dict.items()}
            node_mask_valid = node_mask[valid_label_mask] if node_mask is not None else None
            labels_dict = {}
            mask_dict = {}
            for task, values in labels_valid.items():
                if task not in mask_valid:
                    continue
                task_values = values[mask_valid[task]]
                if task_values.numel() == 0:
                    continue
                if not self._metric_safe_mask(task_values, task).any():
                    continue
                labels_dict[task] = task_values
                mask_dict[task] = mask_valid[task]
            if not labels_dict:
                continue

            logits_dict = self.model(
                    pitch_spelling=pitch_spelling,
                    key_signature=key_signature,
                    x_dict=x_dict,
                    edge_index_dict=edge_index_dict,
                    batch_dict=batch_dict,
                    batch_size=batch_size,
                    neighbor_mask_node=num_sampled_nodes_dict,
                    neighbor_mask_edge=num_sampled_edges_dict,
                    label_context=label_context,
                )
            logits_dict = {k: v[valid_label_mask] for k, v in logits_dict.items()}
            logits_dict = {k: (v[mask_dict[k]] if k in mask_dict.keys() else v) for k, v in logits_dict.items()}

            raw_task_node_masks = {}
            task_loss_masks = {}
            if node_mask_valid is not None:
                for task in labels_dict.keys():
                    raw_task_mask = node_mask_valid[mask_dict[task]]
                    raw_task_node_masks[task] = raw_task_mask
                    if self.masked_prediction_train and task in self.masked_tasks:
                        task_loss_masks[task] = torch.where(
                            raw_task_mask > 0.9,
                            torch.ones_like(raw_task_mask),
                            torch.zeros_like(raw_task_mask),
                        )
                    else:
                        task_loss_masks[task] = raw_task_mask
                if self.constraint_mode == "hard":
                    for task in labels_dict.keys():
                        _, context_indices, _ = split_nodes_by_mask(raw_task_node_masks[task])
                        if context_indices.numel() == 0:
                            continue
                        valid_context = self._metric_safe_mask(labels_dict[task], task)[context_indices]
                        context_indices = context_indices[valid_context]
                        if context_indices.numel() == 0:
                            continue
                        logits_dict[task] = clamp_logits_to_labels(
                            logits_dict[task],
                            labels_dict[task],
                            context_indices,
                            num_classes=self.task_dict.get(task),
                        )

            node_mask_for_loss = task_loss_masks if task_loss_masks else None
            loss_dict = self.clf_loss(logits_dict, labels_dict, node_mask=node_mask_for_loss)
            total_loss = loss_dict.pop("total") / len(labels_dict.keys())
            accuracy_dict = {}
            f1_dict = {}
            for task_name in labels_dict.keys():
                acc = self._safe_metric_value(
                    self.accuracy_dict[task_name], logits_dict[task_name], labels_dict[task_name], task_name
                )
                f1 = self._safe_metric_value(
                    self.f1_dict[task_name], logits_dict[task_name], labels_dict[task_name], task_name
                )
                if acc is not None and f1 is not None:
                    accuracy_dict[task_name] = acc
                    f1_dict[task_name] = f1
            self.log("val/total_loss", total_loss.item(), batch_size=batch_size, prog_bar=True)

            for k in loss_dict.keys():
                self.log(f"val/{k}_loss", loss_dict[k].item(), batch_size=batch_size)
                if k in accuracy_dict:
                    self.log(f"val/{k}_acc", accuracy_dict[k], batch_size=batch_size)
                if k in f1_dict:
                    self.log(f"val/{k}_f1", f1_dict[k], batch_size=batch_size)
            if self.masked_prediction_train:
                masked_losses = [loss_dict[t] for t in self.masked_tasks if t in loss_dict]
                if masked_losses:
                    self.log(
                        "val/masked_target_total_loss",
                        torch.stack(masked_losses).mean(),
                        batch_size=batch_size,
                    )
                for task in self.masked_tasks:
                    if task not in labels_dict:
                        continue
                    raw_mask = raw_task_node_masks.get(task)
                    if raw_mask is None:
                        continue
                    target_indices, context_indices, _ = split_nodes_by_mask(raw_mask)
                    if target_indices.numel() > 0:
                        target_acc = (
                            logits_dict[task][target_indices].argmax(-1) == labels_dict[task][target_indices]
                        ).float().mean()
                        self.log(f"val/{task}_target_acc", target_acc, batch_size=batch_size)
                    if context_indices.numel() > 0:
                        valid_context = self._metric_safe_mask(labels_dict[task], task)[context_indices]
                        context_indices = context_indices[valid_context]
                    if context_indices.numel() > 0:
                        consistency = (
                            logits_dict[task][context_indices].argmax(-1) == labels_dict[task][context_indices]
                        ).float().mean()
                        self.log(f"val/{task}_known_consistency", consistency, batch_size=batch_size)

            # RNA accuracy calculation based on in_label notes only
            if "tpc_in_label" in logits_dict.keys():
                rna_keys = ["quality", "inversion", "degree1", "degree2", "localkey"]
                mask = logits_dict["tpc_in_label"].argmax(-1).bool()
                if not mask.any():
                    continue
                if all([k in labels_dict.keys() for k in rna_keys]):
                    stacked = []
                    for k in rna_keys:                        
                        valid = mask & self._metric_safe_mask(labels_dict[k], k)
                        if not valid.any():
                            continue
                        rna_acc = self.accuracy_dict[k](logits_dict[k][valid], labels_dict[k][valid])
                        stacked.append(logits_dict[k][valid].argmax(-1).eq(labels_dict[k][valid]))
                        self.log(f"val/NCT_{k}_acc", rna_acc, batch_size=batch_size)
                    # stack and get accuracy over all rna keys
                    if stacked:
                        min_len = min(x.numel() for x in stacked)
                        if min_len > 0:
                            stacked = [x[:min_len] for x in stacked]
                            total_rna_acc = torch.stack(stacked).all(dim=0).float().mean()
                            self.log("val/total_rna_acc", total_rna_acc, batch_size=batch_size)

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

        if not self.automatic_optimization and self.scheduler_type == "plateau":
            metric = self.trainer.callback_metrics.get(self.monitor_metric, None)
            if metric is not None:
                if isinstance(metric, torch.Tensor):
                    metric = float(metric.detach().cpu().item())
                else:
                    metric = float(metric)
                self._manual_scheduler_step("epoch", metric=metric)

    def test_step(self, combined_batch, batch_idx) -> STEP_OUTPUT:
        for gtask_key, batch in combined_batch.items():
            if batch is None:
                print("Batch is None")
                continue
            x_dict = self._maybe_encode_x_dict(batch, batch.x_dict)
            batch_size = batch["note"].batch_size
            total_nodes = int(batch["note"].x.size(0))
            labels_dict = {k: batch["note"][k][:batch_size] for k in self.task_dict.keys() if k in batch["note"].keys()}
            pitch_spelling = batch["note"].pitch_spelling
            key_signature = batch["note"].key_signature
            # Keep only valid class indices and map everything else to ignore_index (-1).
            labels_dict = {
                k: torch.where(
                    (labels_dict[k] >= 0) & (labels_dict[k] < self.task_dict[k]),
                    labels_dict[k],
                    torch.full_like(labels_dict[k], -1),
                )
                for k in labels_dict.keys()
            }
            edge_index_dict = batch.edge_index_dict
            batch_dict = batch.batch_dict
            num_sampled_edges_dict = batch.num_sampled_edges_dict
            num_sampled_nodes_dict = batch.num_sampled_nodes_dict
            mask_dict = self.create_mask_dict(labels_dict, batch, batch_size)
            device = labels_dict[list(labels_dict.keys())[0]].device if labels_dict else batch["note"].x.device
            node_mask = self._get_node_mask_for_batch(
                batch=batch,
                batch_size=batch_size,
                device=device,
                allow_sampling=self.masked_prediction_train,
            )
            batch_conditioning = self._build_batch_masked_conditioning(
                labels_dict=labels_dict,
                node_mask=node_mask,
                batch_size=batch_size,
                total_nodes=total_nodes,
                device=device,
            )
            label_context = self._build_model_label_context(
                conditioning=batch_conditioning,
                num_nodes=total_nodes,
                dtype=x_dict["note"].dtype,
                device=x_dict["note"].device,
            )
            # NOTE: mask to remove invalid labels
            if "valid_label" not in batch["note"].keys():
                valid_label_mask = torch.ones_like(batch["note"]["pitch_spelling"][:batch_size]).bool()
            else:
                valid_label_mask = batch["note"]["valid_label"][:batch_size].bool()

            labels_valid = {k: v[valid_label_mask] for k, v in labels_dict.items()}
            mask_valid = {k: v[valid_label_mask] for k, v in mask_dict.items()}
            node_mask_valid = node_mask[valid_label_mask] if node_mask is not None else None
            labels_dict = {}
            mask_dict = {}
            for task, values in labels_valid.items():
                if task not in mask_valid:
                    continue
                task_values = values[mask_valid[task]]
                if task_values.numel() == 0:
                    continue
                if not self._metric_safe_mask(task_values, task).any():
                    continue
                labels_dict[task] = task_values
                mask_dict[task] = mask_valid[task]
            if not labels_dict:
                continue

            logits_dict = self.model(
                pitch_spelling=pitch_spelling,
                key_signature=key_signature,
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                batch_dict=batch_dict,
                batch_size=batch_size,
                neighbor_mask_node=num_sampled_nodes_dict,
                neighbor_mask_edge=num_sampled_edges_dict,
                label_context=label_context,
            )
            logits_dict = {k: v[valid_label_mask] for k, v in logits_dict.items()}
            logits_dict = {k: (v[mask_dict[k]] if k in mask_dict.keys() else v) for k, v in logits_dict.items()}

            raw_task_node_masks = {}
            task_loss_masks = {}
            if node_mask_valid is not None:
                for task in labels_dict.keys():
                    raw_task_mask = node_mask_valid[mask_dict[task]]
                    raw_task_node_masks[task] = raw_task_mask
                    if self.masked_prediction_train and task in self.masked_tasks:
                        task_loss_masks[task] = torch.where(
                            raw_task_mask > 0.9,
                            torch.ones_like(raw_task_mask),
                            torch.zeros_like(raw_task_mask),
                        )
                    else:
                        task_loss_masks[task] = raw_task_mask
                if self.constraint_mode == "hard":
                    for task in labels_dict.keys():
                        _, context_indices, _ = split_nodes_by_mask(raw_task_node_masks[task])
                        if context_indices.numel() == 0:
                            continue
                        valid_context = self._metric_safe_mask(labels_dict[task], task)[context_indices]
                        context_indices = context_indices[valid_context]
                        if context_indices.numel() == 0:
                            continue
                        logits_dict[task] = clamp_logits_to_labels(
                            logits_dict[task],
                            labels_dict[task],
                            context_indices,
                            num_classes=self.task_dict.get(task),
                        )

            logits_softmax_dict = {k: v.softmax(-1) for k, v in logits_dict.items()}
            node_mask_for_loss = task_loss_masks if task_loss_masks else None
            loss_dict = self.clf_loss(logits_dict, labels_dict, node_mask=node_mask_for_loss)
            total_loss = loss_dict.pop("total") / len(labels_dict.keys())
            accuracy_dict = {}
            f1_dict = {}
            for task_name in labels_dict.keys():
                acc = self._safe_metric_value(
                    self.accuracy_dict[task_name], logits_dict[task_name], labels_dict[task_name], task_name
                )
                f1 = self._safe_metric_value(
                    self.f1_dict[task_name], logits_dict[task_name], labels_dict[task_name], task_name
                )
                if acc is not None and f1 is not None:
                    accuracy_dict[task_name] = acc
                    f1_dict[task_name] = f1

            self.log("test/total_loss", total_loss.item(), add_dataloader_idx=True, batch_size=batch_size, prog_bar=True)
            if self.masked_prediction_train:
                masked_losses = [loss_dict[t] for t in self.masked_tasks if t in loss_dict]
                if masked_losses:
                    self.log(
                        "test/masked_target_total_loss",
                        torch.stack(masked_losses).mean(),
                        add_dataloader_idx=True,
                        batch_size=batch_size,
                    )
                for task in self.masked_tasks:
                    if task not in labels_dict:
                        continue
                    raw_mask = raw_task_node_masks.get(task)
                    if raw_mask is None:
                        continue
                    target_indices, context_indices, _ = split_nodes_by_mask(raw_mask)
                    if target_indices.numel() > 0:
                        target_acc = (
                            logits_dict[task][target_indices].argmax(-1) == labels_dict[task][target_indices]
                        ).float().mean()
                        self.log(
                            f"test/{task}_target_acc",
                            target_acc,
                            add_dataloader_idx=True,
                            batch_size=batch_size,
                        )
                    if context_indices.numel() > 0:
                        valid_context = self._metric_safe_mask(labels_dict[task], task)[context_indices]
                        context_indices = context_indices[valid_context]
                    if context_indices.numel() > 0:
                        consistency = (
                            logits_dict[task][context_indices].argmax(-1) == labels_dict[task][context_indices]
                        ).float().mean()
                        self.log(
                            f"test/{task}_known_consistency",
                            consistency,
                            add_dataloader_idx=True,
                            batch_size=batch_size,
                        )

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
                # Remap edges from full-batch node indices to valid-label node indices
                # so they match logits_dict/labels_dict, which are already filtered by valid_label_mask.
                remap = torch.full((batch_size,), -1, dtype=torch.long, device=onset_edges.device)
                remap[valid_label_mask] = torch.arange(
                    int(valid_label_mask.sum().item()),
                    dtype=torch.long,
                    device=onset_edges.device,
                )
                edge_valid = valid_label_mask[onset_edges[0]] & valid_label_mask[onset_edges[1]]
                onset_edges = onset_edges[:, edge_valid]
                if onset_edges.numel() > 0:
                    onset_edges = torch.stack([remap[onset_edges[0]], remap[onset_edges[1]]], dim=0)
                    onset_edges = onset_edges[:, (onset_edges[0] >= 0) & (onset_edges[1] >= 0)]
                # aggregate the logit predictions based on the onset edges
                aggregate_logit_dict = {}
                for k, v in logits_softmax_dict.items():
                    if k in rna_keys:
                        if onset_edges.numel() == 0:
                            aggregate_logit_dict[k] = v
                        else:
                            aggregate_logit_dict[k] = torch_scatter.scatter_mean(
                                v[onset_edges[0]],
                                onset_edges[1],
                                dim=0,
                                out=v.clone(),
                            ).softmax(-1)
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
                valid_rna = torch.stack([self._metric_safe_mask(rna_labels[k], k) for k in rna_keys]).all(dim=0)
                if valid_rna.any():
                    # rna_accuracy is the logical and from the comparison of all the rna keys
                    rna_accuracy = (torch.stack([rna_preds[k][valid_rna] == rna_labels[k][valid_rna] for k in rna_keys]).t().float().mean(
                        -1) == 1).float().mean()
                    self.log(f"test/RN(Onset)_{gtask_key}_accuracy", rna_accuracy.item(), add_dataloader_idx=True, batch_size=batch_size)


            for k in labels_dict.keys():
                if k in accuracy_dict:
                    self.log(f"test/{k}_{gtask_key}_acc", accuracy_dict[k], add_dataloader_idx=True, batch_size=batch_size)
                if k in f1_dict:
                    self.log(f"test/{k}_{gtask_key}_f1", f1_dict[k], add_dataloader_idx=True, batch_size=batch_size)

            # RNA accuracy calculation based on in_label notes only
            if "tpc_in_label" in logits_dict.keys():
                rna_keys = ["quality", "inversion", "degree1", "degree2", "localkey"]
                mask = logits_dict["tpc_in_label"].argmax(-1).bool()
                if not mask.any():
                    continue
                if all([k in labels_dict.keys() for k in rna_keys]):
                    stacked = []
                    for k in rna_keys:
                        valid = mask & self._metric_safe_mask(labels_dict[k], k)
                        if not valid.any():
                            continue
                        rna_acc = self.accuracy_dict[k](logits_dict[k][valid], labels_dict[k][valid])
                        self.log(f"test/NCT_{k}_{gtask_key}_acc", rna_acc, batch_size=batch_size)
                        stacked.append(logits_dict[k][valid].argmax(-1) == labels_dict[k][valid])
                    if stacked:
                        min_len = min(x.numel() for x in stacked)
                        if min_len > 0:
                            stacked = [x[:min_len] for x in stacked]
                            rna_accuracy = (torch.stack(stacked).t().float().mean(-1) == 1).float().mean()
                            self.log(f"test/RN(NCT)_{gtask_key}_accuracy", rna_accuracy.item(), add_dataloader_idx=True, batch_size=batch_size)

    def predict_step(self, batch, batch_idx):
        x_dict = self._maybe_encode_x_dict(batch, batch.x_dict)
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
            x_dict = self._maybe_encode_x_dict(batch, batch.x_dict)
            labels_dict = {k: batch["note"][k] for k in self.task_dict.keys() if k in batch["note"].keys()}
            pitch_spelling = batch["note"].pitch_spelling
            key_signature = batch["note"].key_signature
            # Keep only valid class indices and map everything else to ignore_index (-1).
            labels_dict = {
                k: torch.where(
                    (labels_dict[k] >= 0) & (labels_dict[k] < self.task_dict[k]),
                    labels_dict[k],
                    torch.full_like(labels_dict[k], -1),
                )
                for k in labels_dict.keys()
            }
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
        monitor_metric = self.monitor_metric

        if self.scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=self.monitor_mode,
                factor=self.plateau_factor,
                patience=self.plateau_patience,
                min_lr=self.plateau_min_lr,
            )
            scheduler_cfg = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
            if self.automatic_optimization:
                scheduler_cfg["monitor"] = monitor_metric
                scheduler_cfg["strict"] = False

            optim_cfg = {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_cfg,
            }
            if self.automatic_optimization:
                optim_cfg["monitor"] = monitor_metric
            return optim_cfg

        # Default scheduler: step-wise warmup + cosine decay.
        if hasattr(self.trainer, "estimated_stepping_batches") and self.trainer.estimated_stepping_batches:
            total_steps = int(self.trainer.estimated_stepping_batches)
        else:
            total_steps = int(max(self.total_epochs, 1) * 1000)
        total_steps = max(1, total_steps)

        warmup_steps = int(total_steps * self.warmup_ratio)
        warmup_steps = min(max(0, warmup_steps), total_steps - 1)
        eta_min = float(self.lr * self.min_lr_ratio)

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=eta_min,
            last_epoch=-1,
        )

        scheduler_cfg = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        if self.automatic_optimization:
            scheduler_cfg["strict"] = False

        optim_cfg = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_cfg,
        }
        if self.automatic_optimization:
            optim_cfg["monitor"] = monitor_metric
        return optim_cfg

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

    def predict(
        self,
        score,
        user_edits: Optional[Dict[str, Any]] = None,
        masked_spec: Optional[Dict[str, Any]] = None,
        return_edit_info: bool = False,
    ):
        """Predict analysis for a musical score.
        
        Args:
            score: Path to score file or partitura Score object
            user_edits: Optional user edit spec (see analysisgnn.utils.user_edits)
            masked_spec: Optional masked conditioning spec (known labels + mask).
            return_edit_info: If True, return (predictions, edit_info)
            
        Returns:
            Dictionary of predictions for each task, or (predictions, edit_info)
        """
        import os
        import partitura as pt
        from analysisgnn.descriptors import select_features
        from graphmuse import create_score_graph
        from analysisgnn.utils.music import PitchEncoder, KeySignatureEncoder
        from analysisgnn.utils.user_edits import normalize_user_edits_to_masked_conditioning
        import numpy as np
        
        # Handle both score objects and file paths
        if isinstance(score, str):
            # It's a file path, load it
            score_obj = pt.load_score(score)
        else:
            # It's already a partitura score object
            score_obj = score
        
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

            node_mask, overrides, conditioning = normalize_user_edits_to_masked_conditioning(
                user_edits=user_edits,
                num_nodes=batch_size,
                tasks_num_classes=self.task_dict,
                device=self.device,
                masked_spec=masked_spec,
            )
            
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
            label_context = self._build_model_label_context(
                conditioning=conditioning,
                num_nodes=batch_size,
                dtype=x_dict["note"].dtype,
                device=x_dict["note"].device,
            )
            
            # Get predictions from the model
            logits_dict = self.model(
                pitch_spelling=pitch_spelling,
                key_signature=key_signature,
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                batch_dict=batch_dict,
                batch_size=batch_size,
                neighbor_mask_node=num_sampled_nodes_dict,
                neighbor_mask_edge=num_sampled_edges_dict,
                label_context=label_context,
            )
            logits_dict = self._apply_known_label_constraints(logits_dict, conditioning)

            if overrides:
                for task, override in overrides.items():
                    if task not in logits_dict:
                        continue
                    indices = override["indices"].to(logits_dict[task].device)
                    if indices.numel() == 0:
                        continue
                    labels = override["labels"].to(logits_dict[task].device)
                    logits_dict[task] = clamp_logits_to_labels(
                        logits_dict[task],
                        labels,
                        indices,
                        num_classes=self.task_dict.get(task),
                    )
            
            # Convert logits to probabilities
            predictions = {k: torch.softmax(v, dim=-1) for k, v in logits_dict.items()}

            # aggregate to onsetwise prediction
            predictions = onsetwise_logit_aggregation(predictions, graph=data, batch_size=batch_size)

            # aggregate to beatwise prediction
            predictions = beatwise_logit_aggregation(predictions, graph=data, batch_size=batch_size)

            # aggregate to measurewise prediction
            predictions = measurewise_logit_aggregation(predictions, graph=data, batch_size=batch_size)
            
            if return_edit_info:
                edit_info = {
                    "node_mask": node_mask,
                    "label_overrides": overrides,
                    "masked_conditioning": conditioning,
                }
                return predictions, edit_info
            return predictions
            
        except Exception as e:
            raise ValueError(f"Failed to process score: {str(e)}")
