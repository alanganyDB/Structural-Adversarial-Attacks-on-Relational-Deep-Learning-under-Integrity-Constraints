import os
import time
import math
import json
import copy
import random
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Embedding, ModuleDict

from sentence_transformers import SentenceTransformer

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from torch.nn import BCEWithLogitsLoss

from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data.stats import StatType
from collections import Counter

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP, MessagePassing, HeteroConv
from torch_geometric.typing import NodeType
from torch_geometric.seed import seed_everything


#####################################
##############  Models ##############
#####################################

class GloveTextEmbedding:
    def __init__(self, device: Optional[str] = None):
        self.model = SentenceTransformer(
            'sentence-transformers/average_word_embeddings_glove.6B.300d',
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))
    


class Model(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        shallow_list: List[NodeType] = [],
        id_awareness: bool = False,
    ):
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={node_type: data[node_type].tf.col_names_dict for node_type in data.node_types},
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[node_type for node_type in data.node_types if 'time' in data[node_type]],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(channels, out_channels=out_channels, norm=norm, num_layers=1)
        self.embedding_dict = ModuleDict({node: Embedding(data.num_nodes_dict[node], channels) for node in shallow_list})
        self.id_awareness_emb = torch.nn.Embedding(1, channels) if id_awareness else None
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(self, batch: HeteroData, entity_table: NodeType) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        return self.head(x_dict[entity_table][: seed_time.size(0)])
    

class WeightedSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='mean'):
        super().__init__(aggr=aggr)
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lin_l = nn.Linear(in_channels[0], out_channels)
        self.lin_r = nn.Linear(in_channels[1], out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)
        x_src, x_dst = x
        if edge_weight is None:
            edge_weight = x_src.new_ones(edge_index.size(1))
        out = self.propagate(edge_index, x=(x_src, x_dst), edge_weight=edge_weight)
        out = self.lin_l(out) + self.lin_r(x_dst)
        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j
    
class AttackableHeteroGraphSAGE(nn.Module):
    def __init__(self, node_types, edge_types, channels, aggr='sum', num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {edge_type: WeightedSAGEConv((channels, channels), channels, aggr=aggr) for edge_type in edge_types},
                aggr='sum',
            )
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({node_type: nn.LayerNorm(channels) for node_type in node_types}))

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None, num_sampled_nodes_dict=None, num_sampled_edges_dict=None):
        if edge_weight_dict is None:
            edge_weight_dict = {}
        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
            x_dict = {k: norm_dict[k](x) for k, x in x_dict.items()}
            x_dict = {k: x.relu() for k, x in x_dict.items()}
        return x_dict
    

class AttackableModel(nn.Module):
    def __init__(self, base_model, attackable_gnn):
        super().__init__()
        self.encoder = base_model.encoder
        self.temporal_encoder = base_model.temporal_encoder
        self.head = base_model.head
        self.embedding_dict = getattr(base_model, 'embedding_dict', None)
        self.gnn = attackable_gnn

    def forward(self, batch, entity_table, edge_weight_dict=None, edge_index_dict=None):
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for node_type in x_dict:
            if node_type in rel_time_dict:
                x_dict[node_type] = x_dict[node_type] + rel_time_dict[node_type]
        if self.embedding_dict is not None:
            for node_type, emb in self.embedding_dict.items():
                x_dict[node_type] = x_dict[node_type] + emb(batch[node_type].n_id)
        if edge_index_dict is None:
            edge_index_dict = batch.edge_index_dict
        x_dict = self.gnn(
            x_dict,
            edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            num_sampled_nodes_dict=getattr(batch, 'num_sampled_nodes_dict', None),
            num_sampled_edges_dict=getattr(batch, 'num_sampled_edges_dict', None),
        )
        return self.head(x_dict[entity_table][: seed_time.size(0)])
    



def build_attackable_model(base_model, data, channels, aggr, num_layers):
    attackable_gnn = AttackableHeteroGraphSAGE(
        node_types=list(data.node_types),
        edge_types=list(data.edge_types),
        channels=channels,
        aggr=aggr,
        num_layers=num_layers,
    )
    return AttackableModel(base_model, attackable_gnn)

######################################
##############  Helpers ##############
######################################


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_everything(seed)

def safe_get_dataset(name: str, root_dir: str = './data', download: bool = True):
    """RelBench API compatibility: some versions accept root=, some do not."""
    try:
        return get_dataset(name, root=root_dir, download=download)
    except TypeError:
        return get_dataset(name, download=download)
    
def safe_get_task(dataset_name: str, task_name: str, root_dir: str = './data', download: bool = True):
    """RelBench API compatibility: some versions accept root=, some do not."""
    try:
        return get_task(dataset_name, task_name, root=root_dir, download=download)
    except TypeError:
        return get_task(dataset_name, task_name, download=download)
    
def safe_get_task_table(task, split: str):
    """RelBench API compatibility: new versions expose task.get_table(split)."""
    if hasattr(task, 'get_table'):
        return task.get_table(split)
    return getattr(task, f'{split}_table')

def infer_task_type(task_name: str, task=None) -> str:
    if task_name in {'driver-dnf', 'driver-top3'}:
        return 'classification'
    return 'regression'

def prediction_for_eval(raw_pred: Tensor, task_type: str) -> Tensor:
    raw_pred = raw_pred.view(-1) if raw_pred.size(-1) == 1 else raw_pred
    if task_type == 'classification':
        return torch.sigmoid(raw_pred.float())
    return raw_pred


######################################
##############   Main   ##############
######################################



def get_y_from_batch(batch, entity_table):
    assert "y" in batch[entity_table], (
        f"No y in batch[{entity_table}]. "
        f"Available keys: {batch[entity_table].keys()}"
    )
    return batch[entity_table].y.view(-1)


def regression_mae_on_batch(model, batch, entity_table):
    model.eval()
    with torch.no_grad():
        pred = model(batch, entity_table).view(-1)
        y = get_y_from_batch(batch, entity_table).to(pred.device).float().view(-1)

        n = min(pred.numel(), y.numel())
        return float(torch.mean(torch.abs(pred[:n] - y[:n])).item())


def classification_bce_on_batch(model, batch, entity_table):
    model.eval()
    with torch.no_grad():
        logits = model(batch, entity_table).view(-1)
        y = get_y_from_batch(batch, entity_table).to(logits.device).float().view(-1)

        n = min(logits.numel(), y.numel())
        return float(F.binary_cross_entropy_with_logits(logits[:n], y[:n]).item())


def classification_accuracy_on_batch(model, batch, entity_table):
    model.eval()
    with torch.no_grad():
        logits = model(batch, entity_table).view(-1)
        y = get_y_from_batch(batch, entity_table).to(logits.device).float().view(-1)

        n = min(logits.numel(), y.numel())
        pred = (logits[:n] >= 0).float()
        acc = (pred == y[:n]).float().mean()

        return float(acc.item())


def eval_metric_on_batch(model, batch, entity_table, task_type="regression", metric_name=None):
    if task_type == "regression":
        if metric_name is None:
            metric_name = "mae"
        if metric_name == "mae":
            return regression_mae_on_batch(model, batch, entity_table)
        raise ValueError(f"Unknown regression metric: {metric_name}")

    if task_type == "classification":
        if metric_name is None:
            metric_name = "accuracy"
        if metric_name == "accuracy":
            return classification_accuracy_on_batch(model, batch, entity_table)
        if metric_name == "bce":
            return classification_bce_on_batch(model, batch, entity_table)
        raise ValueError(f"Unknown classification metric: {metric_name}")

    raise ValueError(f"Unknown task_type: {task_type}")


# Backward compatibility avec ton ancien code regression.
def mae_on_batch(model, batch, entity_table):
    return regression_mae_on_batch(model, batch, entity_table)


######################################
##############   Main + ##############
######################################



def _build_edge_lookup(edge_index):
    src, dst = edge_index
    return {(int(src[e]), int(dst[e])): int(e) for e in range(edge_index.size(1))}

def apply_forward_rewirings(batch, attacked_relation, selected_rewirings, reverse_relation=None):
    new_batch = copy.deepcopy(batch)
    new_batch.edge_index_dict = dict(batch.edge_index_dict)

    edge_index = batch.edge_index_dict[attacked_relation].clone()

    # attacked_relation = child -> parent
    # row: child fixed, old_dst -> new_dst
    lookup = _build_edge_lookup(edge_index)

    for row in selected_rewirings:
        child = int(row["child"])
        old_dst = int(row["old_dst"])
        new_dst = int(row["new_dst"])

        pos = lookup.get((child, old_dst), None)
        if pos is not None:
            edge_index[1, pos] = new_dst

    new_batch.edge_index_dict[attacked_relation] = edge_index

    # Update reverse relation: parent -> child
    if reverse_relation is not None and reverse_relation in batch.edge_index_dict:
        rev_edge_index = batch.edge_index_dict[reverse_relation].clone()
        rev_lookup = _build_edge_lookup(rev_edge_index)

        for row in selected_rewirings:
            child = int(row["child"])
            old_dst = int(row["old_dst"])
            new_dst = int(row["new_dst"])

            # old reverse edge: old_parent -> child
            pos = rev_lookup.get((old_dst, child), None)
            if pos is not None:
                # new reverse edge: new_parent -> child
                rev_edge_index[0, pos] = new_dst

        new_batch.edge_index_dict[reverse_relation] = rev_edge_index

    return new_batch


def default_reverse_relation(edge_type):
    src, rel, dst = edge_type
    if rel.startswith('rev_'):
        return (dst, rel[len('rev_'):], src)
    return (dst, f'rev_{rel}', src)



def gradient_order_from_score_info(score_info, topk_per_child=1):
    """Return a global proposal order from a ranked top-k-per-child pool.

    IMPORTANT: this returns proposals, not necessarily final DB edits.  When
    applying inside one local batch, we keep at most one final edit per child.
    In the future global reconstruction stage, top-k gives backup proposals
    when batch-level proposals conflict.
    """
    topk_per_child = max(1, int(topk_per_child))
    pool = score_info.get('scored_rewirings_all', [])
    if not pool:
        pool = score_info.get('scored_rewirings', [])
    rows = [r for r in pool if int(r.get('rank_within_child', 1)) <= topk_per_child]
    rows = sorted(rows, key=lambda x: x.get('score', x.get('score_raw', -np.inf)), reverse=True)
    return rows



def get_default_relations():
    # All reverse FK->PK relations in rel-f1 that can be rewired at graph level.
    # We attack the reverse direction because each child tuple has exactly one current parent.
    return [
        ('races', 'rev_f2p_raceId', 'constructor_results'),
        ('constructors', 'rev_f2p_constructorId', 'constructor_results'),
        ('races', 'rev_f2p_raceId', 'results'),
        ('drivers', 'rev_f2p_driverId', 'results'),
        ('constructors', 'rev_f2p_constructorId', 'results'),
        ('races', 'rev_f2p_raceId', 'qualifying'),
        ('drivers', 'rev_f2p_driverId', 'qualifying'),
        ('constructors', 'rev_f2p_constructorId', 'qualifying'),
        ('circuits', 'rev_f2p_circuitId', 'races'),
        ('races', 'rev_f2p_raceId', 'standings'),
        ('drivers', 'rev_f2p_driverId', 'standings'),
        ('races', 'rev_f2p_raceId', 'constructor_standings'),
        ('constructors', 'rev_f2p_constructorId', 'constructor_standings'),
    ]


def _get_num_local_nodes(batch, node_type):
    store = batch[node_type]
    if hasattr(store, 'num_nodes') and store.num_nodes is not None:
        return int(store.num_nodes)
    if hasattr(store, 'n_id'):
        return int(store.n_id.size(0))
    raise ValueError(f'Cannot infer num_nodes for {node_type}')



def summarize(df_all):
    agg_cols = {
        'attacked_delta_mae': ['mean', 'std'],
        'attacked_delta_mse': ['mean', 'std'],
        'attacked_delta_rmse': ['mean', 'std'],
        'attacked_delta_mean_pred': ['mean', 'std'],
        'attacked_delta_target_mean_pred': ['mean', 'std'],
        'attacked_delta_rest_mean_pred': ['mean', 'std'],
        'attacked_delta_target_rest_gap': ['mean', 'std'],
        'attacked_advantage_score': ['mean', 'std'],
        'apply_time_sec': ['mean'],
        'eval_time_sec': ['mean'],
        'total_eval_time_sec': ['mean'],
        'random_order_build_time_sec': ['mean'],
        'n_selected': ['mean'],
        'n_candidate_edges': ['mean'],
    }
    rel_summary = df_all.groupby(['relation', 'method', 'budget_frac']).agg(agg_cols)
    rel_summary.columns = ['_'.join(c).strip('_') for c in rel_summary.columns]
    rel_summary = rel_summary.reset_index()
    global_summary = df_all.groupby(['method', 'budget_frac']).agg(agg_cols)
    global_summary.columns = ['_'.join(c).strip('_') for c in global_summary.columns]
    global_summary = global_summary.reset_index()
    return rel_summary, global_summary






def sample_candidate_support_per_child_vectorized(
    batch,
    attacked_relation,
    max_candidates_per_dst=25,
    include_existing=True,
):
    """
    Candidate sampler for FORWARD FK -> PK relations.

    Convention:
        attacked_relation = (child_table, f2p_fk_name, parent_table)

    Therefore:
        edge_index[0] = child / FK row  = fixed source
        edge_index[1] = parent / PK row = current destination

    A rewiring means:
        keep the child fixed,
        replace its current parent dst by a new candidate parent dst.

    Output layout per child:
        [existing edge, candidate_1, ..., candidate_k]

    So for each child c:
        candidate_edge_index contains:
            c -> old_parent
            c -> new_parent_1
            ...
            c -> new_parent_k
    """

    edge_index = batch.edge_index_dict[attacked_relation]
    device = edge_index.device
    dtype = edge_index.dtype

    child_type, _, parent_type = attacked_relation

    src_child, dst_parent = edge_index

    num_parent_nodes = _get_num_local_nodes(batch, parent_type)

    # Collect one current parent per child/FK row.
    # In a clean FK relation, each child should have one parent.
    # We still keep the first occurrence for safety.
    seen = set()
    child_ids = []
    old_dsts = []
    old_edge_pos = []

    for e in range(int(edge_index.size(1))):
        child = int(src_child[e])
        if child in seen:
            continue
        seen.add(child)

        child_ids.append(child)
        old_dsts.append(int(dst_parent[e]))
        old_edge_pos.append(e)

    n_child = len(child_ids)

    if n_child == 0 or num_parent_nodes <= 1 or max_candidates_per_dst <= 0:
        empty_edge_index = torch.empty((2, 0), dtype=dtype, device=device)
        empty_mask = torch.empty((0,), dtype=torch.bool, device=device)

        support = {
            "fast": True,
            "orientation": "forward_fk_to_pk",
            "child_type": child_type,
            "parent_type": parent_type,
            "num_parent_nodes": int(num_parent_nodes),
            "n_child": int(n_child),
            "k_eff": 0,
        }

        return empty_edge_index, empty_mask, support

    k_eff = min(int(max_candidates_per_dst), int(num_parent_nodes) - 1)

    child_t = torch.tensor(child_ids, dtype=dtype, device=device)
    old_dst_t = torch.tensor(old_dsts, dtype=dtype, device=device)
    old_edge_pos_t = torch.tensor(old_edge_pos, dtype=torch.long, device=device)

    # Random candidate parents.
    # Shape: [n_child, k_eff]
    cand_dst = torch.randint(
        low=0,
        high=int(num_parent_nodes),
        size=(n_child, k_eff),
        device=device,
        dtype=dtype,
    )

    # Avoid proposing the already-existing parent.
    old_dst_mat = old_dst_t.view(-1, 1).expand_as(cand_dst)
    collision = cand_dst.eq(old_dst_mat)

    if bool(collision.any().item()):
        cand_dst = torch.where(
            collision,
            (cand_dst + 1) % int(num_parent_nodes),
            cand_dst,
        )

    if include_existing:
        # First column = current true edge.
        dst_mat = torch.cat([old_dst_t.view(-1, 1), cand_dst], dim=1)

        # Child source is fixed across all candidate parents.
        src_mat = child_t.view(-1, 1).expand(n_child, k_eff + 1)

        existing_mask = torch.zeros(
            n_child,
            k_eff + 1,
            dtype=torch.bool,
            device=device,
        )
        existing_mask[:, 0] = True

    else:
        dst_mat = cand_dst
        src_mat = child_t.view(-1, 1).expand(n_child, k_eff)

        existing_mask = torch.zeros(
            n_child,
            k_eff,
            dtype=torch.bool,
            device=device,
        )

    candidate_edge_index = torch.stack(
        [
            src_mat.reshape(-1),
            dst_mat.reshape(-1),
        ],
        dim=0,
    )

    support = {
        "fast": True,
        "orientation": "forward_fk_to_pk",

        # Relation metadata.
        "child_type": child_type,
        "parent_type": parent_type,
        "num_parent_nodes": int(num_parent_nodes),

        # Candidate layout metadata.
        "n_child": int(n_child),
        "k_eff": int(k_eff),

        # One row per rewired child.
        "child_ids": child_t.long(),
        "old_dsts": old_dst_t.long(),
        "old_edge_pos": old_edge_pos_t.long(),
        "cand_dsts": cand_dst.long(),
    }

    return candidate_edge_index, existing_mask.reshape(-1), support



def targeted_output_objective(pred, y=None, direction="increase", mode="mae", q=0.75, alpha=0.25):
    pred = pred.view(-1)

    if mode == "mae":
        assert y is not None, "mode='mae' requires y"
        y = y.to(pred.device).float().view(-1)

        n = min(pred.numel(), y.numel())
        obj = F.l1_loss(pred[:n], y[:n])

    elif mode == "bce":
        assert y is not None, "mode='bce' requires y"
        y = y.to(pred.device).float().view(-1)

        n = min(pred.numel(), y.numel())
        obj = F.binary_cross_entropy_with_logits(pred[:n], y[:n])

    elif mode == "mean":
        obj = pred.mean()

    elif mode in ["quantile", "contrastive"]:
        thresh = torch.quantile(pred.detach(), q)
        target = pred[pred >= thresh]
        rest = pred[pred < thresh]

        if target.numel() == 0:
            target = pred

        if mode == "quantile":
            obj = target.mean()
        else:
            rest_mean = rest.mean() if rest.numel() > 0 else pred.mean()
            obj = target.mean() - alpha * rest_mean

    else:
        raise ValueError(mode)

    if direction == "increase":
        return obj

    if direction == "decrease":
        return -obj

    raise ValueError(direction)

def score_targeted_rewirings_once(
    model,
    batch,
    entity_table,
    attacked_relation,
    max_candidates_per_dst=25,
    direction="increase",          # "increase" ou "decrease"
    objective_mode="mean",
    q=0.75,
    alpha=0.25,
    score_norm="raw",
    max_topk_per_child=10,
    verbose=False,
    plot_grad=True,
):
    """
    Score rewiring candidates for FORWARD FK -> PK relations.

    Convention:
        attacked_relation = (child_table, f2p_fk_name, parent_table)

    Rewiring:
        child/FK src is fixed
        parent/PK dst is replaced

    Candidate layout per child:
        [old_parent, cand_1, ..., cand_k]
    """

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    info_out = {
        "status": "unknown",
        "skip_reason": "",
        "relation": str(attacked_relation),
        "scored_rewirings": [],
        "scored_rewirings_all": [],
        "E": 0,
        "n_candidate_edges": 0,
        "candidate_build_time_sec": 0.0,
        "backward_time_sec": 0.0,
        "scoring_time_sec": 0.0,
        "score_total_time_sec": 0.0,
        "score_norm": score_norm,
        "max_topk_per_child": int(max_topk_per_child),
    }
    if attacked_relation not in batch.edge_index_dict:
        info_out.update(status="skip", skip_reason="missing_in_batch")
        return info_out

    edge_index = batch.edge_index_dict[attacked_relation]
    E = int(edge_index.size(1))

    info_out.update({
        "E": E,
        "num_edges_batch": E,
        "num_unique_src": int(edge_index[0].unique().numel()) if edge_index.numel() else 0,
        "num_unique_dst": int(edge_index[1].unique().numel()) if edge_index.numel() else 0,
    })

    if E == 0:
        info_out.update(status="skip", skip_reason="empty_relation")
        return info_out

    # ------------------------------------------------------------------
    # 1. Build candidates
    # ------------------------------------------------------------------
    t_build = time.time()

    try:
        candidate_edge_index, existing_mask, support = sample_candidate_support_per_child_vectorized(
            batch=batch,
            attacked_relation=attacked_relation,
            max_candidates_per_dst=max_candidates_per_dst,
            include_existing=True,
        )
    except Exception as e:
        candidate_build_time = time.time() - t_build
        info_out.update(
            status="skip",
            skip_reason="candidate_build_failed",
            error=repr(e),
            candidate_build_time_sec=candidate_build_time,
            score_total_time_sec=candidate_build_time,
        )
        return info_out
    print(candidate_edge_index)
    candidate_build_time = time.time() - t_build

    n_candidate_edges = int(candidate_edge_index.size(1))
    n_child = int(support.get("n_child", 0))
    k_eff = int(support.get("k_eff", 0))
    n_children_with_candidates = n_child if k_eff > 0 else 0

    info_out.update({
        "n_candidate_edges": n_candidate_edges,
        "num_current_children": n_child,
        "num_children_with_candidates": n_children_with_candidates,
        "candidate_build_time_sec": candidate_build_time,
        "candidate_sampler": "vectorized_with_replacement_forward_fk_to_pk",
        "k_eff_per_child": k_eff,
    })

    if verbose:
        print(
            f"[score] relation={attacked_relation} | "
            f"children={n_child} | k_eff={k_eff} | "
            f"candidate_edges={n_candidate_edges} | "
            f"build_time={candidate_build_time:.2f}s",
            flush=True,
        )

    if n_candidate_edges == 0 or n_children_with_candidates == 0:
        info_out.update(
            status="skip",
            skip_reason="no_candidates",
            score_total_time_sec=candidate_build_time,
        )
        return info_out

    # ------------------------------------------------------------------
    # 2. Sanity check candidate layout
    # ------------------------------------------------------------------
    block = k_eff + 1
    expected_n_edges = n_child * block

    src = candidate_edge_index[0]
    old_src = src[existing_mask]

    unique_src, src_counts = src.unique(return_counts=True)
    unique_old_src, old_src_counts = old_src.unique(return_counts=True)

    layout_ok = expected_n_edges == n_candidate_edges
    all_src_counts_ok = bool((src_counts == block).all().item())
    all_old_src_counts_ok = bool((old_src_counts == 1).all().item())

    info_out.update({
        "sanity_layout_ok": bool(layout_ok),
        "sanity_all_src_counts_eq_block": all_src_counts_ok,
        "sanity_all_old_src_counts_eq_1": all_old_src_counts_ok,
    })


    # ------------------------------------------------------------------
    # 3. Differentiable edge mask
    # ------------------------------------------------------------------
    m = existing_mask.float().detach().clone().requires_grad_(True)


    edge_index_dict = dict(batch.edge_index_dict)
    edge_index_dict[attacked_relation] = candidate_edge_index

    edge_weight_dict = {et: None for et in batch.edge_index_dict.keys()}
    edge_weight_dict[attacked_relation] = m

    # ------------------------------------------------------------------
    # 4. Forward / backward
    # ------------------------------------------------------------------
    t_backward = time.time()

    model.zero_grad(set_to_none=True)

    pred = model(
        batch,
        entity_table,
        edge_weight_dict=edge_weight_dict,
        edge_index_dict=edge_index_dict,
    ).view(-1)


    #pred = model(batch, entity_table).view(-1)
    y = batch[entity_table].y.view(-1)

    objective = targeted_output_objective(
        pred=pred,
        y=y,
        direction=direction,
        mode=objective_mode,
    )
  
    info_out["pred_requires_grad"] = bool(pred.requires_grad)
    info_out["objective_requires_grad"] = bool(objective.requires_grad)
    info_out["objective_has_grad_fn"] = bool(objective.grad_fn is not None)
    info_out["objective"] = float(objective.detach().cpu().item())

    if not objective.requires_grad:
        backward_time = time.time() - t_backward
        info_out.update(
            status="skip",
            skip_reason="objective_no_grad",
            backward_time_sec=backward_time,
            score_total_time_sec=candidate_build_time + backward_time,
        )
        return info_out


    objective.backward()

    backward_time = time.time() - t_backward
    info_out["backward_time_sec"] = backward_time
    info_out["mask_requires_grad"] = bool(m.requires_grad)
    info_out["mask_grad_is_none"] = bool(m.grad is None)
    if m.grad is None:
        info_out.update(
            status="skip",
            skip_reason="mask_grad_none",
            score_total_time_sec=candidate_build_time + backward_time,
        )
        return info_out

    grad = m.grad.detach()
    grad_cpu = grad.float().cpu()

    abs_grad = grad.abs() ##### ici c'est chelou, pourquoi tu me prend la valeur absolue du gradient J'EN VEUX SURTOUT PAS 
    grad_nonzero_1e_12 = int((abs_grad > 1e-12).sum().cpu().item())

    info_out.update({
        "grad_abs_sum": float(abs_grad.sum().cpu().item()),
        "grad_abs_max": float(abs_grad.max().cpu().item()),
        "grad_abs_mean": float(abs_grad.mean().cpu().item()),
        "grad_nonzero_1e-12": grad_nonzero_1e_12,
        "grad_nonzero_1e-9": int((abs_grad > 1e-9).sum().cpu().item()),
        "grad_nonzero_1e-6": int((abs_grad > 1e-6).sum().cpu().item()),
    })

    if grad_nonzero_1e_12 == 0:
        info_out.update(
            status="ok_zero_grad",
            skip_reason="",
            score_total_time_sec=candidate_build_time + backward_time,
        )
        return info_out


    # ------------------------------------------------------------------
    # 5. Sanity score per child
    # ------------------------------------------------------------------
    scores_inc = []
    scores_dec = []

    for c in range(n_child):
        s = c * block
        old_g = grad[s] ## valeur de l'ancien gradient 
        new_g = grad[s + 1 : s + block] ## valeur de tous les gradients qui sont associer a un potentiel rewireing avec block candidats  

        scores_inc.append((new_g.max() - old_g).item()) # la valeur qui nous iteresse. 
        scores_dec.append((old_g - new_g.min()).item())

    scores_inc = torch.tensor(scores_inc)
    scores_dec = torch.tensor(scores_dec)

    # print("n_child:", n_child, "| k_eff:", k_eff)

    # print("[increase] min/mean/max:",
    #       scores_inc.min().item(),
    #       scores_inc.mean().item(),
    #       scores_inc.max().item())
    # print("[increase] positive:",
    #       (scores_inc > 0).sum().item(), "/", scores_inc.numel())

    # print("[decrease] min/mean/max:",
    #       scores_dec.min().item(),
    #       scores_dec.mean().item(),
    #       scores_dec.max().item())
    # print("[decrease] positive:",
    #       (scores_dec > 0).sum().item(), "/", scores_dec.numel())

    if plot_grad:
        plt.figure(figsize=(8, 5))
        plt.hist(grad_cpu.numpy(), bins=200, log=True)
        plt.xlabel("Gradient value")
        plt.ylabel("Count (log scale)")
        plt.title("Distribution of m.grad")
        plt.grid(alpha=0.2)
        plt.show()

        topk = torch.topk(grad_cpu, min(20, grad_cpu.numel()))
        print("Top gradients:")
        for idx in topk.indices.tolist():
            print(
                idx,
                "src=", int(candidate_edge_index[0, idx]),
                "dst=", int(candidate_edge_index[1, idx]),
                "old=", bool(existing_mask[idx]),
                "grad=", float(grad_cpu[idx]),
            )

    # ------------------------------------------------------------------
    # 6. Vectorized scoring / top-k per child
    # ------------------------------------------------------------------
    t_scoring = time.time()

    eps = 1e-12
    rows_all = []
    best_rows = []

    child_ids = support["child_ids"].detach().cpu().numpy().astype(np.int64)
    old_dsts = support["old_dsts"].detach().cpu().numpy().astype(np.int64)
    old_edge_pos = support["old_edge_pos"].detach().cpu().numpy().astype(np.int64)
    cand_dsts = support["cand_dsts"].detach().cpu().numpy().astype(np.int64)

    old_pos_np = np.arange(n_child, dtype=np.int64) * block
    new_pos_np = old_pos_np[:, None] + 1 + np.arange(k_eff, dtype=np.int64)[None, :]

    g_old = grad_cpu[torch.from_numpy(old_pos_np)].numpy().astype(np.float64)
    g_new = (
        grad_cpu[torch.from_numpy(new_pos_np.reshape(-1))]
        .numpy()
        .reshape(n_child, k_eff)
        .astype(np.float64)
    )

    if direction == "increase":
        score_raw = g_new - g_old[:, None]
    elif direction == "decrease":
        score_raw = g_old[:, None] - g_new
    else:
        raise ValueError("direction must be 'increase' or 'decrease'")

    mean = score_raw.mean(axis=1, keepdims=True)
    std = score_raw.std(axis=1, keepdims=True)
    med = np.median(score_raw, axis=1, keepdims=True)
    mad = np.median(np.abs(score_raw - med), axis=1, keepdims=True)


    score_z = (score_raw - mean) / (std + eps)
    score_robust_z = (score_raw - med) / (1.4826 * mad + eps)

    # ============================================================
    # MIN-MAX NORMALIZATION PER RELATION
    # Ici la fonction est appelée relation par relation,
    # donc min/max sont bien calculés à l'intérieur d'une relation.
    # ============================================================

    rel_min = score_raw.min()
    rel_max = score_raw.max()

    score_minmax_relation = (
        (score_raw - rel_min)
        / (rel_max - rel_min + eps)
    )

    if score_norm == "raw":
        score_mat = score_raw

    elif score_norm == "zscore":
        score_mat = score_z

    elif score_norm == "robust_zscore":
        # HACK TEMPORAIRE :
        # on remplace robust_zscore par min-max relation
        # pour ne rien modifier après la cellule 27.
        score_mat = score_minmax_relation

    elif score_norm == "minmax_relation":
        score_mat = score_minmax_relation

    else:
        raise ValueError(f"Unknown score_norm={score_norm}")

    max_topk_per_child = max(1, int(max_topk_per_child))
    k_keep = min(max_topk_per_child, k_eff)

    top_idx_unsorted = np.argpartition(-score_mat, kth=k_keep - 1, axis=1)[:, :k_keep]
    top_scores_unsorted = np.take_along_axis(score_mat, top_idx_unsorted, axis=1)
    order = np.argsort(-top_scores_unsorted, axis=1)
    top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)

    for i in range(n_child):
        for rank0, j in enumerate(top_idx[i], start=1):

            j = int(j)

            r = {
                "child": int(child_ids[i]),

                "old_dst": int(old_dsts[i]),
                "new_dst": int(cand_dsts[i, j]),

                "old_edge_pos": int(old_edge_pos[i]),

                "g_old": float(g_old[i]),
                "g_new": float(g_new[i, j]),

                "score_raw": float(score_raw[i, j]),

                "score_child_mean": float(mean[i, 0]),
                "score_child_std": float(std[i, 0]),
                "score_child_median": float(med[i, 0]),
                "score_child_mad": float(mad[i, 0]),

                "score_z": float(score_z[i, j]),
                "score_robust_z": float(score_robust_z[i, j]),
                "score_minmax_relation": float(score_minmax_relation[i, j]),

                "score": float(score_mat[i, j]),

                "rank_within_child": int(rank0),
            }

            # -------------------------------------------------
            # IMPORTANT:
            # keep old parent if rewiring is not beneficial
            # -------------------------------------------------
            if r["score"] <= 0:
                continue

            rows_all.append(r)

            if rank0 == 1:
                best_rows.append(r)

    rows_all = sorted(rows_all, key=lambda x: x["score"], reverse=True)
    best_rows = sorted(best_rows, key=lambda x: x["score"], reverse=True)

    scoring_time = time.time() - t_scoring

    if verbose:
        print(
            f"[score] relation={attacked_relation} | "
            f"scoring done in {scoring_time:.2f}s | "
            f"materialized_rows={len(rows_all)}",
            flush=True,
        )

    info_out.update({
        "status": "ok_nonzero_grad" if len(best_rows) > 0 else "skip",
        "skip_reason": "" if len(best_rows) > 0 else "no_scored_rewirings",
        "scored_rewirings": best_rows,
        "scored_rewirings_all": rows_all,
        "scoring_time_sec": scoring_time,
        "score_total_time_sec": candidate_build_time + backward_time + scoring_time,
        "direction": direction,
        "objective_mode": objective_mode,
    })

    return info_out


def predict_loader(model, loader, entity_table, device, task_type='regression'):
    model.eval()
    pred_list = []
    for batch in loader:
        batch = batch.to(device)
        raw_pred = model(batch, entity_table)
        pred = prediction_for_eval(raw_pred, task_type)
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


def train_model(model, loader_dict, entity_table, task, val_table, device, epochs, lr, task_type='regression'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_score = -math.inf if task_type == 'classification' else math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in tqdm(loader_dict['train'], desc=f'train epoch {epoch}', leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            raw_pred = model(batch, entity_table)
            pred = raw_pred.view(-1) if raw_pred.size(-1) == 1 else raw_pred
            y = batch[entity_table].y.float()
            y = y[: pred.numel()].view_as(pred)
            if task_type == 'classification':
                loss = F.binary_cross_entropy_with_logits(pred.float(), y.float())
            else:
                loss = F.l1_loss(pred.float(), y.float())
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu().item()) * pred.numel()
            count += pred.numel()
        val_pred = predict_loader(model, loader_dict['val'], entity_table, device, task_type=task_type)
        val_metrics = task.evaluate(val_pred, val_table)
        log(f"Epoch {epoch:02d} | train_loss={total/max(count,1):.6f} | val_metrics={val_metrics}")
        metric_key = 'auroc' if task_type == 'classification' and 'auroc' in val_metrics else ('roc_auc' if task_type == 'classification' and 'roc_auc' in val_metrics else 'mae')
        metric_value = float(val_metrics[metric_key])
        improved = metric_value > best_score if task_type == 'classification' else metric_value < best_score
        if improved:
            best_score = metric_value
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {'best_val_metric': best_score, 'task_type': task_type}


def choose_best_metric(metrics):
    preferred = [
        "roc_auc",
        "auroc",
        "average_precision",
        "ap",
        "accuracy",
        "f1",
    ]

    for key in preferred:
        if key in metrics:
            return key, True

    for key in metrics:
        if key not in {"loss", "mae", "rmse", "mse"}:
            return key, True

    key = list(metrics.keys())[0]
    return key, key not in {"loss", "mae", "rmse", "mse"}



def train_binary_task(model, task, loader_dict, val_table, test_table, device= "cpu", epochs=10, lr = 1e-5):
    entity_table = task.entity_table
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = BCEWithLogitsLoss()

    def train_one_epoch():
        model.train()
        loss_accum = 0.0
        count_accum = 0

        for batch in tqdm(loader_dict["train"], desc="train", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch, entity_table).view(-1)
            y = batch[entity_table].y.float().view(-1).to(logits.device)

            n = min(logits.numel(), y.numel())
            loss = loss_fn(logits[:n].float(), y[:n].float())

            loss.backward()
            optimizer.step()

            loss_accum += float(loss.detach().item()) * n
            count_accum += n

        return loss_accum / max(count_accum, 1)

    @torch.no_grad()
    def predict(loader):
        model.eval()
        pred_list = []

        for batch in tqdm(loader, desc="eval", leave=False):
            batch = batch.to(device)
            logits = model(batch, entity_table).view(-1)
            probs = torch.sigmoid(logits)
            pred_list.append(probs.detach().cpu())

        return torch.cat(pred_list, dim=0).numpy()

    best_state = None
    best_metric_value = None
    best_metric_name = None
    best_val_metrics = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch()

        val_pred = predict(loader_dict["val"])
        val_metrics = task.evaluate(val_pred, val_table)

        if best_metric_name is None:
            best_metric_name, higher_is_better = choose_best_metric(val_metrics)
            best_metric_value = -math.inf if higher_is_better else math.inf

        current = val_metrics[best_metric_name]
        improved = (
            current > best_metric_value
            if higher_is_better
            else current < best_metric_value
        )

        if improved:
            best_metric_value = current
            best_val_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val={val_metrics}"
        )

    model.load_state_dict(best_state)

    return {
        "best_metric": best_metric_name,
        "best_val_metrics": best_val_metrics,
        "test_metrics": None,
    }




def metric_sort_reverse(task_type, metric_name):
    if task_type == "regression" and metric_name == "mae":
        return True

    if task_type == "classification" and metric_name == "accuracy":
        return False

    if task_type == "classification" and metric_name == "bce":
        return True

    raise ValueError((task_type, metric_name))



# 
def exact_rerank_candidates_by_metric(
    model,
    batch,
    entity_table,
    candidates,
    task_type="regression",
    eval_metric="mae",
):
    clean_metric = eval_metric_on_batch(
        model=model,
        batch=batch,
        entity_table=entity_table,
        task_type=task_type,
        metric_name=eval_metric,
    )

    scored = []

    for c in tqdm(candidates, desc=f"Exact rerank by {eval_metric}", leave=False):
        tmp_batch = apply_global_rewirings(batch, [c])

        attacked_metric = eval_metric_on_batch(
            model=model,
            batch=tmp_batch,
            entity_table=entity_table,
            task_type=task_type,
            metric_name=eval_metric,
        )

        c = dict(c)
        c["clean_metric"] = float(clean_metric)
        c["attacked_metric"] = float(attacked_metric)

        if task_type == "regression":
            c["delta_metric"] = float(attacked_metric - clean_metric)
        else:
            c["delta_metric"] = np.nan

        scored.append(c)

    scored = sorted(
        scored,
        key=lambda x: x["attacked_metric"],
        reverse=metric_sort_reverse(task_type, eval_metric),
    )

    return scored

def apply_global_rewirings(batch, selected):
    adv_batch = batch.clone()

    for rel in sorted(set(c["relation"] for c in selected), key=str):
        rel_selected = [c for c in selected if c["relation"] == rel]

        if len(rel_selected) == 0:
            continue

        reverse_rel = default_reverse_relation(rel)

        adv_batch = apply_forward_rewirings(
            batch=adv_batch,
            attacked_relation=rel,
            selected_rewirings=rel_selected,
            reverse_relation=reverse_rel,
        )

    return adv_batch

def select_unique_relation_child(order, budget):
    selected = []
    used = set()

    for c in order:
        key = (str(c["relation"]), int(c["child"]))

        if key in used:
            continue

        selected.append(c)
        used.add(key)

        if len(selected) >= int(budget):
            break

    return selected



def compute_diversity_stats(selected, all_relations=None):
    """
    Diversity measured AFTER final selection.
    selected = final list of rewiring operations.
    """

    n_selected = len(selected)

    if n_selected == 0:
        return {
            "n_relations_touched": 0,
            "relation_coverage": 0.0,
            "max_relation_share": 0.0,
            "n_unique_children": 0,
            "unique_child_ratio": 0.0,
            "relation_edit_counts": {},
        }

    rel_counter = Counter(c["relation"] for c in selected)

    n_relations_touched = len(rel_counter)

    if all_relations is None:
        relation_coverage = np.nan
    else:
        relation_coverage = n_relations_touched / max(len(all_relations), 1)

    max_relation_share = max(rel_counter.values()) / n_selected

    unique_children = len(set(c["child"] for c in selected))
    unique_child_ratio = unique_children / n_selected

    return {
        "n_relations_touched": int(n_relations_touched),
        "relation_coverage": float(relation_coverage),
        "max_relation_share": float(max_relation_share),
        "n_unique_children": int(unique_children),
        "unique_child_ratio": float(unique_child_ratio),
        "relation_edit_counts": dict(rel_counter),
    }




import numpy as np


def random_budget_partition(relations, B, rng):
    counts = rng.multinomial(
        B,
        [1 / len(relations)] * len(relations),
    )

    return {
        rel: int(c)
        for rel, c in zip(relations, counts)
    }


def random_multirelational_rewirings(
    batch,
    relations,
    B,
    seed=0,
):
    rng = np.random.default_rng(seed)

    budget_per_rel = random_budget_partition(
        relations,
        B,
        rng,
    )

    candidates = []

    for rel in relations:

        edge_index = batch.edge_index_dict[rel]

        children = edge_index[0].cpu().numpy()
        parents = edge_index[1].cpu().numpy()

        possible_parents = np.unique(parents)

        child_to_parent = {
            int(c): int(p)
            for c, p in zip(children, parents)
        }

        selected_children = rng.choice(
            list(child_to_parent.keys()),
            size=min(
                budget_per_rel[rel],
                len(child_to_parent),
            ),
            replace=False,
        )

        for child in selected_children:

            old_parent = child_to_parent[child]

            new_parent = rng.choice(possible_parents)

            while new_parent == old_parent:
                new_parent = rng.choice(possible_parents)

            candidates.append({
                "relation": rel,
                "attacked_relation": rel,
                "reverse_relation": default_reverse_relation(rel),

                "child": int(child),

                "old_dst": int(old_parent),
                "new_dst": int(new_parent),

                "score": 0.0,
            })

    return candidates