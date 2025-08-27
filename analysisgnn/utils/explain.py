from typing import Tuple
from torch_geometric.explain import Explainer, HeteroExplanation
from torch_geometric.explain.config import ExplanationType, ModelMode


def hetero_fidelity(
    explainer: Explainer,
    explanation: HeteroExplanation,
) -> Tuple[float, float]:
    r"""Evaluates the fidelity of an
    :class:`~torch_geometric.explain.Explainer` given an
    :class:`~torch_geometric.explain.Explanation`, as described in the
    `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for
    Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper.

    Fidelity evaluates the contribution of the produced explanatory subgraph
    to the initial prediction, either by giving only the subgraph to the model
    (fidelity-) or by removing it from the entire graph (fidelity+).
    The fidelity scores capture how good an explainable model reproduces the
    natural phenomenon or the GNN model logic.

    For **phenomenon** explanations, the fidelity scores are given by:

    .. math::
        \textrm{fid}_{+} &= \frac{1}{N} \sum_{i = 1}^N
        \| \mathbb{1}(\hat{y}_i = y_i) -
        \mathbb{1}( \hat{y}_i^{G_{C \setminus S}} = y_i) \|

        \textrm{fid}_{-} &= \frac{1}{N} \sum_{i = 1}^N
        \| \mathbb{1}(\hat{y}_i = y_i) -
        \mathbb{1}( \hat{y}_i^{G_S} = y_i) \|

    For **model** explanations, the fidelity scores are given by:

    .. math::
        \textrm{fid}_{+} &= 1 - \frac{1}{N} \sum_{i = 1}^N
        \mathbb{1}( \hat{y}_i^{G_{C \setminus S}} = \hat{y}_i)

        \textrm{fid}_{-} &= 1 - \frac{1}{N} \sum_{i = 1}^N
        \mathbb{1}( \hat{y}_i^{G_S} = \hat{y}_i)

    Args:
        explainer (Explainer): The explainer to evaluate.
        explanation (Explanation): The explanation to evaluate.
    """
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    edge_mask = {k: (explanation[k].edge_mask > 0).long() for k in explanation.edge_types}
    node_mask = {k: (explanation[k].node_mask > 0).long() for k in explanation.node_types}
    kwargs = {}
    # kwargs = {key: explanation[key] for key in explanation._model_args}

    y = explanation.target
    if explainer.explanation_type == ExplanationType.phenomenon:
        y_hat = explainer.get_prediction(
            explanation.x_dict,
            explanation.edge_index_dict,
            **kwargs,
        )
        y_hat = explainer.get_target(y_hat)

    explain_y_hat = explainer.get_masked_prediction(
        explanation.x_dict,
        explanation.edge_index_dict,
        node_mask,
        edge_mask,
        # **kwargs,
    )
    explain_y_hat = explainer.get_target(explain_y_hat)

    complement_y_hat = explainer.get_masked_prediction(
        explanation.x_dict,
        explanation.edge_index_dict,
        {k: 1 - m for k, m in node_mask.items()},
        {k: 1 - m for k, m in edge_mask.items()},
        # **kwargs,
    )
    complement_y_hat = explainer.get_target(complement_y_hat)

    if explanation.get('index') is not None:
        y = y[explanation.index]
        if explainer.explanation_type == ExplanationType.phenomenon:
            y_hat = y_hat[explanation.index]
        explain_y_hat = explain_y_hat[explanation.index]
        complement_y_hat = complement_y_hat[explanation.index]

    if explainer.explanation_type == ExplanationType.model:
        pos_fidelity = 1. - (complement_y_hat == y).float().mean()
        neg_fidelity = 1. - (explain_y_hat == y).float().mean()
    else:
        pos_fidelity = ((y_hat == y).float() -
                        (complement_y_hat == y).float()).abs().mean()
        neg_fidelity = ((y_hat == y).float() -
                        (explain_y_hat == y).float()).abs().mean()

    return float(pos_fidelity), float(neg_fidelity)