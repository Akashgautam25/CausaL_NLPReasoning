"""
Bayesian-network layer (Phase 2) built with pgmpy on top of the evidence DAG.

WHAT THIS IS / IS NOT
  * The structure comes from literature-extracted causal claims. It is a *hypothesised* causal graph.
  * CPD numbers come from one of two sources, recorded in `model.parameter_source`:
      - "ILLUSTRATIVE": documented, hand-set noisy-OR style numbers. NOT learned from data, NOT empirical.
      - "ESTIMATED": maximum-likelihood / Bayesian estimates from a supplied DataFrame (e.g. BRFSS/NHANES).
  * do() queries are only valid if the DAG is correct and there is no unmeasured confounding (causal sufficiency).
    Extracting edges from text does not establish identification, so intervention results are conditional
    on those assumptions and must not be read as measured causal effects.
"""
from __future__ import annotations

import itertools
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import CausalInference, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

STATES = ["no", "yes"]
CAUSAL_DISCLAIMER = (
    "Intervention (do) results are conditional on the assumed DAG being correct and on causal sufficiency "
    "(no unmeasured confounders). A graph extracted from observational text does not establish causal "
    "identification; with ILLUSTRATIVE parameters the numbers demonstrate the mechanics only.")


def _illustrative_cpd(node: str, parents: list[str], polarity: dict[str, str], base: float, strength: float,
                      prior: float) -> TabularCPD:
    """Root: P(yes)=prior. Child: noisy-OR-like, each active 'increases' parent adds `strength` to the base
    rate (combined as 1-prod(1-s)); each active 'decreases' parent scales the probability by (1-strength)."""
    if not parents:
        return TabularCPD(node, 2, [[1 - prior], [prior]], state_names={node: STATES})
    cols = []
    for combo in itertools.product([0, 1], repeat=len(parents)):
        inc = [strength for p, v in zip(parents, combo) if v and polarity.get(p) != "decreases"]
        dec = [p for p, v in zip(parents, combo) if v and polarity.get(p) == "decreases"]
        p_yes = 1 - (1 - base) * np.prod([1 - s for s in inc]) if inc else base
        p_yes *= (1 - strength) ** len(dec)
        p_yes = float(min(max(p_yes, 0.001), 0.999))
        cols.append([1 - p_yes, p_yes])
    values = np.array(cols).T
    return TabularCPD(node, 2, values.tolist(), evidence=parents, evidence_card=[2] * len(parents),
                      state_names={node: STATES, **{p: STATES for p in parents}})


def build_bayesian_network(dag: nx.DiGraph, base_rate: float = 0.10, strength: float = 0.30, root_prior: float = 0.30,
                           data: pd.DataFrame | None = None, extra_nodes: list[str] | None = None):
    """
    Convert an acyclic evidence graph to a pgmpy DiscreteBayesianNetwork (binary variables).
    If `data` is given, CPDs are ESTIMATED from it (Bayesian estimator, uniform Dirichlet prior);
    otherwise ILLUSTRATIVE parameters are used. Raises ValueError for cyclic input.
    """
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("graph must be acyclic to form a Bayesian network")
    nodes = sorted(set(dag.nodes()) | set(extra_nodes or []))
    if not nodes:
        raise ValueError("empty graph")
    bn = DiscreteBayesianNetwork()
    bn.add_nodes_from(nodes)
    bn.add_edges_from(dag.edges())
    if data is not None:
        bn = fit_parameters_from_data(bn, data)
        bn.parameter_source = "ESTIMATED"
    else:
        for n in nodes:
            parents = sorted(bn.get_parents(n))
            pol = {p: dag[p][n].get("polarity", "increases") for p in parents}
            bn.add_cpds(_illustrative_cpd(n, parents, pol, base_rate, strength, root_prior))
        bn.parameter_source = "ILLUSTRATIVE"
    bn.check_model()
    return bn


def fit_parameters_from_data(bn: DiscreteBayesianNetwork, data: pd.DataFrame, estimator: str = "bayes"):
    missing = set(bn.nodes()) - set(data.columns)
    if missing:
        raise ValueError(f"data lacks columns for nodes: {sorted(missing)}")
    d = data[list(bn.nodes())].copy()
    for c in d.columns:
        d[c] = d[c].map({0: "no", 1: "yes", "no": "no", "yes": "yes"})
    states = {c: STATES for c in d.columns}
    if estimator == "mle":
        cpds = MaximumLikelihoodEstimator(bn, d, state_names=states).get_parameters()
    else:
        cpds = BayesianEstimator(bn, d, state_names=states).get_parameters(prior_type="dirichlet", pseudo_counts=1)
    bn.cpds = []
    bn.add_cpds(*cpds)
    return bn


def simulate_data_from_network(bn, n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Forward-sample synthetic rows (0/1). SYNTHETIC: used only to test estimation code, never as evidence."""
    from pgmpy.sampling import BayesianModelSampling
    df = BayesianModelSampling(bn).forward_sample(size=n, seed=seed, show_progress=False)
    return df.replace({"no": 0, "yes": 1}).astype(int)


def query_conditional(bn, target: str, evidence: dict[str, str] | None = None) -> dict[str, float]:
    """P(target | evidence) by exact variable elimination (observational)."""
    q = VariableElimination(bn).query([target], evidence=evidence or {}, show_progress=False)
    return {s: float(v) for s, v in zip(q.state_names[target], q.values)}


def query_intervention(bn, target: str, do: dict[str, str]) -> dict[str, Any]:
    """P(target | do(...)) via backdoor adjustment (pgmpy CausalInference). See CAUSAL_DISCLAIMER."""
    ci = CausalInference(bn)
    q = ci.query([target], do=do, show_progress=False)
    return {"distribution": {s: float(v) for s, v in zip(q.state_names[target], q.values)},
            "parameter_source": getattr(bn, "parameter_source", "unknown"), "disclaimer": CAUSAL_DISCLAIMER}
