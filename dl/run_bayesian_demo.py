"""
Bayesian reasoning demo. Two networks:
  (A) the REAL AML evidence DAG (results/graph/evidence_graph_after_repair.graphml), ILLUSTRATIVE parameters;
  (B) a HYPOTHETICAL textbook DAG with a confounder (Age) to show why P(Y|X) differs from P(Y|do(X)).
All CPD numbers are illustrative, not learned from data. See src/bayesian/network.py for assumptions.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.bayesian import (CAUSAL_DISCLAIMER, build_bayesian_network, fit_parameters_from_data, query_conditional,
                          query_intervention, simulate_data_from_network)

ROOT = Path(__file__).resolve().parents[1]


def load_real_dag() -> nx.DiGraph:
    g = nx.read_graphml(ROOT / "results/graph/evidence_graph_after_repair.graphml")
    d = nx.DiGraph()
    for u, v, data in g.edges(data=True):
        d.add_edge(u, v, polarity=data.get("polarity", "increases"))
    return d


def hypothetical_dag() -> nx.DiGraph:
    g = nx.DiGraph()
    for u, v, p in [("Age", "BMICat", "increases"), ("Age", "Outcome", "increases"), ("BMICat", "Outcome", "increases"),
                    ("PhysActivity", "BMICat", "decreases")]:
        g.add_edge(u, v, polarity=p)
    return g


def draw(g, path, title):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    pos = nx.spring_layout(g, seed=3)
    nx.draw_networkx(g, pos, ax=ax, node_color="#cfe3ff", node_size=2200, font_size=8, arrows=True, arrowsize=18)
    ax.set_title(title, fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    out = ROOT / "dl/results/bayesian"
    out.mkdir(parents=True, exist_ok=True)
    res = {"disclaimer": CAUSAL_DISCLAIMER}
    real = load_real_dag()
    bn = build_bayesian_network(real)
    res["real_aml_dag"] = {"edges": [f"{u}->{v}" for u, v in real.edges()], "parameter_source": bn.parameter_source,
                           "note": "single extracted causal edge: too sparse for meaningful intervention analysis",
                           "P(Outcome|BMICat=yes)": query_conditional(bn, "Outcome", {"BMICat": "yes"}),
                           "P(Outcome|do(BMICat=yes))": query_intervention(bn, "Outcome", {"BMICat": "yes"})["distribution"]}
    draw(real, out / "bn_real_aml_dag.png", "Real AML evidence DAG")
    h = hypothetical_dag()
    bh = build_bayesian_network(h)
    obs = query_conditional(bh, "Outcome", {"BMICat": "yes"})
    do = query_intervention(bh, "Outcome", {"BMICat": "yes"})["distribution"]
    base = query_conditional(bh, "Outcome", {})
    prot = query_conditional(bh, "Outcome", {"PhysActivity": "yes"})
    df = simulate_data_from_network(bh, 20000, seed=42)
    est = build_bayesian_network(h, data=df)
    res["hypothetical_confounded_dag"] = {
        "edges": [f"{u}->{v}" for u, v in h.edges()], "parameter_source": bh.parameter_source,
        "P(Outcome=yes)": base["yes"], "P(Outcome=yes|BMICat=yes) [observational]": obs["yes"],
        "P(Outcome=yes|do(BMICat=yes)) [interventional, backdoor via Age]": do["yes"],
        "confounding_gap": obs["yes"] - do["yes"], "P(Outcome=yes|PhysActivity=yes)": prot["yes"],
        "estimation_check": {"data": "SYNTHETIC forward samples (n=20000) from the illustrative network; tests the fitting code only",
                             "P(Outcome=yes|BMICat=yes) true": obs["yes"],
                             "P(Outcome=yes|BMICat=yes) re-estimated": query_conditional(est, "Outcome", {"BMICat": "yes"})["yes"]}}
    draw(h, out / "bn_hypothetical_dag.png", "Hypothetical DAG with confounder (illustrative)")
    (out / "bayesian_demo_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
