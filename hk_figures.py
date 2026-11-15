"""Figures for the manuscript.
Fig. 1: limit structure of an exact d = 1 run on a 14-agent graph (stubborn, trapped and free
        agents; active, inactive and critical edges), drawn with networkx + pygraphviz (neato).
Fig. 2: max-norm distance to the limit versus time with switching times, d = 1 and d = 2."""
import numpy as np, networkx as nx, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fractions import Fraction as Fr
from hk_critical_edges import step, frozen_limit
from hk_balanced_2d import step as step2, d2
from hk_certify_2d import limit_of_pattern, certify

import os
OUT = "figures/"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(4); sig = Fr(1, 4)

def run1(G, S, x, T):
    adj = {v: list(G[v]) for v in G}; U = [v for v in G if v not in S]
    Ep = [e for e in G.edges() if not (e[0] in S and e[1] in S)]
    traj = [x]
    for _ in range(T): x = step(x, adj, U, sig); traj.append(x)
    return traj, U, Ep

# ---- search a run with trapped + free (size>=2) components and a critical edge at the limit
chosen = None
for trial in range(4000):
    n = 14; G = nx.connected_watts_strogatz_graph(n, 4, 0.3, seed=int(rng.integers(1e9)))
    S = set(rng.choice(n, size=2, replace=False).tolist())
    x = [Fr(int(rng.integers(0, 25)), 24) for _ in range(n)]
    for (i, j) in list(G.edges())[:6]:
        if j not in S: x[j] = x[i] + (sig if rng.random() < .5 else -sig)
    traj, U, Ep = run1(G, S, x, 120)
    y, N = frozen_limit(traj[-1], G, S, U, sig)
    H = nx.Graph(); H.add_nodes_from(U)
    H.add_edges_from((i, j) for i in U for j in N[i] if j in U and j != i)
    comps = list(nx.connected_components(H))
    trapped = [c for c in comps if any(j in S for i in c for j in N[i])]
    free = [c for c in comps if c not in trapped and len(c) >= 2]
    crit = [e for e in Ep if abs(y[e[0]]-y[e[1]]) == sig]
    nsw = sum(1 for t in range(1, len(traj)) if any((abs(traj[t][a]-traj[t][b]) <= sig) != (abs(traj[t-1][a]-traj[t-1][b]) <= sig) for a, b in Ep))
    if trapped and free and crit and nsw >= 3:
        chosen = (G, S, U, Ep, traj, y, N, trapped, free, crit); break
assert chosen, "no example found"
G, S, U, Ep, traj, y, N, trapped, free, crit = chosen
A = nx.nx_agraph.to_agraph(nx.Graph())
A.graph_attr.update(overlap="false", sep="+12", size="5,4")
A.node_attr.update(fontsize="10", fontname="Helvetica", width="0.45", fixedsize="true")
tset = set().union(*trapped)
for v in G:
    shape = "square" if v in S else "circle"
    fill = "#444444" if v in S else ("#9ecae1" if v in tset else "#fdd0a2")
    font = "white" if v in S else "black"
    A.add_node(v, label=f"{float(y[v]):.2f}", shape=shape, style="filled", fillcolor=fill,
               fontcolor=font)
act = set(); 
for i in U:
    for j in N[i]:
        if j != i: act.add((min(i, j), max(i, j)))
for a, b in G.edges():
    e = (min(a, b), max(a, b))
    if a in S and b in S: A.add_edge(a, b, color="#bbbbbb", style="dotted"); continue
    if e in [(min(p, q), max(p, q)) for p, q in crit]:
        A.add_edge(a, b, color="#d62728", style="dashed", penwidth="1.6")
    elif e in act: A.add_edge(a, b, color="black", penwidth="1.4")
    else: A.add_edge(a, b, color="#bbbbbb", style="dotted")
A.layout(prog="neato"); A.draw(OUT + "fig_limit_structure.pdf")
switches = sum(1 for t in range(1, len(traj)) if any(
    (abs(traj[t][a]-traj[t][b]) <= sig) != (abs(traj[t-1][a]-traj[t-1][b]) <= sig) for a, b in Ep))
print("Fig1: n=14, sigma=1/4, S=", sorted(S), "| trapped sizes", [len(c) for c in trapped],
      "| free sizes", [len(c) for c in free], "| critical edges", len(crit), "| switches", switches)

# ---- Fig. 2
fig, ax = plt.subplots(figsize=(6.0, 3.2))
xinf = y   # exact limit from the frozen pattern
err1 = [max(abs(float(z[i]-xinf[i])) for i in U) for z in traj[:80]]
sw1 = [t for t in range(1, 80) if any((abs(traj[t][a]-traj[t][b]) <= sig) !=
                                      (abs(traj[t-1][a]-traj[t-1][b]) <= sig) for a, b in Ep)]
ax.semilogy(range(80), [max(e, 1e-17) for e in err1], color="#1f77b4", label="$d=1$ (Fig. 1 run)")
for t in sw1: ax.axvline(t, color="#1f77b4", alpha=.25, lw=.8)
# d = 2 run: random geometric-like graph, 3 stubborn agents
rng2 = np.random.default_rng(12); n2 = 12
G2 = nx.connected_watts_strogatz_graph(n2, 4, 0.3, seed=7); S2 = {0, 5, 9}; U2 = [v for v in G2 if v not in S2]
adj2 = {v: list(G2[v]) for v in G2}
x2 = [(Fr(int(rng2.integers(0, 21)), 10), Fr(int(rng2.integers(0, 21)), 10)) for _ in range(n2)]
Ep2 = [e for e in G2.edges() if not (e[0] in S2 and e[1] in S2)]
tr2 = [x2]
for _ in range(120): x2 = step2(x2, adj2, U2); tr2.append(x2)
xi2 = limit_of_pattern(tr2[-1], adj2, S2, U2, Fr(1))   # exact limit of the terminal pattern
print('Fig2: d=2 freezing certified at t=120:', certify(tr2[-1], xi2, Ep2, Fr(1)))
err2 = [max(max(abs(float(z[i][0]-xi2[i][0])), abs(float(z[i][1]-xi2[i][1]))) for i in U2) for z in tr2[:81]]
sw2 = [t for t in range(1, 81) if any((d2(tr2[t][a], tr2[t][b]) <= 1) != (d2(tr2[t-1][a], tr2[t-1][b]) <= 1)
                                      for a, b in Ep2)]
ax.semilogy(range(81), [max(e, 1e-17) for e in err2], color="#d62728", ls="--", label="$d=2$")
for t in sw2: ax.axvline(t, color="#d62728", alpha=.25, lw=.8, ls="--")
ax.set_xlabel("$t$"); ax.set_ylabel(r"$\max_i \|x_i(t)-x_i^*\|_\infty$")
ax.legend(frameon=False, fontsize=9); fig.tight_layout(); fig.savefig(OUT + "fig_convergence.pdf")
print("Fig2: d=1 switch times", sw1, "| d=2 switch times", sw2)
