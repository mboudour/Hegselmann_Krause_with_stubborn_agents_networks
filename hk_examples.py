"""Exact checks of the worked examples in the manuscript (d = 1).
Example A: limit that is not a fixed point of T.
Example B: smallest instance where the truncated energy increases under heterogeneous bounds.
Example C: Etesami's slow first switch (n = 3, two stubborn agents)."""
import itertools
from fractions import Fraction as Fr

def hk(x, adj, sig):            # sig[i] = confidence bound of agent i (0 = stubborn)
    y = list(x)
    for i in range(len(x)):
        if sig[i] == 0: continue
        nb = [i] + [j for j in adj[i] if abs(x[i]-x[j]) <= sig[i]]
        y[i] = sum(x[j] for j in nb) / len(nb)
    return y

# ---- Example A: path s - i - j - k, sigma = 1
adj = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
sig = [Fr(0), Fr(1), Fr(1), Fr(1)]
x = [Fr(0), Fr(11, 10), Fr(1), Fr(9, 10)]
active_si = False
for t in range(200):
    if abs(x[1]-x[0]) <= 1: active_si = True
    x = hk(x, adj, sig)
print("A: x(200) =", [float(v) for v in x], "| edge s-i ever active:", active_si,
      "| x_i(200)-1 =", float(x[1]-1))
xs = [Fr(0), Fr(1), Fr(1), Fr(1)]
print("A: T(x*) =", [str(v) for v in hk(xs, adj, sig)])

# ---- Example B: search small instances (path/triangle, 3 agents) where V increases
def V(x, edges, s): return sum(min((x[i]-x[j])**2, s**2) for i, j in edges)
best = None
grid = [Fr(k, 4) for k in range(0, 9)]
for edges in ([(0, 1), (1, 2)], [(0, 1), (1, 2), (0, 2)]):
    adj = {0: [], 1: [], 2: []}
    for a, b in edges: adj[a].append(b); adj[b].append(a)
    for s0, s1, s2 in itertools.product([Fr(1, 2), Fr(1), Fr(3, 2)], repeat=3):
        if len({s0, s1, s2}) == 1: continue
        smax = max(s0, s1, s2)
        for x in itertools.product(grid, repeat=3):
            x = list(x); y = hk(x, adj, [s0, s1, s2])
            if V(y, edges, smax) > V(x, edges, smax):
                best = (edges, [str(v) for v in (s0, s1, s2)], [str(v) for v in x],
                        str(V(x, edges, smax)), str(V(y, edges, smax)))
                break
        if best: break
    if best: break
print("B: heterogeneous bounds, V(x) -> V(Tx):", best)

# ---- Example C: complete graph on 3 agents, agents 0,1 stubborn, agent 2 regular, sigma_2 = 1
for m in (2, 3, 4, 5):
    x = [Fr(-1) + Fr(1, 2**(2*m)), Fr(0), Fr(1)]
    adj = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    sig = [Fr(0), Fr(0), Fr(1)]
    t, first = 0, None
    def act(x): return abs(x[2]-x[0]) <= 1
    a0 = act(x)
    while t < 5000:
        x = hk(x, adj, sig); t += 1
        if act(x) != a0: first = t; break
    print(f"C: m={m}: first switch of the interaction graph at t={first}")
