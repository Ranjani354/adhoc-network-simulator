import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from copy import deepcopy
from itertools import combinations
import pandas as pd
import time

st.set_page_config(layout="wide")
st.title("💡 Advanced Emergency Response Ad Hoc Network Simulator")

# --- UI / Simulation Parameters ---
col1, col2 = st.columns(2)
with col1:
    num_nodes = st.slider("Number of Mobile Nodes", 5, 30, 15)
    area_size = st.slider("Area Size (meters)", 50, 500, 200)
    communication_range = st.slider("Communication Range (meters)", 10, 100, 40)
    failure_prob = st.slider("Node Failure Probability per Round", 0.0, 0.2, 0.05)
    rounds = st.slider("Simulation Rounds", 10, 100, 30)
    messages_per_round = st.slider("Messages per Round", 1, 5, 2)

with col2:
    routing_protocol = st.selectbox("Routing Protocol", ["Shortest Path", "Flooding", "Energy-Aware"])
    mobility_model = st.selectbox("Node Mobility Model", ["Random Walk", "Group Mobility"])
    show_paths = st.checkbox("Show Last Message Path", True)
    enable_replay = st.checkbox("Step-by-step Replay", False)
    random_seed = st.number_input("Random Seed (for reproducibility)", 0, 9999, 42)

# Energy model parameters
initial_energy = st.number_input("Initial Node Energy (units)", 10, 1000, 100)
tx_energy = st.number_input("Energy per Transmission (units)", 1, 10, 2)
rx_energy = st.number_input("Energy per Reception (units)", 1, 10, 1)

# Set random seed
random.seed(random_seed)
np.random.seed(random_seed)

# --- Node Initialization ---
@st.cache_data(show_spinner=False, persist=True)
def initialize_nodes(num_nodes, area_size, initial_energy):
    nodes = {}
    for i in range(num_nodes):
        nodes[i] = {
            "pos": [random.uniform(0, area_size), random.uniform(0, area_size)],
            "alive": True,
            "energy": float(initial_energy),
            "delivered": 0,
            "failed": 0,
            "energy_history": [initial_energy],
            "messages_sent": 0,
            "messages_received": 0
        }
    return nodes

if 'nodes' not in st.session_state or st.button("🔄 Reset Simulation"):
    st.session_state.nodes = initialize_nodes(num_nodes, area_size, initial_energy)
    st.session_state.history = []
    st.session_state.event_log = []
    st.session_state.current_round = 0

nodes = st.session_state.nodes
history = st.session_state.history
event_log = st.session_state.event_log

# --- Mobility Models ---
def move_nodes(nodes_dict, area_size, model):
    if model == "Random Walk":
        for n in nodes_dict.values():
            if n["alive"]:
                theta = random.uniform(0, 2 * math.pi)
                step = random.uniform(0, 8)
                n["pos"][0] = min(area_size, max(0, n["pos"][0] + step * math.cos(theta)))
                n["pos"][1] = min(area_size, max(0, n["pos"][1] + step * math.sin(theta)))
    elif model == "Group Mobility":
        theta = random.uniform(0, 2 * math.pi)
        for n in nodes_dict.values():
            if n["alive"]:
                step = random.uniform(4, 8)
                n["pos"][0] = min(area_size, max(0, n["pos"][0] + step * math.cos(theta)))
                n["pos"][1] = min(area_size, max(0, n["pos"][1] + step * math.sin(theta)))

# --- Node Failure ---
def apply_failure(nodes_dict, failure_prob):
    for i, n in nodes_dict.items():
        if n["alive"] and random.random() < failure_prob:
            n["alive"] = False
            st.session_state.event_log.append(f"Round {st.session_state.current_round}: Node {i} failed (random failure).")

# --- Energy Update ---
def update_energy(nodes_dict, path, tx_energy, rx_energy):
    if not path: return
    for idx, node in enumerate(path):
        if not nodes_dict[node]["alive"]: continue
        if idx == 0 and len(path) > 1:
            nodes_dict[node]["energy"] -= tx_energy
            nodes_dict[node]["messages_sent"] += 1
        elif idx == len(path)-1 and len(path) > 1:
            nodes_dict[node]["energy"] -= rx_energy
            nodes_dict[node]["messages_received"] += 1
        elif 0 < idx < len(path)-1:
            nodes_dict[node]["energy"] -= (tx_energy + rx_energy)
        elif len(path) == 1:
            nodes_dict[node]["energy"] -= rx_energy
            nodes_dict[node]["messages_received"] += 1
        # Node death
        if nodes_dict[node]["energy"] <= 0 and nodes_dict[node]["alive"]:
            nodes_dict[node]["alive"] = False
            st.session_state.event_log.append(f"Round {st.session_state.current_round}: Node {node} died (energy depleted).")
        # record energy history
        nodes_dict[node]["energy_history"].append(max(0, nodes_dict[node]["energy"]))

# --- Energy Color ---
def compute_color(node, initial_energy):
    if not node["alive"]: return "red"
    ratio = max(0.0, node["energy"] / float(initial_energy))
    if ratio > 0.6: return "green"
    elif ratio > 0.3: return "yellow"
    else: return "orange"

# --- Flooding Routing ---
def flooding(G, src, dst, max_hops=10):
    visited = set([src])
    queue = [(src, [src])]
    while queue:
        node, path = queue.pop(0)
        if node == dst: return path
        if len(path) >= max_hops: continue
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

# --- Energy-aware Shortest Path ---
def energy_aware_path(G, nodes_dict, src, dst):
    for u, v in G.edges():
        avg_energy = (nodes_dict[u]["energy"] + nodes_dict[v]["energy"]) / 2
        G[u][v]["weight"] = 1.0 / max(avg_energy, 1)
    try:
        return nx.shortest_path(G, src, dst, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

# --- Simulation Loop ---
if enable_replay:
    replay_round = st.number_input("Replay round", 1, rounds, 1)
    st.session_state.current_round = int(replay_round)
else:
    st.session_state.current_round = int(rounds)

nodes_sim = deepcopy(nodes)
delivery_stats = []
failures_per_round = []

G_last = nx.Graph()

for t in range(st.session_state.current_round):
    st.session_state.current_round = t
    move_nodes(nodes_sim, area_size, mobility_model)
    apply_failure(nodes_sim, failure_prob)

    alive_nodes = [i for i, n in nodes_sim.items() if n["alive"]]
    G = nx.Graph()
    G.add_nodes_from(alive_nodes)
    for i, j in combinations(alive_nodes, 2):
        if math.hypot(nodes_sim[i]["pos"][0]-nodes_sim[j]["pos"][0],
                      nodes_sim[i]["pos"][1]-nodes_sim[j]["pos"][1]) <= communication_range:
            G.add_edge(i, j)
    G_last = G

    # Multi-message per round
    for _ in range(messages_per_round):
        if not alive_nodes: break
        src = random.choice(alive_nodes)
        dst = 0 if 0 in alive_nodes else random.choice(alive_nodes)
        path = None
        if routing_protocol == "Shortest Path":
            try:
                path = nx.shortest_path(G, src, dst)
            except: path = None
        elif routing_protocol == "Flooding":
            path = flooding(G, src, dst)
        elif routing_protocol == "Energy-Aware":
            path = energy_aware_path(G, nodes_sim, src, dst)

        if path:
            delivery_stats.append(len(path)-1)
            history.append({"round": t, "src": src, "dst": dst, "path": path, "delivered": True})
            st.session_state.event_log.append(f"Round {t}: Message {src}->{dst} delivered via {path} ({len(path)-1} hops).")
            update_energy(nodes_sim, path, tx_energy, rx_energy)
            nodes_sim[src]["delivered"] += 1
        else:
            delivery_stats.append(None)
            history.append({"round": t, "src": src, "dst": dst, "path": [], "delivered": False})
            st.session_state.event_log.append(f"Round {t}: Message {src}->{dst} delivery failed.")

    failures_per_round.append(len([n for n in nodes_sim.values() if not n["alive"]]))

# --- Network Plot (with fixed color length) ---
fig, ax = plt.subplots(figsize=(6,6))
alive_nodes = list(G_last.nodes())
pos = {i: nodes_sim[i]["pos"] for i in alive_nodes}
node_color_map = ["gold" if i==0 else compute_color(nodes_sim[i], initial_energy) for i in alive_nodes]
nx.draw(G_last, pos, with_labels=True, node_color=node_color_map, ax=ax, node_size=500)

# Draw dead nodes
for i in nodes_sim:
    if not nodes_sim[i]["alive"]:
        ax.scatter(nodes_sim[i]["pos"][0], nodes_sim[i]["pos"][1], marker='x', s=200, color='red', linewidths=3)

# Highlight last message path
last_success = next((h for h in reversed(history) if h["delivered"]), None)
if show_paths and last_success and len(last_success["path"])>=2:
    path_edges = list(zip(last_success["path"], last_success["path"][1:]))
    nx.draw_networkx_edges(G_last, pos, edgelist=path_edges, ax=ax, width=4, edge_color='blue')

ax.set_title(f"Network Topology (Round {st.session_state.current_round})")
ax.set_xlim(0, area_size)
ax.set_ylim(0, area_size)
st.pyplot(fig)

# --- Node Table & Metrics ---
with st.expander("📊 Node Details & Metrics"):
    node_table = []
    for i in sorted(nodes_sim.keys()):
        n = nodes_sim[i]
        node_table.append({
            "Node": i,
            "Status": "Alive" if n["alive"] else "Dead",
            "Energy": f"{n['energy']:.1f}",
            "Delivered": n["delivered"],
            "Messages Sent": n["messages_sent"],
            "Messages Received": n["messages_received"]
        })
    df_nodes = pd.DataFrame(node_table).set_index("
