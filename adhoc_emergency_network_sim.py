import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import math

# --- Simulation Parameters ---
st.title("Advanced Emergency Response Ad Hoc Network Simulator")
num_nodes = st.slider("Number of Mobile Nodes", 5, 30, 15)
area_size = st.slider("Area Size (meters)", 50, 500, 200)
communication_range = st.slider("Communication Range (meters)", 10, 100, 40)
failure_prob = st.slider("Node Failure Probability per Round", 0.0, 0.2, 0.05)
rounds = st.slider("Simulation Rounds", 10, 100, 30)
routing_protocol = st.selectbox("Routing Protocol", ["Shortest Path", "Flooding"])
mobility_model = st.selectbox("Node Mobility Model", ["Random Walk", "Group Mobility"])
show_paths = st.checkbox("Show Message Path", True)
enable_replay = st.checkbox("Step-by-step Replay", False)

# Energy model parameters
initial_energy = st.number_input("Initial Node Energy (units)", 10, 1000, 100)
tx_energy = st.number_input("Energy per Transmission (units)", 1, 10, 2)
rx_energy = st.number_input("Energy per Reception (units)", 1, 10, 1)

# --- Node Initialization ---
@st.cache_data(show_spinner=False, persist=True)
def initialize_nodes(num_nodes, area_size, initial_energy):
    nodes = {}
    for i in range(num_nodes):
        nodes[i] = {
            "pos": [random.uniform(0, area_size), random.uniform(0, area_size)],
            "alive": True,
            "energy": initial_energy,
            "delivered": 0,
            "failed": 0,
        }
    return nodes

if 'nodes' not in st.session_state or st.button("Reset Simulation"):
    st.session_state.nodes = initialize_nodes(num_nodes, area_size, initial_energy)
    st.session_state.history = []
    st.session_state.event_log = []
    st.session_state.current_round = 0

nodes = st.session_state.nodes
history = st.session_state.history
event_log = st.session_state.event_log
current_round = st.session_state.current_round

# --- Mobility Models ---
def move_nodes(nodes, area_size, model):
    if model == "Random Walk":
        for i in nodes:
            if nodes[i]["alive"]:
                theta = random.uniform(0, 2 * np.pi)
                step = random.uniform(0, 8)
                nodes[i]["pos"][0] = min(area_size, max(0, nodes[i]["pos"][0] + step * np.cos(theta)))
                nodes[i]["pos"][1] = min(area_size, max(0, nodes[i]["pos"][1] + step * np.sin(theta)))
    elif model == "Group Mobility":
        # All alive nodes move in roughly the same direction
        theta = random.uniform(0, 2 * np.pi)
        for i in nodes:
            if nodes[i]["alive"]:
                step = random.uniform(4, 8)
                nodes[i]["pos"][0] = min(area_size, max(0, nodes[i]["pos"][0] + step * np.cos(theta)))
                nodes[i]["pos"][1] = min(area_size, max(0, nodes[i]["pos"][1] + step * np.sin(theta)))

def apply_failure(nodes, failure_prob):
    for i in nodes:
        if nodes[i]["alive"] and random.random() < failure_prob:
            nodes[i]["alive"] = False
            st.session_state.event_log.append(f"Round: Node {i} failed (random failure).")

def update_energy(nodes, path, tx_energy, rx_energy):
    # For each transmission, deduct tx_energy from sender, rx_energy from receiver (except last)
    if not path:
        return
    for idx, node in enumerate(path):
        if idx == 0:
            nodes[node]["energy"] -= tx_energy
        elif idx == len(path)-1:
            nodes[node]["energy"] -= rx_energy
        else:
            nodes[node]["energy"] -= (tx_energy + rx_energy)
        if nodes[node]["energy"] <= 0 and nodes[node]["alive"]:
            nodes[node]["alive"] = False
            st.session_state.event_log.append(f"Node {node} died (energy depleted).")

def compute_color(node, initial_energy):
    # Energy color gradient: green (full) -> yellow -> red (empty)
    if not node["alive"]:
        return "red"
    ratio = max(0, node["energy"] / initial_energy)
    if ratio > 0.6:
        return "green"
    elif ratio > 0.3:
        return "yellow"
    else:
        return "orange"

def flooding(G, src, dst, max_hops=10):
    # Basic flooding to find if dst is reachable (not optimal, for demo)
    visited = set()
    queue = [(src, [src])]
    while queue:
        node, path = queue.pop(0)
        if node == dst:
            return path
        if len(path) > max_hops:
            continue
        for neighbor in G.neighbors(node):
            if neighbor not in path:
                queue.append((neighbor, path + [neighbor]))
    return None

# --- Simulation Loop ---
if enable_replay:
    # Replay mode: run up to current_round only, allow stepping
    replay_round = st.number_input("Replay round", 1, rounds, 1)
    st.session_state.current_round = replay_round
else:
    st.session_state.current_round = rounds

delivery_stats = []
failures_per_round = []
history.clear()
event_log_display = []

# Copy node states for simulation
nodes_sim = {i: nodes[i].copy() for i in nodes}

for t in range(st.session_state.current_round):
    move_nodes(nodes_sim, area_size, mobility_model)
    apply_failure(nodes_sim, failure_prob)
    # Build network graph based on alive nodes and proximity
    G = nx.Graph()
    alive_nodes = [i for i in nodes_sim if nodes_sim[i]["alive"]]
    G.add_nodes_from(alive_nodes)
    for i in alive_nodes:
        for j in alive_nodes:
            if i < j:
                dist = np.linalg.norm(np.array(nodes_sim[i]["pos"]) - np.array(nodes_sim[j]["pos"]))
                if dist < communication_range:
                    G.add_edge(i, j)
    # Message passing simulation
    if alive_nodes:
        src = random.choice(alive_nodes)
        dst = 0 if 0 in alive_nodes else alive_nodes[0]  # Command center is node 0 if alive
        path = None
        if routing_protocol == "Shortest Path":
            try:
                path = nx.shortest_path(G, src, dst)
            except nx.NetworkXNoPath:
                path = None
        elif routing_protocol == "Flooding":
            path = flooding(G, src, dst)
        if path:
            delivery_stats.append(len(path)-1)
            history.append({"round": t, "src": src, "dst": dst, "path": path, "delivered": True})
            event_log.append(f"Round {t}: Message delivered {src}->{dst} via {path} ({len(path)-1} hops).")
            # Energy update
            update_energy(nodes_sim, path, tx_energy, rx_energy)
            nodes_sim[src]["delivered"] += 1
        else:
            delivery_stats.append(None)
            history.append({"round": t, "src": src, "dst": dst, "path": [], "delivered": False})
            event_log.append(f"Round {t}: Message delivery failed {src}->{dst}.")
    failures = [i for i in nodes_sim if not nodes_sim[i]["alive"]]
    failures_per_round.append(len(failures))

# --- Visualization ---
fig, ax = plt.subplots(figsize=(6,6))

# Only plot alive nodes; keep color list in same order as alive_nodes
alive_nodes = [i for i in nodes_sim if nodes_sim[i]["alive"]]
alive_colors = []
alive_pos = {}
for i in alive_nodes:
    alive_colors.append(compute_color(nodes_sim[i], initial_energy))
    alive_pos[i] = nodes_sim[i]["pos"]

nx.draw(G, alive_pos, with_labels=True, node_color=alive_colors, ax=ax, node_size=500)

# Draw failed nodes as red 'X'
for i in nodes_sim:
    if not nodes_sim[i]["alive"]:
        ax.scatter(*nodes_sim[i]["pos"], color='red', marker='x', s=200, label='_nolegend_')

ax.set_title(f"Network Topology (Round {st.session_state.current_round})")
ax.set_xlim(0, area_size)
ax.set_ylim(0, area_size)

# Highlight last successful message path
last_success = None
for h in reversed(history):
    if h["delivered"]:
        last_success = h
        break
if show_paths and last_success:
    path = last_success["path"]
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G, alive_pos, edgelist=path_edges, ax=ax, width=4, edge_color='blue')

st.pyplot(fig)

# --- Node Info ---
with st.expander("Click for Node Details Table"):
    import pandas as pd
    node_table = []
    for i in nodes_sim:
        status = "Alive" if nodes_sim[i]["alive"] else "Dead"
        node_table.append({
            "Node": i,
            "Status": status,
            "Energy": f"{nodes_sim[i]['energy']:.1f}",
            "Delivered": nodes_sim[i]["delivered"]
        })
    st.dataframe(pd.DataFrame(node_table).set_index("Node"))

# --- Performance metrics ---
successful = [h for h in delivery_stats if h is not None]
st.write(f"**Delivery Ratio:** {len(successful)}/{len(delivery_stats)} ({100*len(successful)/len(delivery_stats):.1f}%)")
if successful:
    st.write(f"**Average Path Length (Hops):** {np.mean(successful):.2f}")
else:
    st.write("**Average Path Length (Hops):** N/A")

failures = [i for i in nodes_sim if not nodes_sim[i]["alive"]]
st.write(f"**Node Failures:** {len(failures)} / {num_nodes} ({100*len(failures)/num_nodes:.1f}%)")

st.write("**Legend:** Green = High Energy, Yellow = Medium, Orange = Low, Red/X = Failed Node, Yellow = Command Center (node 0)")

# --- Event Log ---
with st.expander("Simulation Event Log"):
    for e in event_log[-30:]:
        st.write(e)

# --- Replay Controls ---
if enable_replay:
    st.warning("Replay mode is enabled. Use the slider above to step through simulation rounds.")

st.info("Try changing parameters, mobility, routing, and pressing 'Reset Simulation' to explore different scenarios. Each run is randomized!")
