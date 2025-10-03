import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

# Simulation parameters
st.title("Dynamic Emergency Response Ad Hoc Network Simulator")
num_nodes = st.slider("Number of Mobile Nodes", 5, 30, 15)
area_size = st.slider("Area Size (meters)", 50, 500, 200)
communication_range = st.slider("Communication Range (meters)", 10, 100, 40)
failure_prob = st.slider("Node Failure Probability per Round", 0.0, 0.2, 0.05)
rounds = st.slider("Simulation Rounds", 10, 100, 30)
show_paths = st.checkbox("Show Message Path", True)

# Initialize node positions and status
@st.cache_data(show_spinner=False, persist=True)
def initialize_nodes(num_nodes, area_size):
    nodes = {}
    for i in range(num_nodes):
        nodes[i] = {
            "pos": [random.uniform(0, area_size), random.uniform(0, area_size)],
            "alive": True
        }
    return nodes

if 'nodes' not in st.session_state or st.button("Reset Simulation"):
    st.session_state.nodes = initialize_nodes(num_nodes, area_size)
    st.session_state.history = []

nodes = st.session_state.nodes
history = st.session_state.history

delivery_stats = []
G = nx.Graph()

for t in range(rounds):
    # Move nodes and simulate failure
    for i in nodes:
        if nodes[i]["alive"]:
            # Simulate random node failure
            if random.random() < failure_prob:
                nodes[i]["alive"] = False
                continue
            # Move node randomly
            theta = random.uniform(0, 2 * np.pi)
            step = random.uniform(0, 8)
            nodes[i]["pos"][0] = min(area_size, max(0, nodes[i]["pos"][0] + step * np.cos(theta)))
            nodes[i]["pos"][1] = min(area_size, max(0, nodes[i]["pos"][1] + step * np.sin(theta)))

    # Build network graph based on alive nodes and proximity
    G.clear()
    alive_nodes = [i for i in nodes if nodes[i]["alive"]]
    G.add_nodes_from(alive_nodes)
    for i in alive_nodes:
        for j in alive_nodes:
            if i < j:
                dist = np.linalg.norm(np.array(nodes[i]["pos"]) - np.array(nodes[j]["pos"]))
                if dist < communication_range:
                    G.add_edge(i, j)

    # Message passing simulation
    if alive_nodes:
        src = random.choice(alive_nodes)
        dst = 0 if 0 in alive_nodes else alive_nodes[0]  # Command center is node 0 if alive
        try:
            path = nx.shortest_path(G, src, dst)
            delivery_stats.append(len(path)-1)
            history.append({"round": t, "src": src, "dst": dst, "path": path, "delivered": True})
        except nx.NetworkXNoPath:
            delivery_stats.append(None)
            history.append({"round": t, "src": src, "dst": dst, "path": [], "delivered": False})

# Visualization
fig, ax = plt.subplots(figsize=(6,6))

# Only plot alive nodes; keep color list in same order as alive_nodes
alive_colors = []
alive_pos = {}
for i in alive_nodes:
    if i == 0:
        alive_colors.append('yellow')  # Command center
    else:
        alive_colors.append('green')
    alive_pos[i] = nodes[i]["pos"]

nx.draw(G, alive_pos, with_labels=True, node_color=alive_colors, ax=ax, node_size=500)

# Draw failed nodes as red 'X'
for i in nodes:
    if not nodes[i]["alive"]:
        ax.scatter(*nodes[i]["pos"], color='red', marker='x', s=200, label='_nolegend_')

ax.set_title("Network Topology (Final Round)")
ax.set_xlim(0, area_size)
ax.set_ylim(0, area_size)

# Highlight last successful message path
if show_paths and history and history[-1]["delivered"]:
    path = history[-1]["path"]
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G, alive_pos, edgelist=path_edges, ax=ax, width=4, edge_color='blue')

st.pyplot(fig)

# Performance metrics
successful = [h for h in delivery_stats if h is not None]
st.write(f"**Delivery Ratio:** {len(successful)}/{len(delivery_stats)} ({100*len(successful)/len(delivery_stats):.1f}%)")
if successful:
    st.write(f"**Average Path Length (Hops):** {np.mean(successful):.2f}")
else:
    st.write("**Average Path Length (Hops):** N/A")

failures = [i for i in nodes if not nodes[i]["alive"]]
st.write(f"**Node Failures:** {len(failures)} / {num_nodes} ({100*len(failures)/num_nodes:.1f}%)")

st.write("**Legend:** Green = Alive Node, Red = Failed Node, Yellow = Command Center (node 0)")

st.info("Try changing parameters (number of nodes, area, range, failure probability) and pressing 'Reset Simulation' to explore different scenarios.")
