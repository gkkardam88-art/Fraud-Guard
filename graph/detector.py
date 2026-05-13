"""
Graph-Based Fraud Detection
----------------------------
Models transactions as a directed graph:
  Nodes = accounts / cards / merchants
  Edges = individual transactions

Three detection algorithms:
  1. High-degree centrality  — money mule accounts with unusually many connections
  2. Cycle detection         — circular money flows (A→B→C→A) = laundering signal
  3. Dense community         — tightly connected clusters = coordinated fraud rings
"""

import logging
import networkx as nx
from typing import Any, Dict, List

log = logging.getLogger(__name__)


def build_graph(transactions: List[Dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for txn in transactions:
        src = txn.get("sender_id",   f"node_{txn.get('transaction_id','?')}_src")
        dst = txn.get("receiver_id", f"node_{txn.get('transaction_id','?')}_dst")
        G.add_node(src, node_type="account")
        G.add_node(dst, node_type="merchant")
        G.add_edge(src, dst,
                   amount=txn.get("amount", 0),
                   transaction_id=txn.get("transaction_id"))
    return G


def detect_high_degree_nodes(G: nx.DiGraph, threshold: int = 5) -> List[Dict]:
    suspicious = []
    for node in G.nodes():
        in_d  = G.in_degree(node)
        out_d = G.out_degree(node)
        if (in_d + out_d) >= threshold:
            suspicious.append({
                "node":              node,
                "in_degree":         in_d,
                "out_degree":        out_d,
                "total_connections": in_d + out_d,
                "risk_reason":       "High transaction velocity — potential money mule",
            })
    return sorted(suspicious, key=lambda x: x["total_connections"], reverse=True)


def detect_cycles(G: nx.DiGraph, max_length: int = 6) -> List[List]:
    try:
        return [c for c in nx.simple_cycles(G) if len(c) <= max_length]
    except Exception as e:
        log.warning(f"Cycle detection error: {e}")
        return []


def detect_communities(G: nx.DiGraph) -> List[Dict]:
    undirected = G.to_undirected()
    results = []
    for component in nx.connected_components(undirected):
        if len(component) < 3:
            continue
        sub     = undirected.subgraph(component)
        density = nx.density(sub)
        results.append({
            "members":    list(component),
            "size":       len(component),
            "density":    round(density, 3),
            "suspicious": density > 0.5,
            "risk_reason": "Dense cluster — coordinated fraud ring suspected" if density > 0.5 else "Normal cluster",
        })
    return sorted(results, key=lambda x: x["density"], reverse=True)


def analyze_graph(transactions: List[Dict]) -> Dict[str, Any]:
    G = build_graph(transactions)
    log.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    high_degree   = detect_high_degree_nodes(G)
    cycles        = detect_cycles(G)
    communities   = detect_communities(G)

    flagged = set(n["node"] for n in high_degree)
    flagged |= set(node for cycle in cycles for node in cycle)

    suspicious_communities = [c for c in communities if c["suspicious"]]

    return {
        "graph_stats": {
            "total_nodes":      G.number_of_nodes(),
            "total_edges":      G.number_of_edges(),
            "suspicious_nodes": len(flagged),
        },
        "high_degree_nodes":     high_degree[:10],
        "detected_cycles":       [{"cycle": c, "length": len(c)} for c in cycles[:5]],
        "suspicious_communities": suspicious_communities[:5],
        "flagged_accounts":      list(flagged),
        "summary": (
            f"Analyzed {G.number_of_nodes()} nodes and {G.number_of_edges()} edges. "
            f"Found {len(flagged)} suspicious accounts, {len(cycles)} money-flow cycles, "
            f"and {len(suspicious_communities)} dense fraud communities."
        ),
    }
