"""Animate per-node RGAT predictions on one social-learning episode."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter
from torch_geometric.data import Data

from src.models.recurrent_gat_agent import RecurrentGATAgent
from src.signal_generator import PrivateSignalGenerator
from src.training_pipeline import _episode_to_model_inputs

COLOR_CORRECT = "#22c55e"
COLOR_WRONG = "#ef4444"
COLOR_EDGE = "#cbd5e1"
# Above this count, individual edges are omitted (complete graphs use a ring hint).
MAX_VIEWER_EDGES = 2500
LARGE_GRAPH_NODE_THRESHOLD = 150
VIEWER_NODE_RING_SCALE = 0.72
VIEWER_PRIVATE_RING_SCALE = 1.12
VIEWER_SVG_MARGIN = 1.34


@dataclass(frozen=True)
class EpisodeTrace:
    """Per-round node-level rollout for visualization."""

    theta: int
    num_nodes: int
    max_horizon: int
    topology: str
    signal_quality: float
    private_signals: np.ndarray
    predictions: np.ndarray
    probs_one: np.ndarray
    comm_messages: np.ndarray
    correct: np.ndarray

    @property
    def error_rates(self) -> np.ndarray:
        return 1.0 - self.correct.mean(axis=1)

    @property
    def agreement_fractions(self) -> np.ndarray:
        counts = []
        for t in range(self.max_horizon):
            preds = self.predictions[t]
            _, freq = np.unique(preds, return_counts=True)
            counts.append(float(freq.max()) / self.num_nodes)
        return np.asarray(counts)


def pyg_to_networkx(graph_data: Data) -> nx.Graph:
    """Convert undirected PyG graph to NetworkX (deduplicate directed edges)."""
    edge_index = graph_data.edge_index.cpu().numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(int(graph_data.num_nodes)))
    seen: set[tuple[int, int]] = set()
    for src, dst in edge_index.T:
        u, v = int(src), int(dst)
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            graph.add_edge(u, v)
    return graph


def circular_layout(
    num_nodes: int, *, radius: float = 1.0
) -> dict[int, tuple[float, float]]:
    """Place nodes evenly on a circle (node i at angle 2π·i/N, starting at top)."""
    positions: dict[int, tuple[float, float]] = {}
    for i in range(num_nodes):
        angle = 2.0 * np.pi * i / num_nodes - np.pi / 2.0
        positions[i] = (radius * float(np.cos(angle)), radius * float(np.sin(angle)))
    return positions


def node_radius_for_count(num_nodes: int) -> float:
    """SVG node radius scaled so large graphs remain readable."""
    return float(np.clip(80.0 / np.sqrt(max(num_nodes, 1)), 3.0, 14.0))


def graph_edge_list(graph: nx.Graph) -> list[list[int]]:
    return [[int(u), int(v)] for u, v in graph.edges()]


def viewer_edge_payload(
    graph: nx.Graph,
    *,
    topology: str,
) -> tuple[list[list[int]], str, int]:
    """Return edges for the HTML viewer, hint mode, and true edge count."""
    edge_count = graph.number_of_edges()
    topo_lower = topology.lower()
    if "complete" in topo_lower:
        return [], "complete", edge_count
    if edge_count > MAX_VIEWER_EDGES:
        return [], "dense", edge_count
    return graph_edge_list(graph), "graph", edge_count


def compute_unanimous_consensus(trace: EpisodeTrace) -> dict[str, Any]:
    """Per-round gossip consensus flags and first unanimous round (1-based).

    Round 1 uses only each node's private signal (outbound messages are still
    zero), so it is a prior belief, not post-gossip consensus. Unanimous
    consensus is therefore tracked from round 2 onward.
    """
    per_round: list[dict[str, Any]] = []
    first_unanimous_t: int | None = None
    first_unanimous_correct: bool | None = None

    for t in range(trace.max_horizon):
        preds = trace.predictions[t]
        all_same = bool(preds.min() == preds.max())
        label = int(preds[0]) if all_same else -1
        after_gossip = t > 0
        unanimous = all_same and after_gossip
        unanimous_correct = unanimous and label == trace.theta
        unanimous_wrong = unanimous and label != trace.theta
        _, freq = np.unique(preds, return_counts=True)
        agreement_fraction = float(freq.max()) / float(trace.num_nodes)
        per_round.append(
            {
                "t": t + 1,
                "after_gossip": after_gossip,
                "unanimous": unanimous,
                "unanimous_correct": unanimous_correct,
                "unanimous_wrong": unanimous_wrong,
                "agreement_fraction": agreement_fraction,
            }
        )
        if unanimous and first_unanimous_t is None:
            first_unanimous_t = t + 1
            first_unanimous_correct = unanimous_correct

    return {
        "per_round": per_round,
        "first_unanimous_t": first_unanimous_t,
        "first_unanimous_correct": first_unanimous_correct,
    }


def layout_positions(graph: nx.Graph, *, topology: str, seed: int) -> dict[int, np.ndarray]:
    """Stable node positions for GIF export (always circular)."""
    del topology, seed
    return {
        node: np.asarray(coords, dtype=float)
        for node, coords in circular_layout(graph.number_of_nodes()).items()
    }


def rollout_episode(
    *,
    model: RecurrentGATAgent,
    graph_data: Data,
    episode: dict[str, int | torch.Tensor],
    device: torch.device,
    signal_quality: float,
    topology: str,
) -> EpisodeTrace:
    """Run one episode and collect per-node predictions round by round."""
    num_nodes = int(graph_data.num_nodes)
    x_sequences, _ = _episode_to_model_inputs(
        episode, num_nodes=num_nodes, device=device
    )
    horizon = int(x_sequences.shape[1])
    theta = int(episode["theta"])
    private_signals = episode["private_signals"].cpu().numpy()

    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(
            x_sequences,
            graph_data.edge_index,
            max_horizon=horizon,
        )
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1).cpu().numpy()
        probs_one = probs[..., 1].cpu().numpy()
        comm_messages = predictions.astype(np.float64)

    if was_training:
        model.train()

    correct = predictions == theta
    return EpisodeTrace(
        theta=theta,
        num_nodes=num_nodes,
        max_horizon=horizon,
        topology=topology,
        signal_quality=signal_quality,
        private_signals=private_signals,
        predictions=predictions,
        probs_one=probs_one,
        comm_messages=comm_messages,
        correct=correct,
    )


def sample_episode(
    *,
    num_nodes: int,
    max_horizon: int,
    signal_quality: float,
    seed: int,
) -> dict[str, int | torch.Tensor]:
    generator = PrivateSignalGenerator(signal_quality=signal_quality, default_seed=seed)
    return generator.generate_episode(
        num_nodes=num_nodes,
        max_horizon=max_horizon,
        seed=seed,
    )


def pack_binary_bitmap(values: np.ndarray) -> str:
    """Pack (T, N) 0/1 values into a compact base64 bitmap for the HTML viewer."""
    flat = values.astype(np.uint8).flatten()
    return base64.b64encode(np.packbits(flat).tobytes()).decode("ascii")


def pack_correct_bitmap(correct: np.ndarray) -> str:
    """Pack (T, N) booleans into a compact base64 bitmap for the HTML viewer."""
    return pack_binary_bitmap(correct.astype(np.uint8))


def _episode_viewer_entry(
    trace: EpisodeTrace,
    *,
    episode_seed: int,
    eval_plot: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "episode_seed": episode_seed,
        "theta": trace.theta,
        "correct_shape": [trace.max_horizon, trace.num_nodes],
        "correct_packed": pack_correct_bitmap(trace.correct),
        "private_shape": [trace.max_horizon, trace.num_nodes],
        "private_packed": pack_binary_bitmap(trace.private_signals),
        "consensus": compute_unanimous_consensus(trace),
    }
    if eval_plot:
        entry["eval_plot"] = eval_plot
    return entry


def rollouts_cache_path(output_path: str | Path) -> Path:
    """Sidecar cache path for precomputed episode rollouts."""
    return Path(output_path).with_suffix(".rollouts.npz")


def save_rollouts_cache(
    path: str | Path,
    traces: list[EpisodeTrace],
    *,
    episode_seeds: list[int],
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Persist rollout results so HTML can be rebuilt without re-running the model."""
    cache = Path(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    trace0 = traces[0]
    checkpoint_mtime = (
        float(Path(checkpoint_path).stat().st_mtime) if checkpoint_path is not None else -1.0
    )
    np.savez_compressed(
        cache,
        episode_seeds=np.asarray(episode_seeds, dtype=np.int64),
        theta=np.asarray([t.theta for t in traces], dtype=np.int64),
        correct=np.stack([t.correct.astype(np.uint8) for t in traces]),
        private_signals=np.stack([t.private_signals.astype(np.uint8) for t in traces]),
        num_nodes=np.int64(trace0.num_nodes),
        max_horizon=np.int64(trace0.max_horizon),
        signal_quality=np.float64(trace0.signal_quality),
        topology=np.asarray(trace0.topology),
        checkpoint_mtime=np.float64(checkpoint_mtime),
    )
    return cache


def rollouts_cache_is_valid(
    path: str | Path,
    *,
    episode_seeds: list[int],
    num_nodes: int,
    max_horizon: int,
    signal_quality: float,
    topology: str,
    checkpoint_path: str | Path | None = None,
) -> bool:
    cache = Path(path)
    if not cache.exists():
        return False
    try:
        data = np.load(cache, allow_pickle=False)
        if list(data["episode_seeds"].tolist()) != list(episode_seeds):
            return False
        if int(data["num_nodes"]) != num_nodes:
            return False
        if int(data["max_horizon"]) != max_horizon:
            return False
        if abs(float(data["signal_quality"]) - signal_quality) > 1e-9:
            return False
        if str(data["topology"].item()) != topology:
            return False
        if checkpoint_path is not None:
            ckpt_mtime = Path(checkpoint_path).stat().st_mtime
            if float(data["checkpoint_mtime"]) < ckpt_mtime - 1e-6:
                return False
        if "private_signals" not in data:
            return False
    except (OSError, KeyError, ValueError, TypeError):
        return False
    return True


def load_rollouts_cache(path: str | Path) -> tuple[list[EpisodeTrace], list[int]]:
    data = np.load(path, allow_pickle=False)
    episode_seeds = [int(x) for x in data["episode_seeds"].tolist()]
    num_nodes = int(data["num_nodes"])
    max_horizon = int(data["max_horizon"])
    signal_quality = float(data["signal_quality"])
    topology = str(data["topology"].item())
    traces: list[EpisodeTrace] = []
    for idx, seed in enumerate(episode_seeds):
        correct = data["correct"][idx].astype(bool)
        private_signals = data["private_signals"][idx].astype(np.int64)
        theta = int(data["theta"][idx])
        predictions = np.where(correct, theta, 1 - theta).astype(np.int64)
        traces.append(
            EpisodeTrace(
                theta=theta,
                num_nodes=num_nodes,
                max_horizon=max_horizon,
                topology=topology,
                signal_quality=signal_quality,
                private_signals=private_signals,
                predictions=predictions,
                probs_one=predictions.astype(float),
                comm_messages=predictions.astype(float),
                correct=correct,
            )
        )
    return traces, episode_seeds


def _viewer_payload(
    traces: list[EpisodeTrace],
    graph: nx.Graph,
    *,
    episode_seeds: list[int],
    eval_plots: dict[int, str] | None = None,
) -> dict[str, Any]:
    if len(traces) != len(episode_seeds):
        raise ValueError("traces and episode_seeds must have the same length.")
    trace0 = traces[0]
    positions = circular_layout(trace0.num_nodes)
    edges, edge_hint, edge_count = viewer_edge_payload(
        graph,
        topology=trace0.topology,
    )
    return {
        "num_nodes": trace0.num_nodes,
        "max_horizon": trace0.max_horizon,
        "topology": trace0.topology,
        "signal_quality": trace0.signal_quality,
        "node_radius": node_radius_for_count(trace0.num_nodes),
        "positions": [[positions[i][0], positions[i][1]] for i in range(trace0.num_nodes)],
        "edges": edges,
        "edge_hint": edge_hint,
        "edge_count": edge_count,
        "episodes": [
            _episode_viewer_entry(
                trace,
                episode_seed=seed,
                eval_plot=(eval_plots or {}).get(seed),
            )
            for trace, seed in zip(traces, episode_seeds, strict=True)
        ],
    }


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Episode viewer — {title}</title>
<style>
  :root {{
    --green: #22c55e;
    --red: #ef4444;
    --edge: #cbd5e1;
    --bg: #0f172a;
    --panel: #1e293b;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 1.25rem 1.5rem 1.5rem;
  }}
  .page-header {{
    max-width: 1280px;
    margin: 0 auto 1rem;
  }}
  h1 {{ font-size: 1.1rem; font-weight: 600; margin: 0 0 0.35rem; }}
  .meta {{ color: var(--muted); font-size: 0.85rem; margin: 0; }}
  .viewer-layout {{
    display: flex;
    align-items: flex-start;
    gap: 1.25rem;
    max-width: 1280px;
    margin: 0 auto;
  }}
  .viewer-graph {{
    flex: 1 1 0;
    min-width: 0;
  }}
  .viewer-sidebar {{
    flex: 0 0 min(380px, 34vw);
    min-width: 280px;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }}
  #graph-wrap {{
    position: relative;
    background: var(--panel);
    border-radius: 12px;
    padding: 1rem;
    width: 100%;
    aspect-ratio: 1;
  }}
  .graph-stage {{
    position: relative;
    width: 100%;
    height: 100%;
  }}
  #edge-canvas,
  svg {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: block;
  }}
  #edge-canvas {{ pointer-events: none; z-index: 0; }}
  svg {{ z-index: 1; }}
  .node {{ stroke: none; }}
  .node.correct {{ fill: var(--green); }}
  .node.wrong {{ fill: var(--red); }}
  .private-signal {{
    stroke: none;
    opacity: 1;
  }}
  .private-signal.inactive {{ fill: #334155; opacity: 0.35; }}
  .private-signal.correct {{ fill: var(--green); opacity: 1; }}
  .private-signal.wrong {{ fill: var(--red); opacity: 1; }}
  #private-signals.hidden {{ visibility: hidden; }}
  .graph-legend {{
    width: 100%;
    margin-top: 0.55rem;
    color: var(--muted);
    font-size: 0.78rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1rem;
    align-items: center;
    justify-content: flex-start;
  }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 0.35rem; }}
  .legend-swatch {{
    width: 0.65rem;
    height: 0.65rem;
    border-radius: 999px;
    display: inline-block;
    border: 1px solid #334155;
  }}
  .legend-swatch.pred-correct {{ background: var(--green); }}
  .legend-swatch.pred-wrong {{ background: var(--red); }}
  .legend-swatch.priv-correct {{ background: var(--green); }}
  .legend-swatch.priv-wrong {{ background: var(--red); }}
  @keyframes node-flash {{
    0% {{ opacity: 0.25; }}
    40% {{ opacity: 1; }}
    100% {{ opacity: 1; }}
  }}
  .node.flash {{ animation: node-flash 0.38s ease-out; }}
  .controls {{
    width: 100%;
    background: var(--panel);
    border-radius: 12px;
    padding: 1rem;
  }}
  .round-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.65rem 0.75rem;
    margin-bottom: 0.75rem;
  }}
  .private-row {{
    margin-bottom: 0.75rem;
  }}
  .private-row label {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.85rem;
    color: var(--muted);
  }}
  #round-slider {{
    flex: 1 1 8rem;
    min-width: 0;
    width: 100%;
    accent-color: var(--accent);
  }}
  #round-label {{
    flex: 0 0 100%;
    font-variant-numeric: tabular-nums;
    font-size: 0.85rem;
    color: var(--muted);
  }}
  button {{
    background: var(--panel);
    color: var(--text);
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 0.4rem 0.9rem;
    cursor: pointer;
    font-size: 0.85rem;
  }}
  button:hover:not(:disabled) {{ border-color: var(--accent); }}
  button:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  .nav-btns {{ display: flex; gap: 0.4rem; flex-shrink: 0; }}
  .episode-row {{
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
  }}
  .episode-row label {{
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 0.35rem;
    font-size: 0.85rem;
    color: var(--muted);
  }}
  #episode-select {{
    background: var(--bg);
    color: var(--text);
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 0.35rem 0.5rem;
    font-size: 0.85rem;
    width: 100%;
  }}
  #status {{
    background: var(--bg);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    line-height: 1.5;
    margin-bottom: 0.75rem;
  }}
  #timeline-wrap {{
    position: relative;
    height: 36px;
    background: #334155;
    border-radius: 6px;
    overflow: hidden;
    cursor: pointer;
  }}
  .tl-seg {{
    position: absolute;
    top: 0; bottom: 0;
    background: rgba(56, 189, 248, 0.25);
  }}
  .tl-marker {{
    position: absolute;
    top: 0; bottom: 0;
    width: 2px;
    background: var(--accent);
    pointer-events: none;
  }}
  .tl-consensus {{
    position: absolute;
    top: 4px; bottom: 4px;
    width: 3px;
    border-radius: 1px;
    pointer-events: none;
  }}
  .tl-consensus.ok {{ background: var(--green); }}
  .tl-consensus.bad {{ background: var(--red); }}
  #playhead {{
    position: absolute;
    top: 0; bottom: 0;
    width: 2px;
    background: #fff;
    pointer-events: none;
    transform: translateX(-50%);
  }}
  .eval-panel {{
    width: 100%;
    margin: 0;
    background: var(--panel);
    border-radius: 12px;
    padding: 0.75rem 1rem 1rem;
  }}
  @media (max-width: 900px) {{
    .viewer-layout {{
      flex-direction: column;
    }}
    .viewer-sidebar {{
      flex: 1 1 auto;
      width: 100%;
      min-width: 0;
    }}
    #round-label {{
      flex: 0 0 auto;
    }}
  }}
  .eval-panel summary {{
    cursor: pointer;
    font-weight: 600;
    font-size: 0.95rem;
  }}
  .eval-caption {{
    color: var(--muted);
    font-size: 0.8rem;
    margin: 0.6rem 0 0.75rem;
    line-height: 1.45;
  }}
  .eval-plot {{
    width: 100%;
    height: auto;
    display: block;
    border-radius: 8px;
    background: #fff;
  }}
</style>
</head>
<body>
<header class="page-header">
  <h1>Social learning episode</h1>
  <p class="meta">{meta}</p>
</header>
<main class="viewer-layout">
  <section class="viewer-graph" aria-label="Episode graph">
    <div id="graph-wrap">
      <div class="graph-stage">
        <canvas id="edge-canvas" aria-hidden="true"></canvas>
        <svg id="graph" viewBox="-{svg_margin} -{svg_margin} {svg_size} {svg_size}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
          <g id="private-signals" class="hidden"></g>
          <g id="nodes"></g>
        </svg>
      </div>
    </div>
    <p class="graph-legend">
      <span class="legend-item"><span class="legend-swatch pred-correct"></span> inner: prediction correct</span>
      <span class="legend-item"><span class="legend-swatch pred-wrong"></span> inner: prediction wrong</span>
      <span class="legend-item"><span class="legend-swatch priv-correct"></span> outer: private signal correct</span>
      <span class="legend-item"><span class="legend-swatch priv-wrong"></span> outer: private signal wrong</span>
    </p>
  </section>
  <aside class="viewer-sidebar" aria-label="Controls and plots">
    <div class="controls">
      <div class="episode-row">
        <label>Signal episode
          <select id="episode-select"></select>
        </label>
        <button type="button" id="new-signal-btn" title="Pick another pre-rendered private-signal draw">New signal</button>
      </div>
      <div id="status"></div>
      <div class="round-row">
        <div class="nav-btns">
          <button type="button" id="prev-btn" title="Previous round">←</button>
          <button type="button" id="play-btn">Play</button>
          <button type="button" id="next-btn" title="Next round">→</button>
        </div>
        <input type="range" id="round-slider" min="1" max="{max_horizon}" value="1" step="1"/>
        <span id="round-label">Round 1 / {max_horizon}</span>
      </div>
      <div class="private-row">
        <label><input type="checkbox" id="show-private-cb"/> Show private signals (outer ring, same round)</label>
      </div>
      <div id="timeline-wrap" title="Click to jump to round"></div>
    </div>
    {eval_plot_section}
    {summary_plot_section}
  </aside>
</main>
<script>
const DATA = {payload_json};

const svg = document.getElementById('graph');
const edgeCanvas = document.getElementById('edge-canvas');
const privateG = document.getElementById('private-signals');
const nodesG = document.getElementById('nodes');
const slider = document.getElementById('round-slider');
const roundLabel = document.getElementById('round-label');
const statusEl = document.getElementById('status');
const timeline = document.getElementById('timeline-wrap');
const playBtn = document.getElementById('play-btn');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const episodeSelect = document.getElementById('episode-select');
const newSignalBtn = document.getElementById('new-signal-btn');
const showPrivateCb = document.getElementById('show-private-cb');
const metaEl = document.querySelector('.meta');

const nodeRingScale = {node_ring_scale};
const privateRingScale = {private_ring_scale};
const r = (DATA.node_radius / 520) * nodeRingScale;
const PRIV_FILL = {{ inactive: '#334155', correct: '{color_correct}', wrong: '{color_wrong}' }};
const nodeEls = [];
const privateEls = [];

const privateFragment = document.createDocumentFragment();
const nodeFragment = document.createDocumentFragment();
DATA.positions.forEach((p, i) => {{
  const cx = p[0] * nodeRingScale;
  const cy = p[1] * nodeRingScale;
  const px = p[0] * privateRingScale;
  const py = p[1] * privateRingScale;
  const priv = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  priv.setAttribute('cx', String(px));
  priv.setAttribute('cy', String(py));
  priv.setAttribute('r', String(r));
  priv.classList.add('private-signal', 'inactive');
  priv.setAttribute('fill', PRIV_FILL.inactive);
  priv.dataset.idx = String(i);
  privateFragment.appendChild(priv);
  privateEls.push(priv);
  const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  circle.setAttribute('cx', cx);
  circle.setAttribute('cy', cy);
  circle.setAttribute('r', r);
  circle.classList.add('node');
  circle.dataset.idx = String(i);
  nodeFragment.appendChild(circle);
  nodeEls.push(circle);
}});
privateG.appendChild(privateFragment);
nodesG.appendChild(nodeFragment);

function edgeLineCoords(u, v) {{
  const p0 = DATA.positions[u];
  const p1 = DATA.positions[v];
  const x0 = p0[0] * nodeRingScale;
  const y0 = p0[1] * nodeRingScale;
  const x1 = p1[0] * nodeRingScale;
  const y1 = p1[1] * nodeRingScale;
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy);
  if (len < 1e-9) return [x0, y0, x1, y1];
  const ux = dx / len;
  const uy = dy / len;
  return [x0 + ux * r, y0 + uy * r, x1 - ux * r, y1 - uy * r];
}}

function viewToCanvas(x, y) {{
  const pt = svg.createSVGPoint();
  pt.x = x;
  pt.y = y;
  const screenPt = pt.matrixTransform(svg.getScreenCTM());
  const canvasRect = edgeCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  return [
    (screenPt.x - canvasRect.left) * dpr,
    (screenPt.y - canvasRect.top) * dpr,
  ];
}}

function drawEdges() {{
  const rect = edgeCanvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  const dpr = window.devicePixelRatio || 1;
  edgeCanvas.width = Math.round(rect.width * dpr);
  edgeCanvas.height = Math.round(rect.height * dpr);
  const ctx = edgeCanvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, edgeCanvas.width, edgeCanvas.height);
  ctx.strokeStyle = 'rgba(203, 213, 225, 0.4)';
  ctx.lineWidth = Math.max(0.5 * dpr, 1);
  ctx.lineCap = 'round';

  if (DATA.edge_hint === 'complete' || DATA.edge_hint === 'dense') {{
    const [cx, cy] = viewToCanvas(0, 0);
    const [rx, ry] = viewToCanvas(nodeRingScale, 0);
    const ringR = Math.hypot(rx - cx, ry - cy);
    ctx.beginPath();
    ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
    ctx.stroke();
    return;
  }}

  ctx.beginPath();
  for (const [u, v] of DATA.edges) {{
    const [x1, y1, x2, y2] = edgeLineCoords(u, v);
    const [cx1, cy1] = viewToCanvas(x1, y1);
    const [cx2, cy2] = viewToCanvas(x2, y2);
    ctx.moveTo(cx1, cy1);
    ctx.lineTo(cx2, cy2);
  }}
  ctx.stroke();
}}

drawEdges();
window.addEventListener('resize', drawEdges);

if (!DATA.episodes) {{
  DATA.episodes = [{{
    episode_seed: DATA.episode_seed ?? 0,
    theta: DATA.theta,
    correct: DATA.correct,
    consensus: DATA.consensus,
  }}];
}}

function unpackBitmap2D(ep, existingKey, shapeKey, packedKey, asBool) {{
  if (ep[existingKey]) return ep[existingKey];
  if (!ep[packedKey] || !ep[shapeKey]) return null;
  const [rounds, nodes] = ep[shapeKey];
  const raw = atob(ep[packedKey]);
  const bits = [];
  const needed = rounds * nodes;
  for (let i = 0; i < raw.length && bits.length < needed; i++) {{
    const byte = raw.charCodeAt(i);
    for (let b = 7; b >= 0 && bits.length < needed; b--) bits.push((byte >> b) & 1);
  }}
  const out = [];
  for (let t = 0; t < rounds; t++) {{
    const row = [];
    for (let n = 0; n < nodes; n++) {{
      const bit = bits[t * nodes + n];
      row.push(asBool ? Boolean(bit) : bit);
    }}
    out.push(row);
  }}
  return out;
}}

DATA.episodes.forEach((ep) => {{
  ep.correct = unpackBitmap2D(ep, 'correct', 'correct_shape', 'correct_packed', true);
  ep.private = unpackBitmap2D(ep, 'private', 'private_shape', 'private_packed', false);
}});

const playhead = document.createElement('div');
playhead.id = 'playhead';
timeline.appendChild(playhead);

const T = DATA.max_horizon;
let activeEpisodeIdx = 0;

function getEpisode() {{
  return DATA.episodes[activeEpisodeIdx];
}}

function updateMetaLine() {{
  const ep = getEpisode();
  const base = metaEl.dataset.base || metaEl.textContent;
  if (!metaEl.dataset.base) metaEl.dataset.base = base;
  metaEl.textContent = `${{base}} · episode seed=${{ep.episode_seed}} · θ=${{ep.theta}}`;
}}

function buildTimeline() {{
  timeline.querySelectorAll('.tl-consensus, .tl-marker').forEach((el) => el.remove());
  const consensus = getEpisode().consensus;
  consensus.per_round.forEach((row, idx) => {{
    if (!row.unanimous) return;
    const seg = document.createElement('div');
    seg.className = 'tl-consensus ' + (row.unanimous_correct ? 'ok' : 'bad');
    const left = ((idx) / T) * 100;
    const width = (1 / T) * 100;
    seg.style.left = left + '%';
    seg.style.width = Math.max(width, 0.4) + '%';
    timeline.appendChild(seg);
  }});
  if (consensus.first_unanimous_t != null) {{
    const m = document.createElement('div');
    m.className = 'tl-marker';
    m.style.left = ((consensus.first_unanimous_t - 0.5) / T) * 100 + '%';
    timeline.appendChild(m);
  }}
}}

function consensusText(tIdx) {{
  const consensus = getEpisode().consensus;
  const row = consensus.per_round[tIdx];
  if (!row.after_gossip) {{
    return 'Prior belief from <strong>private signal only</strong> — no gossip yet';
  }}
  const first = consensus.first_unanimous_t;
  const firstOk = consensus.first_unanimous_correct;
  let line = '';
  if (row.unanimous) {{
    line = row.unanimous_correct
      ? 'Unanimous consensus: <strong>yes (correct)</strong>'
      : 'Unanimous consensus: <strong>yes (wrong)</strong>';
  }} else {{
    line = 'Unanimous consensus: <strong>no</strong>';
  }}
  if (first != null) {{
    const tag = firstOk ? 'correct' : 'wrong';
    if (tIdx + 1 >= first) {{
      line += ` &nbsp;|&nbsp; first reached at <strong>t=${{first}}</strong> (${{tag}})`;
    }} else {{
      line += ` &nbsp;|&nbsp; first at <strong>t=${{first}}</strong> (${{tag}}) — not yet`;
    }}
  }} else {{
    line += ' &nbsp;|&nbsp; unanimous consensus <strong>never reached</strong>';
  }}
  return line;
}}

let currentRound = null;
let prevCorrect = null;
const largeGraph = DATA.num_nodes > {large_graph_threshold};

function flashNode(el) {{
  el.classList.remove('flash');
  el.addEventListener('animationend', () => el.classList.remove('flash'), {{ once: true }});
  requestAnimationFrame(() => el.classList.add('flash'));
}}

function triggerNodeFlash(correct) {{
  nodeEls.forEach((el, i) => {{
    if (largeGraph && prevCorrect !== null && prevCorrect[i] === correct[i]) return;
    flashNode(el);
  }});
}}

function updateEvalPlot() {{
  const evalImg = document.getElementById('eval-plot');
  if (!evalImg) return;
  const plot = getEpisode().eval_plot;
  if (plot) evalImg.src = plot;
}}

function setEpisode(idx) {{
  if (timer) {{
    clearInterval(timer);
    timer = null;
    playBtn.textContent = 'Play';
  }}
  activeEpisodeIdx = idx;
  episodeSelect.value = String(idx);
  buildTimeline();
  currentRound = null;
  prevCorrect = null;
  updateMetaLine();
  updateEvalPlot();
  renderRound(1);
}}

function pickNewSignal() {{
  if (DATA.episodes.length <= 1) return;
  let next = activeEpisodeIdx;
  while (next === activeEpisodeIdx) {{
    next = Math.floor(Math.random() * DATA.episodes.length);
  }}
  setEpisode(next);
}}

function renderPrivateSignals() {{
  const show = showPrivateCb.checked;
  privateG.classList.toggle('hidden', !show);
  if (!show) return;
  const ep = getEpisode();
  const tOneBased = currentRound ?? 1;
  const tIdx = tOneBased - 1;
  const privateRow = ep.private ? ep.private[tIdx] : null;
  const theta = ep.theta;
  privateEls.forEach((el, i) => {{
    el.classList.remove('inactive', 'correct', 'wrong');
    if (!privateRow) {{
      el.classList.add('inactive');
      el.setAttribute('fill', PRIV_FILL.inactive);
      return;
    }}
    const signalCorrect = Number(privateRow[i]) === theta;
    if (signalCorrect) {{
      el.classList.add('correct');
      el.setAttribute('fill', PRIV_FILL.correct);
    }} else {{
      el.classList.add('wrong');
      el.setAttribute('fill', PRIV_FILL.wrong);
    }}
  }});
}}

function renderRound(tOneBased) {{
  const ep = getEpisode();
  const tIdx = tOneBased - 1;
  const correct = ep.correct[tIdx];
  const roundChanged = currentRound !== null && currentRound !== tOneBased;
  currentRound = tOneBased;
  nodeEls.forEach((el, i) => {{
    el.classList.toggle('correct', correct[i]);
    el.classList.toggle('wrong', !correct[i]);
  }});
  if (roundChanged) triggerNodeFlash(correct);
  prevCorrect = correct.slice();
  const err = 1 - correct.filter(Boolean).length / DATA.num_nodes;
  const agree = ep.consensus.per_round[tIdx].agreement_fraction;
  statusEl.innerHTML =
    `<strong>θ=${{ep.theta}}</strong> &nbsp;|&nbsp; error=${{(err * 100).toFixed(1)}}%` +
    ` &nbsp;|&nbsp; agreement=${{(agree * 100).toFixed(1)}}%<br/>` +
    consensusText(tIdx);
  roundLabel.textContent = tOneBased === 1
    ? `Round 1 / ${{T}} (prior)`
    : `Round ${{tOneBased}} / ${{T}}`;
  slider.value = String(tOneBased);
  playhead.style.left = ((tOneBased - 0.5) / T) * 100 + '%';
  prevBtn.disabled = tOneBased <= 1;
  nextBtn.disabled = tOneBased >= T;
  renderPrivateSignals();
}}

slider.addEventListener('input', () => renderRound(Number(slider.value)));
showPrivateCb.addEventListener('change', renderPrivateSignals);

prevBtn.addEventListener('click', () => {{
  const t = Number(slider.value);
  if (t > 1) renderRound(t - 1);
}});

nextBtn.addEventListener('click', () => {{
  const t = Number(slider.value);
  if (t < T) renderRound(t + 1);
}});

timeline.addEventListener('click', (ev) => {{
  const rect = timeline.getBoundingClientRect();
  const frac = (ev.clientX - rect.left) / rect.width;
  const t = Math.min(T, Math.max(1, Math.round(frac * T)));
  renderRound(t);
}});

let timer = null;
playBtn.addEventListener('click', () => {{
  if (timer) {{
    clearInterval(timer);
    timer = null;
    playBtn.textContent = 'Play';
    return;
  }}
  playBtn.textContent = 'Pause';
  timer = setInterval(() => {{
    let t = Number(slider.value) + 1;
    if (t > T) t = 1;
    renderRound(t);
  }}, 500);
}});

DATA.episodes.forEach((ep, idx) => {{
  const opt = document.createElement('option');
  opt.value = String(idx);
  opt.textContent = `seed ${{ep.episode_seed}} (θ=${{ep.theta}})`;
  episodeSelect.appendChild(opt);
}});
episodeSelect.addEventListener('change', () => setEpisode(Number(episodeSelect.value)));
newSignalBtn.addEventListener('click', pickNewSignal);
newSignalBtn.disabled = DATA.episodes.length <= 1;
episodeSelect.disabled = DATA.episodes.length <= 1;

buildTimeline();
updateMetaLine();
renderRound(1);
renderPrivateSignals();
</script>
</body>
</html>
"""


def _eval_plot_section(eval_plot_filename: str | None) -> str:
    if not eval_plot_filename:
        return ""
    return f"""<details class="eval-panel" open>
  <summary>Error decay for selected signal (anchored at t=2)</summary>
  <p class="eval-caption">
    Empirical ε(t) for the <strong>currently selected</strong> private-signal draw
    (this episode rollout, not a test-set mean). GAT and HST curves share α and ε∞
    at <strong>t=2</strong>; fits use rounds t≥2.
  </p>
  <img id="eval-plot" class="eval-plot" src="{eval_plot_filename}" alt="Anchored t=2 learning-rate plot"/>
</details>"""


def _summary_plot_section(summary_plot_filename: str | None) -> str:
    if not summary_plot_filename:
        return ""
    return f"""<details class="eval-panel" open>
  <summary>Training evaluation summary (test-set mean, anchored at t=2)</summary>
  <p class="eval-caption">
    Mean empirical ε(t) over held-out <strong>test episodes</strong> from training
    evaluation (<code>metrics.json</code>). Same anchored t=2 GAT/HST comparison as
    in the paper-style plots.
  </p>
  <img id="summary-plot" class="eval-plot" src="{summary_plot_filename}" alt="Test-set mean anchored t=2 plot"/>
</details>"""


def save_interactive_episode_view(
    trace: EpisodeTrace | list[EpisodeTrace],
    graph: nx.Graph,
    output_path: str | Path,
    *,
    episode_seeds: list[int] | None = None,
    condition_key: str | None = None,
    summary_plot_filename: str | None = None,
) -> Path:
    """Write a self-contained interactive HTML viewer with scrubbable timeline."""
    from .learning_rate_plots import stage_per_episode_eval_plots

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    traces = trace if isinstance(trace, list) else [trace]
    seeds = episode_seeds if episode_seeds is not None else [0] * len(traces)
    if len(seeds) != len(traces):
        raise ValueError("episode_seeds length must match number of traces.")
    trace0 = traces[0]
    eval_plots: dict[int, str] = {}
    if condition_key is not None:
        eval_plots = stage_per_episode_eval_plots(
            traces=traces,
            episode_seeds=seeds,
            html_path=output,
            signal_quality=trace0.signal_quality,
            condition_key=condition_key,
        )
    payload = _viewer_payload(
        traces,
        graph,
        episode_seeds=seeds,
        eval_plots=eval_plots,
    )
    first_eval_plot = eval_plots.get(seeds[0]) if eval_plots else None
    meta = (
        f"n={trace0.num_nodes} · q={trace0.signal_quality:.2f} · "
        f"{trace0.topology}"
    )
    html = _HTML_TEMPLATE.format(
        title=meta,
        meta=meta,
        max_horizon=trace0.max_horizon,
        svg_margin=VIEWER_SVG_MARGIN,
        svg_size=VIEWER_SVG_MARGIN * 2,
        node_ring_scale=VIEWER_NODE_RING_SCALE,
        private_ring_scale=VIEWER_PRIVATE_RING_SCALE,
        color_correct=COLOR_CORRECT,
        color_wrong=COLOR_WRONG,
        large_graph_threshold=LARGE_GRAPH_NODE_THRESHOLD,
        eval_plot_section=_eval_plot_section(first_eval_plot),
        summary_plot_section=_summary_plot_section(summary_plot_filename),
        payload_json=json.dumps(payload, separators=(",", ":")),
    )
    output.write_text(html, encoding="utf-8")
    return output


def save_episode_animation(
    trace: EpisodeTrace,
    graph: nx.Graph,
    output_path: str | Path,
    *,
    layout_seed: int = 0,
    fps: int = 2,
    dpi: int = 120,
    frame_step: int = 1,
) -> Path:
    """Write a GIF with circular green/red nodes (optional legacy export)."""
    del layout_seed
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    positions = {
        node: np.asarray(coords, dtype=float)
        for node, coords in circular_layout(trace.num_nodes).items()
    }
    node_size = float(np.clip(12000.0 / trace.num_nodes, 80.0, 900.0))
    consensus = compute_unanimous_consensus(trace)

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    fig.subplots_adjust(top=0.9)

    timesteps = list(range(0, trace.max_horizon, max(1, frame_step)))
    if timesteps[-1] != trace.max_horizon - 1:
        timesteps.append(trace.max_horizon - 1)

    def _draw_frame(t: int) -> None:
        ax.clear()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)

        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            alpha=0.35,
            width=0.35,
            edge_color=COLOR_EDGE,
        )
        colors = [
            COLOR_CORRECT if trace.correct[t, node] else COLOR_WRONG
            for node in range(trace.num_nodes)
        ]
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            node_color=colors,
            node_size=node_size,
            linewidths=0,
        )

        row = consensus["per_round"][t]
        err = 1.0 - trace.correct[t].mean()
        round_tag = " (prior)" if t == 0 else ""
        title = (
            f"Round {t + 1}/{trace.max_horizon}{round_tag}  |  θ={trace.theta}  |  "
            f"error={err:.0%}"
        )
        if not row["after_gossip"]:
            title += "  |  private signal only"
        elif row["unanimous"]:
            tag = "correct" if row["unanimous_correct"] else "wrong"
            title += f"  |  unanimous ({tag})"
        elif consensus["first_unanimous_t"] is not None and t + 1 < consensus["first_unanimous_t"]:
            title += f"  |  unanimous at t={consensus['first_unanimous_t']}"
        ax.set_title(title, fontsize=11, pad=8)

    def _update(frame_idx: int) -> None:
        _draw_frame(timesteps[frame_idx])

    anim = FuncAnimation(fig, _update, frames=len(timesteps), interval=1000 // fps)
    writer = PillowWriter(fps=fps)
    anim.save(output, writer=writer, dpi=dpi)
    plt.close(fig)
    return output


def save_checkpoint(model: RecurrentGATAgent, path: str | Path, metadata: dict[str, Any]) -> Path:
    """Persist model weights and run metadata for animation reuse."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[RecurrentGATAgent, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    metadata = dict(payload["metadata"])
    model = RecurrentGATAgent(
        num_features_signal=1,
        hidden_dim=int(metadata["hidden_dim"]),
        num_heads=int(metadata["num_heads"]),
        dropout=float(metadata.get("dropout", 0.0)),
        communication_mode=str(metadata["communication_mode"]),
        communication_dim=metadata.get("communication_dim"),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata


__all__ = [
    "EpisodeTrace",
    "MAX_VIEWER_EDGES",
    "VIEWER_NODE_RING_SCALE",
    "VIEWER_PRIVATE_RING_SCALE",
    "VIEWER_SVG_MARGIN",
    "circular_layout",
    "compute_unanimous_consensus",
    "layout_positions",
    "load_checkpoint",
    "load_rollouts_cache",
    "node_radius_for_count",
    "pack_binary_bitmap",
    "pack_correct_bitmap",
    "pyg_to_networkx",
    "rollout_episode",
    "rollouts_cache_is_valid",
    "rollouts_cache_path",
    "sample_episode",
    "save_checkpoint",
    "save_episode_animation",
    "save_interactive_episode_view",
    "save_rollouts_cache",
    "viewer_edge_payload",
]
