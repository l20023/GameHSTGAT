"""Graph generation utilities for PyTorch Geometric experiments."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
import torch
from torch_geometric.data import Data


class GraphGenerator:
    """Generate proposal graph topologies as PyG ``Data`` objects."""

    def __init__(
        self,
        *,
        default_k: int = 2,
        default_ws_probs: list[float] | None = None,
        default_seed: int | None = None,
        ws_tries: int = 100,
    ) -> None:
        self.default_k = default_k
        self.default_ws_probs = [0.0, 0.1] if default_ws_probs is None else default_ws_probs
        self.default_seed = default_seed
        self.ws_tries = ws_tries

    @staticmethod
    def _validate_num_nodes(num_nodes: int) -> None:
        if num_nodes < 3:
            raise ValueError("num_nodes must be >= 3.")

    @staticmethod
    def _validate_ws_params(num_nodes: int, k: int, p: float) -> None:
        if k <= 0 or k >= num_nodes:
            raise ValueError("k must satisfy 0 < k < num_nodes.")
        if k % 2 != 0:
            raise ValueError("k must be even for Watts-Strogatz graphs.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")

    @staticmethod
    def _nx_to_pyg_data(
        graph: nx.Graph,
        *,
        topology: str,
        metadata: dict[str, Any] | None = None,
    ) -> Data:
        edges = sorted((min(u, v), max(u, v)) for u, v in graph.edges())

        directed_edges = []
        for u, v in edges:
            directed_edges.append((u, v))
            directed_edges.append((v, u))

        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        data = Data(edge_index=edge_index, num_nodes=graph.number_of_nodes())
        data.topology = topology

        if metadata:
            for key, value in metadata.items():
                setattr(data, key, value)

        return data

    def generate_complete(self, num_nodes: int) -> Data:
        """Generate a complete graph and return it as PyG Data."""
        self._validate_num_nodes(num_nodes)
        graph = nx.complete_graph(num_nodes)
        return self._nx_to_pyg_data(
            graph,
            topology="complete",
            metadata={"num_nodes_value": num_nodes},
        )

    def generate_watts_strogatz(
        self,
        num_nodes: int,
        *,
        k: int | None = None,
        p: float = 0.0,
        seed: int | None = None,
    ) -> Data:
        """Generate a connected Watts-Strogatz graph and return PyG Data."""
        k_value = self.default_k if k is None else k
        seed_value = self.default_seed if seed is None else seed
        self._validate_num_nodes(num_nodes)
        self._validate_ws_params(num_nodes, k_value, p)

        graph = nx.connected_watts_strogatz_graph(
            n=num_nodes,
            k=k_value,
            p=p,
            tries=self.ws_tries,
            seed=seed_value,
        )
        return self._nx_to_pyg_data(
            graph,
            topology="watts_strogatz",
            metadata={
                "num_nodes_value": num_nodes,
                "k": k_value,
                "p": float(p),
                "seed": seed_value,
            },
        )

    def generate_topology(
        self,
        topology: str,
        num_nodes: int,
        *,
        k: int | None = None,
        p: float | None = None,
        seed: int | None = None,
    ) -> Data:
        """Generate a topology by name."""
        normalized = topology.strip().lower()
        if normalized == "complete":
            return self.generate_complete(num_nodes)

        if normalized in {"watts_strogatz", "ws"}:
            ws_p = 0.0 if p is None else p
            return self.generate_watts_strogatz(num_nodes=num_nodes, k=k, p=ws_p, seed=seed)

        raise ValueError(
            f"Unsupported topology '{topology}'. Use 'complete', 'watts_strogatz', or 'ws'."
        )

    def generate_experiment_graphs(
        self,
        num_nodes_list: list[int],
        *,
        ws_probs: list[float] | None = None,
        k: int | None = None,
        seed: int | None = None,
    ) -> dict[int, dict[str, Data]]:
        """Generate proposal graphs for each node size."""
        probabilities = self.default_ws_probs if ws_probs is None else ws_probs
        output: dict[int, dict[str, Data]] = {}

        for num_nodes in num_nodes_list:
            output[num_nodes] = {"complete": self.generate_complete(num_nodes)}
            for p_value in probabilities:
                key = f"ws_p_{p_value:.1f}"
                output[num_nodes][key] = self.generate_watts_strogatz(
                    num_nodes=num_nodes,
                    k=k,
                    p=p_value,
                    seed=seed,
                )

        return output

    @staticmethod
    def _format_probability(p: float) -> str:
        return f"{p:.1f}".replace(".", "p")

    @staticmethod
    def save_graph(data: Data, path: str | Path) -> None:
        graph_path = Path(path)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".pt.tmp", dir=graph_path.parent)
        os.close(fd)
        try:
            torch.save(data, tmp_path)
            os.replace(tmp_path, graph_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @staticmethod
    def load_graph(path: str | Path) -> Data:
        return torch.load(Path(path), map_location="cpu", weights_only=False)

    def generate_and_store_experiment_graphs(
        self,
        storage_dir: str | Path,
        num_nodes_list: list[int],
        *,
        ws_probs: list[float] | None = None,
        k: int | None = None,
        seed: int | None = None,
    ) -> dict[int, dict[str, Data]]:
        """
        Generate proposal graphs and cache them to disk.

        Existing files are loaded instead of regenerated.
        """
        base_dir = Path(storage_dir)
        probabilities = self.default_ws_probs if ws_probs is None else ws_probs
        output: dict[int, dict[str, Data]] = {}

        for num_nodes in num_nodes_list:
            node_dir = base_dir / f"n_{num_nodes}"
            output[num_nodes] = {}

            complete_path = node_dir / "complete.pt"
            if complete_path.exists():
                complete_data = self.load_graph(complete_path)
            else:
                complete_data = self.generate_complete(num_nodes)
                self.save_graph(complete_data, complete_path)
            output[num_nodes]["complete"] = complete_data

            for p_value in probabilities:
                p_key = self._format_probability(p_value)
                key = f"ws_p_{p_value:.1f}"
                if seed is not None:
                    key = f"{key}_seed_{seed}"

                file_name = f"ws_p_{p_key}.pt"
                if seed is not None:
                    file_name = f"ws_p_{p_key}_seed_{seed}.pt"
                ws_path = node_dir / file_name

                if ws_path.exists():
                    ws_data = self.load_graph(ws_path)
                else:
                    ws_data = self.generate_watts_strogatz(
                        num_nodes=num_nodes,
                        k=k,
                        p=p_value,
                        seed=seed,
                    )
                    self.save_graph(ws_data, ws_path)
                output[num_nodes][key] = ws_data

        return output


__all__ = ["GraphGenerator"]
