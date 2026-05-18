import networkx as nx
import pytest
import torch
from torch_geometric.data import Data

from src.graph_generator import GraphGenerator


def _to_undirected_nx_graph(data: Data) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    edge_pairs = data.edge_index.t().tolist()
    graph.add_edges_from((u, v) for u, v in edge_pairs if u != v)
    return graph


def _unique_undirected_edge_count(data: Data) -> int:
    edge_pairs = data.edge_index.t().tolist()
    unique_edges = {(min(u, v), max(u, v)) for u, v in edge_pairs if u != v}
    return len(unique_edges)


@pytest.mark.parametrize("num_nodes", [10, 50, 100])
def test_generators_return_pyg_data_and_correct_node_count(num_nodes: int) -> None:
    generator = GraphGenerator()
    complete = generator.generate_complete(num_nodes)
    ws_local = generator.generate_watts_strogatz(num_nodes, k=2, p=0.0, seed=42)
    ws_small_world = generator.generate_watts_strogatz(num_nodes, k=2, p=0.1, seed=42)

    for graph_data in (complete, ws_local, ws_small_world):
        assert isinstance(graph_data, Data)
        assert graph_data.num_nodes == num_nodes


@pytest.mark.parametrize("num_nodes", [10, 50, 100])
def test_complete_graph_has_expected_undirected_edge_count(num_nodes: int) -> None:
    generator = GraphGenerator()
    data = generator.generate_complete(num_nodes)
    expected_edges = num_nodes * (num_nodes - 1) // 2
    assert _unique_undirected_edge_count(data) == expected_edges


@pytest.mark.parametrize("p_value", [0.0, 0.1])
def test_watts_strogatz_graphs_are_connected(p_value: float) -> None:
    generator = GraphGenerator()
    data = generator.generate_watts_strogatz(num_nodes=100, k=2, p=p_value, seed=123)
    graph = _to_undirected_nx_graph(data)
    assert nx.is_connected(graph)


def test_watts_strogatz_generation_is_deterministic_for_fixed_seed() -> None:
    generator = GraphGenerator()
    graph_1 = generator.generate_watts_strogatz(num_nodes=50, k=2, p=0.1, seed=77)
    graph_2 = generator.generate_watts_strogatz(num_nodes=50, k=2, p=0.1, seed=77)
    assert torch.equal(graph_1.edge_index, graph_2.edge_index)


def test_generate_topology_dispatches() -> None:
    generator = GraphGenerator()
    complete = generator.generate_topology("complete", num_nodes=10)
    ws = generator.generate_topology("ws", num_nodes=10, k=2, p=0.1, seed=11)
    assert complete.topology == "complete"
    assert ws.topology == "watts_strogatz"


def test_class_based_generator_api() -> None:
    generator = GraphGenerator(default_k=2, default_seed=19)
    complete = generator.generate_complete(num_nodes=10)
    ws = generator.generate_watts_strogatz(num_nodes=10, p=0.1)
    assert complete.topology == "complete"
    assert ws.topology == "watts_strogatz"
    assert ws.seed == 19


def test_generate_and_store_experiment_graphs_caches_existing_files(tmp_path) -> None:
    generator = GraphGenerator()
    output_first = generator.generate_and_store_experiment_graphs(
        storage_dir=tmp_path,
        num_nodes_list=[10],
        ws_probs=[0.0, 0.1],
        seed=3,
    )

    expected_file_count = 1 + 2  # complete + two WS probabilities
    all_files = list(tmp_path.rglob("*.pt"))
    assert len(all_files) == expected_file_count

    file_mtimes_before = {path: path.stat().st_mtime for path in all_files}
    output_second = generator.generate_and_store_experiment_graphs(
        storage_dir=tmp_path,
        num_nodes_list=[10],
        ws_probs=[0.0, 0.1],
        seed=3,
    )
    file_mtimes_after = {path: path.stat().st_mtime for path in all_files}
    assert file_mtimes_before == file_mtimes_after

    assert torch.equal(
        output_first[10]["ws_p_0.1_seed_3"].edge_index,
        output_second[10]["ws_p_0.1_seed_3"].edge_index,
    )


def test_invalid_parameters_raise_value_error() -> None:
    generator = GraphGenerator()
    with pytest.raises(ValueError):
        generator.generate_watts_strogatz(num_nodes=2, k=2, p=0.1)
    with pytest.raises(ValueError):
        generator.generate_watts_strogatz(num_nodes=10, k=3, p=0.1)
    with pytest.raises(ValueError):
        generator.generate_watts_strogatz(num_nodes=10, k=2, p=1.5)
