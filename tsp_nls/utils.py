import os
import pickle

import torch

# Optional torch_geometric import
try:
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    Data = None
    HAS_TORCH_GEOMETRIC = False


@torch.no_grad()
def gen_distance_matrix(tsp_coordinates):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        distance_matrix: torch tensor [n_nodes, n_nodes] for EUC distances
    '''
    n_nodes = len(tsp_coordinates)

    # Vectorized Euclidean distance computation
    # More memory efficient than norm for large matrices
    diff = tsp_coordinates.unsqueeze(0) - tsp_coordinates.unsqueeze(1)  # [n_nodes, n_nodes, 2]
    distances = torch.sqrt((diff ** 2).sum(dim=-1))  # [n_nodes, n_nodes]

    # Set diagonal to large value to prevent self-loops
    distances.fill_diagonal_(1e9)

    return distances


def gen_pyg_data(tsp_coordinates, k_sparse, start_node=None):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
        k_sparse: int, number of edges to keep for each node
        start_node: int, index of the start node, if None, use random start node
    Returns:
        pyg_data: pyg Data instance
        distances: distance matrix
    '''
    if not HAS_TORCH_GEOMETRIC:
        raise ImportError("torch_geometric not available")
    n_nodes = len(tsp_coordinates)
    distances = gen_distance_matrix(tsp_coordinates)
    topk_values, topk_indices = torch.topk(distances, 
                                           k=k_sparse, 
                                           dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                repeats=k_sparse),
        torch.flatten(topk_indices)
        ])
    edge_attr = topk_values.reshape(-1, 1)

    if start_node is None:
        node_feature = tsp_coordinates
    else:
        node_feature = torch.zeros((n_nodes,1), device=tsp_coordinates.device, dtype=tsp_coordinates.dtype)
        node_feature[start_node, 0] = 1.0
    pyg_data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances


def gen_pyg_data_tsplib(tsp_coordinates, k_sparse, start_node=None):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
        k_sparse: int, number of edges to keep for each node
        start_node: int, index of the start node, if None, use random start node
    Returns:
        pyg_data: pyg Data instance
        distances: distance matrix
    '''
    if not HAS_TORCH_GEOMETRIC:
        raise ImportError("torch_geometric not available")
    n_nodes = len(tsp_coordinates)
    distances = gen_distance_matrix(tsp_coordinates)
    topk_values, topk_indices = torch.topk(distances, k=k_sparse, dim=1, largest=False)
    edge_index = torch.stack(
        [
            torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device), repeats=k_sparse),
            torch.flatten(topk_indices)
        ]
    )
    edge_attr = topk_values.reshape(-1, 1)

    if start_node is None:
        node_feature = tsp_coordinates
    else:
        node_feature = torch.zeros((n_nodes,1), device=tsp_coordinates.device, dtype=tsp_coordinates.dtype)
        node_feature[start_node, 0] = 1.0
    pyg_data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances


def load_val_dataset(n_nodes, k_sparse, device, start_node=None):
    if not os.path.isfile(f"../data/tsp/valDataset-{n_nodes}.pt"):
        val_tensor = torch.rand((50, n_nodes, 2))
        torch.save(val_tensor, f"../data/tsp/valDataset-{n_nodes}.pt")
    else:
        val_tensor = torch.load(f"../data/tsp/valDataset-{n_nodes}.pt")

    val_list = []
    for instance in val_tensor:
        instance = instance.to(device)
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse, start_node=start_node)
        val_list.append((data, distances))
    return val_list


def load_test_dataset(n_nodes, k_sparse, device, start_node=None, filename=None):
    filename = filename or f"../data/tsp/testDataset-{n_nodes}.pt"
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"File {filename} not found, please download the test dataset from the original repository."
        )
    test_tensor = torch.load(filename)

    test_list = []
    for instance in test_tensor:
        instance = instance.to(device)
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse, start_node=start_node)
        test_list.append((data, distances))
    return test_list


def load_tsplib_dataset(n_nodes, k_sparse_factor, device, start_node=None, filename=None):
    scale_map = {200: ("100", "299"), 500: ("300", "699"), 1000: ("700", "1499")}
    filename = filename or f"../data/tsp/tsplib/tsplib_{'_'.join(scale_map[n_nodes])}.pkl"
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"File {filename} not found, please download the test dataset from the original repository."
        )

    with open(filename, "rb") as f:
        tsplib_list = pickle.load(f)

    test_list = []
    scale_list = []
    name_list = []
    for instance, scale, name in tsplib_list:
        instance = instance.to(device)
        data, distances = gen_pyg_data_tsplib(instance, k_sparse=instance.shape[0] // k_sparse_factor, start_node=start_node)
        test_list.append((data, distances))
        scale_list.append(scale)
        name_list.append(name)
    return test_list, scale_list, name_list


if __name__ == "__main__":
    pass
