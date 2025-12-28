"""Tests for TSP neural network components."""

import pytest

# Optional torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

if HAS_TORCH and HAS_TORCH_GEOMETRIC:
    from tsp_nls.net import Net, EmbNet, ParNet
    from tsp_nls.utils import gen_pyg_data
else:
    Net = EmbNet = ParNet = None
    gen_pyg_data = None


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCH_GEOMETRIC, reason="PyTorch or PyTorch Geometric not available")
class TestTSPNet:
    """Test suite for TSP neural networks."""

    def test_emb_net_initialization(self):
        """Test EmbNet initialization."""
        depth = 6
        feats = 2
        units = 16

        net = EmbNet(depth=depth, feats=feats, units=units)

        assert net.depth == depth
        assert net.feats == feats
        assert net.units == units

    def test_emb_net_forward(self, small_tsp_instance):
        """Test EmbNet forward pass."""
        coordinates = small_tsp_instance['coordinates']
        k_sparse = 5

        pyg_data, _ = gen_pyg_data(coordinates, k_sparse)

        net = EmbNet()
        edge_embeddings = net(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)

        # Check output shape
        n_edges = pyg_data.edge_index.shape[1]
        assert edge_embeddings.shape == (n_edges, net.units)

        # Check finite values
        assert torch.all(torch.isfinite(edge_embeddings))

    def test_par_net_initialization(self):
        """Test ParNet initialization."""
        depth = 3
        units = 32
        preds = 1

        net = ParNet(depth=depth, units=units, preds=preds)

        assert net.depth == depth  # depth is number of hidden layers
        assert net.units == units
        assert net.preds == preds

    def test_par_net_forward(self):
        """Test ParNet forward pass."""
        batch_size = 4
        input_dim = 32

        net = ParNet()
        x = torch.randn(batch_size, input_dim)
        output = net(x)

        # Check output shape (sigmoid output, squeezed)
        assert output.shape == (batch_size,)
        assert torch.all((output >= 0) & (output <= 1))

    def test_main_net_initialization(self):
        """Test main Net initialization."""
        net = Net(gfn=False)
        assert not net.gfn
        assert net.Z_net is None

        net_gfn = Net(gfn=True)
        assert net_gfn.gfn
        assert net_gfn.Z_net is not None

    def test_main_net_forward(self, small_tsp_instance):
        """Test main Net forward pass."""
        coordinates = small_tsp_instance['coordinates']
        k_sparse = 5

        pyg_data, _ = gen_pyg_data(coordinates, k_sparse)

        # Test without GFlowNet
        net = Net(gfn=False)
        heu_vec = net(pyg_data)

        n_nodes = pyg_data.x.shape[0]
        expected_size = n_nodes * k_sparse  # Number of edges with k_sparse
        assert heu_vec.shape == (expected_size,)

        # Test with GFlowNet
        net_gfn = Net(gfn=True)
        heu_vec_gfn, logZ = net_gfn(pyg_data, return_logZ=True)

        assert heu_vec_gfn.shape == (expected_size,)
        assert len(logZ) == 1  # Single Z for non-guided
        assert torch.isfinite(logZ[0])

    def test_net_reshape(self, small_tsp_instance):
        """Test Net reshape method."""
        coordinates = small_tsp_instance['coordinates']
        n_nodes = small_tsp_instance['n_nodes']
        k_sparse = 5

        pyg_data, _ = gen_pyg_data(coordinates, k_sparse)

        net = Net()
        heu_vec = net(pyg_data)
        heu_mat = net.reshape(pyg_data, heu_vec)

        # Check reshaped matrix
        assert heu_mat.shape == (n_nodes, n_nodes)
        assert torch.all(torch.isfinite(heu_mat))

    def test_net_guided_exploration(self, small_tsp_instance):
        """Test Net with guided exploration (two Z values)."""
        coordinates = small_tsp_instance['coordinates']
        k_sparse = 5

        pyg_data, _ = gen_pyg_data(coordinates, k_sparse)

        net = Net(gfn=True)
        heu_vec, logZs = net(pyg_data, return_logZ=True)

        # Should return Z values for guided exploration
        assert isinstance(logZs, torch.Tensor)  # Single Z tensor
        assert logZs.numel() == 1  # Single Z value
        assert torch.isfinite(logZs).all()

    @pytest.mark.parametrize("depth", [3, 6, 12])
    def test_emb_net_depths(self, depth):
        """Test EmbNet with different depths."""
        net = EmbNet(depth=depth)

        # Create minimal test data
        x = torch.randn(5, 2)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        edge_attr = torch.randn(5, 1)

        output = net(x, edge_index, edge_attr)

        # Should work regardless of depth
        assert output.shape[0] == edge_index.shape[1]
        assert output.shape[1] == net.units
        assert torch.all(torch.isfinite(output))

    def test_net_batch_processing(self, small_tsp_instance):
        """Test Net with batch processing."""
        coordinates = small_tsp_instance['coordinates']
        k_sparse = 5

        # Create batch of identical instances
        batch_coords = torch.stack([coordinates] * 3)  # Batch size 3
        batch_data = []
        for i in range(3):
            pyg_data, _ = gen_pyg_data(batch_coords[i], k_sparse)
            batch_data.append(pyg_data)

        net = Net()

        # Process each instance
        results = []
        for pyg_data in batch_data:
            result = net(pyg_data)
            results.append(result)

        # All results should have same shape
        expected_edges = batch_data[0].edge_index.shape[1]
        for result in results:
            assert result.shape[0] == expected_edges

    def test_net_different_configurations(self):
        """Test Net with different configurations."""
        # Test different GFN settings
        net_gfn = Net(gfn=True)
        assert net_gfn.gfn is True
        assert net_gfn.Z_net is not None

        net_no_gfn = Net(gfn=False)
        assert net_no_gfn.gfn is False
        assert net_no_gfn.Z_net is None

    def test_net_device_consistency(self, device):
        """Test that Net respects device placement."""
        if not torch.cuda.is_available() and device.type == 'cuda':
            pytest.skip("CUDA not available")

        # Create simple test data
        x = torch.randn(5, 2, device=device)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], device=device)
        edge_attr = torch.randn(5, 1, device=device)

        # Manually create data object
        from torch_geometric.data import Data
        pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        net = Net().to(device)

        # Forward pass
        output = net(pyg_data)

        # Should be on correct device
        assert output.device == device

    def test_net_device_consistency(self, device):
        """Test that network respects device placement."""
        if not torch.cuda.is_available() and device.type == 'cuda':
            pytest.skip("CUDA not available")

        net = Net().to(device)

        # Create test data on same device
        x = torch.randn(5, 2, device=device)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], device=device)
        edge_attr = torch.randn(5, 1, device=device)

        # Manually set device for EmbNet
        net.emb_net = net.emb_net.to(device)

        output = net.emb_net(x, edge_index, edge_attr)

        assert output.device == device
