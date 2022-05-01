import torch

from models import metric


class TestBuildSparseAdjacency:
    """
    Class of tests for ``build_sparse_adjacency`` function in the ``metric`` module.
    """

    def test_build_sparse_adjacency(self):
        """
        Test for `metric.build_sparse_adjacency` when `symmetryc=False` and `remove_reference=False`.
        """
        # Build a dense sparse matrix of size (n_nodes,n_nodes) with nnz non-zero elements.
        n_nodes = 10
        nnz = n_nodes

        # Non-zero entries
        idx1 = []
        idx2 = []
        weights1 = []
        weights2 = []
        weights_diag = torch.rand(n_nodes)

        matrix = torch.diag(1 / (1 + weights_diag ** 2))
        for _ in range(nnz):
            i = torch.randint(0, n_nodes, (1,))
            j = torch.randint(0, n_nodes, (1,))
            if [i, j] in idx1 or [j, i] in idx1 or i == j:
                # Make sure that an entry will not have multiple values
                continue
            w1 = torch.rand(1)
            w2 = torch.rand(1)
            matrix[i, j] = 1 / (1 + w1 ** 2)
            matrix[j, i] = 1 / (1 + w2 ** 2)

            idx1.append([i, j])
            weights1.append(w1)

            idx2.append([j, i])
            weights2.append(w2)

        idx1 = torch.tensor(idx1).t()
        idx2 = torch.tensor(idx2).t()
        weights1 = torch.tensor(weights1)
        weights2 = torch.tensor(weights2)

        sum_row = matrix.sum(dim=1)
        answer = matrix / sum_row.view(n_nodes, 1)

        idx_identity = torch.cat(
            (
                torch.arange(n_nodes).view(1, n_nodes),
                torch.arange(n_nodes).view(1, n_nodes),
            ),
            dim=0,
        )

        idx = torch.cat((idx_identity, idx1, idx2), dim=1)
        weights = torch.cat((weights_diag, weights1, weights2))
        x = metric.build_sparse_adjacency(idx, weights, device="cpu", symmetric=False, remove_reference=False)
        # assert (torch.abs(x.to_dense() - answer) < 1e-6).sum() == answer.nelement()
        assert (torch.abs(x.to_dense() - answer) < 1e-6).sum() == answer.nelement()

    def test_build_sparse_adjacency_symmetric(self):
        """
        Test for `metric.build_sparse_adjacency` when `symmetric=True`.
        """
        # Build a dense sparse matrix of size (n_nodes,n_nodes) with nnz non-zero elements.
        n_nodes = 10
        nnz = n_nodes

        # Non-zero entries
        idx1 = []
        idx2 = []
        weights1 = []
        weights2 = []
        weights_diag = torch.rand(n_nodes)

        matrix = torch.diag(1 / (1 + weights_diag ** 2))
        for _ in range(nnz):
            i = torch.randint(0, n_nodes, (1,))
            j = torch.randint(0, n_nodes, (1,))
            if [i, j] in idx1 or [j, i] in idx1 or i == j:
                # Make sure that an entry will not have multiple values
                continue
            w1 = torch.rand(1)
            w2 = torch.rand(1)
            matrix[i, j] = 1 / (1 + w1 ** 2)
            matrix[j, i] = 1 / (1 + w2 ** 2)

            idx1.append([i, j])
            weights1.append(w1)

            idx2.append([j, i])
            weights2.append(w2)

        idx1 = torch.tensor(idx1).t()
        idx2 = torch.tensor(idx2).t()
        weights1 = torch.tensor(weights1)
        weights2 = torch.tensor(weights2)

        sum_row = matrix.sum(dim=1)
        matrix = matrix / sum_row.view(n_nodes, 1)
        answer = 0.5 * (matrix + matrix.t())  # Symmetrize matrix (this is what will be tested against)

        idx_identity = torch.cat(
            (
                torch.arange(n_nodes).view(1, n_nodes),
                torch.arange(n_nodes).view(1, n_nodes),
            ),
            dim=0,
        )

        idx = torch.cat((idx_identity, idx1, idx2), dim=1)
        weights = torch.cat((weights_diag, weights1, weights2))
        x = metric.build_sparse_adjacency(idx, weights, device="cpu", symmetric=True, remove_reference=False)
        assert (torch.abs(x.to_dense() - answer) < 1e-6).sum() == answer.nelement()

    def test_build_sparse_adjacency_remove_reference(self):
        """
        Test for `metric.build_sparse_adjacency` when `remove_reference=True`.
        """
        # Build a dense sparse matrix of size (n_nodes,n_nodes) with nnz non-zero elements.
        n_nodes = 10
        nnz = n_nodes

        # Non-zero entries
        idx1 = []
        idx2 = []
        weights1 = []
        weights2 = []
        weights_diag = torch.rand(n_nodes)

        # Construct matrix with nothing along the diagonal
        matrix = torch.zeros(n_nodes, n_nodes)
        for i in range(n_nodes):
            j = torch.randint(0, n_nodes, (1,))
            while [i, j] in idx1 or [j, i] in idx1 or i == j:
                # Make sure that an entry will not have multiple values
                j = torch.randint(0, n_nodes, (1,))
            w1 = torch.rand(1)
            w2 = torch.rand(1)
            matrix[i, j] = 1 / (1 + w1 ** 2)
            matrix[j, i] = 1 / (1 + w2 ** 2)

            idx1.append([i, j])
            weights1.append(w1)

            idx2.append([j, i])
            weights2.append(w2)

        idx1 = torch.tensor(idx1).t()
        idx2 = torch.tensor(idx2).t()
        weights1 = torch.tensor(weights1)
        weights2 = torch.tensor(weights2)

        sum_row = matrix.sum(dim=1)
        answer = matrix / sum_row.view(n_nodes, 1)

        idx_identity = torch.cat(
            (
                torch.arange(n_nodes).view(1, n_nodes),
                torch.arange(n_nodes).view(1, n_nodes),
            ),
            dim=0,
        )

        idx = torch.cat((idx_identity, idx1, idx2), dim=1)
        weights = torch.cat((weights_diag, weights1, weights2))
        x = metric.build_sparse_adjacency(idx, weights, device="cpu", symmetric=False, remove_reference=True)
        assert (torch.abs(x.to_dense() - answer) < 1e-6).sum() == answer.nelement()
