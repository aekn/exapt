import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from exapt.graph import Graph, load_graph
from exapt.model import DistMultDecoder, RelationalGNN, RelationalLayer
from exapt.train import disease_bpr


class RelationalLayerTests(unittest.TestCase):
    def test_incoming_mean_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nodes.csv").write_text(
                "index,type,id\n"
                "10,drug,D10\n20,gene/protein,P20\n"
                "30,drug,D30\n40,disease,C40\n"
            )
            (root / "graph.csv").write_text(
                "x_index,x_type,relation,y_index,y_type\n"
                "10,drug,drug_protein,20,gene/protein\n"
                "30,drug,drug_protein,20,gene/protein\n"
            )
            graph = load_graph(root)

        layer = RelationalLayer(1, len(graph.relations))
        with torch.no_grad():
            layer.self_linear.weight.zero_()
            layer.self_linear.bias.zero_()
            for parameter in layer.rel_linear.parameters():
                parameter.fill_(1)

        features = torch.tensor(
            [[2.0], [0.0], [6.0], [9.0]], requires_grad=True
        )
        updated = layer(features, graph.adjacency)

        # mean of incoming features: (2 + 6) / 2 = 4
        torch.testing.assert_close(
            updated.flatten(), torch.tensor([0.0, 4.0, 0.0, 0.0])
        )

        updated.sum().backward()
        gradient = features.grad
        assert gradient is not None
        torch.testing.assert_close(
            gradient.flatten(), torch.tensor([0.5, 0.0, 0.5, 0.0])
        )

    def test_relation_specific_messages_are_added(self):
        # relation 0: node 0 -> node 2; relation 1: node 1 -> node 2
        indices = (torch.tensor([[2], [0]]), torch.tensor([[2], [1]]))
        adjacency = tuple(
            torch.sparse_coo_tensor(
                index, [1.0], (3, 3), check_invariants=True
            ).coalesce()
            for index in indices
        )
        layer = RelationalLayer(1, 2)
        with torch.no_grad():
            layer.self_linear.weight.fill_(0.5)
            layer.self_linear.bias.fill_(1)
            for linear, scale in zip(layer.rel_linear, (2.0, 3.0), strict=True):
                assert isinstance(linear, torch.nn.Linear)
                linear.weight.fill_(scale)

        features = torch.tensor([[4.0], [5.0], [0.0]], requires_grad=True)
        updated = layer(features, adjacency)
        # H' = 0.5 * H + 1 + 2 * (A_0 @ H) + 3 * (A_1 @ H)
        torch.testing.assert_close(updated.flatten(), torch.tensor([3., 3.5, 24.]))

        updated.sum().backward()
        gradient = features.grad
        assert gradient is not None
        torch.testing.assert_close(gradient.flatten(), torch.tensor([2.5, 3.5, 0.5]))


class DistMultTests(unittest.TestCase):
    def test_therapeutic_heads_and_batched_scores(self):
        decoder = DistMultDecoder(width=2)
        with torch.no_grad():
            decoder.weight.copy_(torch.tensor([
                [1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
            ]))

        embeddings = torch.tensor([[2., 1.], [3., 4.], [1., 2.]])
        drugs = torch.tensor([0])
        diseases = torch.tensor([1])

        # 2*w[0]*3 + 1*w[1]*4, independently for each relation
        scores = torch.stack([
            decoder(embeddings, drugs, diseases, relation)
            for relation in range(3)
        ])
        torch.testing.assert_close(scores.flatten(), torch.tensor([14., 34., 54.]))

        negatives = torch.tensor([[0, 2]])
        batched = decoder(embeddings, negatives, diseases[:, None], 0)
        self.assertEqual(tuple(batched.shape), (1, 2))
        torch.testing.assert_close(batched, torch.tensor([[14., 19.]]))

        scores.sum().backward()
        gradient = decoder.weight.grad
        assert gradient is not None
        torch.testing.assert_close(gradient, torch.tensor([[6., 4.]] * 3))

    def test_indication_head_is_the_first_decoder_row(self):
        model = RelationalGNN(3, 0, width=2)
        with torch.no_grad():
            model.indication.copy_(torch.tensor([0.5, 2.0]))

        embeddings = torch.tensor([
            [2., 1.],  # drug A
            [1., 0.],  # drug B
            [1., 3.],  # disease
        ])
        scores = model.score(embeddings, torch.tensor([0, 1]), torch.tensor([2, 2]))
        torch.testing.assert_close(scores, torch.tensor([7., 0.5]))
        torch.testing.assert_close(model.indication, model.decoder.weight[0])

    def test_all_three_heads_can_fit_fixed_pairs(self):
        torch.manual_seed(163)
        model = RelationalGNN(5, 0, width=8)
        graph = Graph({i: i for i in range(5)}, (), ())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        # (relation, positive drug, negative drug, disease)
        examples = ((0, 0, 1, 3), (1, 1, 0, 3), (2, 0, 1, 4))
        for _ in range(100):
            optimizer.zero_grad()
            embeddings = model(graph)
            losses = []
            for relation, positive, negative, disease in examples:
                disease_ids = torch.tensor([disease])
                pos = model.score(
                    embeddings, torch.tensor([positive]), disease_ids, relation
                )
                neg = model.score(
                    embeddings, torch.tensor([[negative]]),
                    disease_ids[:, None], relation,
                )
                losses.append(disease_bpr(pos, neg, torch.ones(1)))

            loss = torch.stack(losses).sum()
            loss.backward()
            optimizer.step()

        # assess updated parameters, not loss before last step
        with torch.no_grad():
            embeddings = model(graph)
            margins = [
                model.score(
                    embeddings, torch.tensor([positive]),
                    torch.tensor([disease]), relation,
                ) - model.score(
                    embeddings, torch.tensor([negative]),
                    torch.tensor([disease]), relation,
                )
                for relation, positive, negative, disease in examples
            ]
            final_loss = torch.stack([F.softplus(-margin) for margin in margins]).sum()
        self.assertLess(final_loss.item(), 0.01)
