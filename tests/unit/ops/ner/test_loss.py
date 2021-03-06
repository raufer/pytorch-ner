import torch
import unittest

from src.ops.ner.loss import active_loss


class TestOpsNERLoss(unittest.TestCase):

    def test_active_parts_of_loss(self):

        loss_fct = torch.nn.CrossEntropyLoss()

        target = torch.tensor([
            [-1, 0, 2, 2, -1, -1, -1, -1, -1, -1],
            [-1, 0, 0, 0, 0, 1, 1, -1, -1, -1]
        ])
        logits = torch.tensor([
            [[0.9672, 0.3671, 0.5196],
             [0.7375, 0.4412, 0.6449],
             [0.5044, 0.9435, 0.1143],
             [0.9318, 0.8636, 0.8863],
             [0.2648, 0.4413, 0.9613],
             [0.7696, 0.5530, 0.9382],
             [0.3856, 0.3141, 0.1020],
             [0.0399, 0.5205, 0.3871],
             [0.3224, 0.1528, 0.6944],
             [0.1538, 0.6460, 0.5724]],

            [[0.2613, 0.2959, 0.1332],
             [0.0477, 0.4868, 0.9213],
             [0.3568, 0.2871, 0.3812],
             [0.2602, 0.8076, 0.6706],
             [0.3575, 0.3397, 0.1995],
             [0.5706, 0.7086, 0.0107],
             [0.9112, 0.2669, 0.1066],
             [0.2189, 0.0382, 0.3898],
             [0.5111, 0.3911, 0.6043],
             [0.3021, 0.5809, 0.4534]]
        ])
        loss_1 = active_loss(loss_fct, target, logits)

        target = torch.tensor([
            [0, 0, 2, 2, -1, -1, -1, -1, -1, -1],
            [-1, 0, 0, 0, 0, 1, 1, -1, -1, -1]
        ])
        logits = torch.tensor([
            [[0.001, 0.3671, 0.5196],
             [0.7375, 0.4412, 0.6449],
             [0.5044, 0.9435, 0.1143],
             [0.9318, 0.8636, 0.8863],
             [0.2648, 0.4413, 0.9613],
             [0.7696, 0.5530, 0.9382],
             [0.3856, 0.3141, 0.1020],
             [0.0399, 0.5205, 0.3871],
             [0.3224, 0.1528, 0.6944],
             [0.1538, 0.6460, 0.5724]],

            [[0.2613, 0.2959, 0.1332],
             [0.0477, 0.4868, 0.9213],
             [0.3568, 0.2871, 0.3812],
             [0.2602, 0.8076, 0.6706],
             [0.3575, 0.3397, 0.1995],
             [0.5706, 0.7086, 0.0107],
             [0.9112, 0.2669, 0.1066],
             [0.2189, 0.0382, 0.3898],
             [0.5111, 0.3911, 0.6043],
             [0.3021, 0.5809, 0.4534]]
        ])
        loss_2 = active_loss(loss_fct, target, logits)

        target = torch.tensor([
            [0, 0, 2, 2, -1, -1, -1, -1, -1, 0],
            [-1, 0, 0, 0, 0, 1, 1, -1, -1, -1]
        ])
        logits = torch.tensor([
            [[0.001, 0.3671, 0.5196],
             [0.7375, 0.4412, 0.6449],
             [0.5044, 0.9435, 0.1143],
             [0.9318, 0.8636, 0.8863],
             [0.2648, 0.4413, 0.9613],
             [0.7696, 0.5530, 0.9382],
             [0.3856, 0.3141, 0.1020],
             [0.0399, 0.5205, 0.3871],
             [0.3224, 0.1528, 0.6944],
             [0.1538, 0.6460, 0.5724]],

            [[0.2613, 0.2959, 0.1332],
             [0.0477, 0.4868, 0.9213],
             [0.3568, 0.2871, 0.3812],
             [0.2602, 0.8076, 0.6706],
             [0.3575, 0.3397, 0.1995],
             [0.5706, 0.7086, 0.0107],
             [0.9112, 0.2669, 0.1066],
             [0.2189, 0.0382, 0.3898],
             [0.5111, 0.3911, 0.6043],
             [0.3021, 0.5809, 0.4534]]
        ])
        loss_3 = active_loss(loss_fct, target, logits)

        self.assertTrue(loss_1 < loss_2)
        self.assertTrue(loss_2 < loss_3)




