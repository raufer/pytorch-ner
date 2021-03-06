import torch
import unittest

from numpy import array

from src.ops.ner.entity_resolution import take_single_entity_for_each_class


class TestOpsNEREntityResolution(unittest.TestCase):

    def test_take_single_entity_for_each_class_basic(self):
        predictions = []
        self.assertListEqual(take_single_entity_for_each_class(predictions), [])

        predictions = [
            ('The', 0, array([0.93692135, 0.01944068, 0.043638])),
            ('authoritites', 0, array([0.00486542, 0.9518131, 0.04332148]))
        ]
        expected = [(a, b) for a, b, c in predictions]
        result = take_single_entity_for_each_class(predictions)
        self.assertListEqual(result, expected)

        predictions = [
            ('The', 0, array([0.93692135, 0.01944068, 0.043638])),
            ('authoritites', 0, array([0.00486542, 0.9518131, 0.04332148])),
            ('specified', 2, array([0.04215204, 0.00930096, 0.94854707])),
            ('in', 2, array([0.05586903, 0.0064729, 0.93765807]))
        ]
        expected = [(a, b) for a, b, c in predictions]
        result = take_single_entity_for_each_class(predictions)
        self.assertListEqual(result, expected)

        predictions = [
            ('The', 0, array([0.93692135, 0.01944068, 0.043638])),
            ('authoritites', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('specified', 2, array([0.04215204, 0.00930096, 0.94854707])),
            ('in', 2, array([0.05586903, 0.0064729, 0.93765807])),
            ('paragraph', 2, array([0.06311218, 0.00647875, 0.93040907])),
            ('2', 2, array([0.04869084, 0.00783127, 0.94347787])),
            ('must', 0, array([0.9694128, 0.00716435, 0.02342291])),
            ('write', 0, array([0.97327912, 0.00752228, 0.01919858])),
            ('a', 0, array([0.9745782, 0.00830096, 0.01712076])),
            ('letter', 0, array([0.97538519, 0.00681078, 0.01780402])),
            ('with', 0, array([0.9730261, 0.00481166, 0.02216221])),
            ('the', 0, array([0.97501367, 0.00495419, 0.0200321])),
            ('notice', 0, array([0.97576523, 0.00637131, 0.01786346])),
            ('period', 0, array([0.97400716, 0.0075533, 0.01843956])),
            ('.', 0, array([0.97274315, 0.01215939, 0.01509754])),
            ('If', 0, array([0.96470869, 0.0205418, 0.0147495])),
            ('the', 0, array([0.96488369, 0.02097801, 0.01413829])),
            ('authorities', 0, array([0.97198874, 0.00554673, 0.02246451])),
            ('thing', 0, array([0.96913761, 0.00376531, 0.02709704])),
            ('it', 0, array([0.97268265, 0.0044065, 0.02291081])),
            ('is', 0, array([0.93735614, 0.01033455, 0.05230937]))
        ]
        expected = [(a, b) for a, b, c in predictions]
        result = take_single_entity_for_each_class(predictions)
        self.assertListEqual(result, expected)

    def test_take_single_entity_for_each_class(self):
        predictions = [
            ('The', 0, array([0.93692135, 0.01944068, 0.043638])),
            ('authoritites', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('specified', 2, array([0.04215204, 0.00930096, 0.94854707])),
            ('in', 2, array([0.05586903, 0.0064729, 0.93765807])),
            ('paragraph', 2, array([0.06311218, 0.00647875, 0.93040907])),
            ('2', 2, array([0.04869084, 0.00783127, 0.94347787])),
            ('must', 0, array([0.9694128, 0.00716435, 0.02342291])),
            ('write', 0, array([0.97327912, 0.00752228, 0.01919858])),
            ('a', 0, array([0.9745782, 0.00830096, 0.01712076])),
            ('letter', 0, array([0.97538519, 0.00681078, 0.01780402])),
            ('with', 0, array([0.9730261, 0.00481166, 0.02216221])),
            ('the', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('notice', 0, array([0.97576523, 0.00637131, 0.01786346])),
            ('period', 0, array([0.97400716, 0.0075533, 0.01843956])),
            ('.', 0, array([0.97274315, 0.01215939, 0.01509754])),
            ('If', 0, array([0.96470869, 0.0205418, 0.0147495])),
            ('the', 0, array([0.96488369, 0.02097801, 0.01413829])),
            ('authorities', 0, array([0.97198874, 0.00554673, 0.02246451])),
            ('thing', 0, array([0.96913761, 0.00376531, 0.02709704])),
            ('it', 0, array([0.97268265, 0.0044065, 0.02291081])),
            ('is', 0, array([0.93735614, 0.01033455, 0.05230937]))
        ]
        expected = [
            ('The', 0),
            ('authoritites', 1),
            ('specified', 2),
            ('in', 2),
            ('paragraph', 2),
            ('2', 2),
            ('must', 0),
            ('write', 0),
            ('a', 0),
            ('letter', 0),
            ('with', 0),
            ('the', 0),
            ('notice', 0),
            ('period', 0),
            ('.', 0),
            ('If', 0),
            ('the', 0),
            ('authorities', 0),
            ('thing', 0),
            ('it', 0),
            ('is', 0)
        ]
        result = take_single_entity_for_each_class(predictions)
        self.assertListEqual(result, expected)

        predictions = [
            ('The', 0, array([0.93692135, 0.01944068, 0.043638])),
            ('authoritites', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('specified', 2, array([0.04215204, 0.00930096, 0.94854707])),
            ('in', 2, array([0.05586903, 0.0064729, 0.93765807])),
            ('paragraph', 2, array([0.06311218, 0.00647875, 0.93040907])),
            ('2', 2, array([0.04869084, 0.00783127, 0.94347787])),
            ('must', 0, array([0.9694128, 0.00716435, 0.02342291])),
            ('write', 0, array([0.97327912, 0.00752228, 0.01919858])),
            ('a', 0, array([0.9745782, 0.00830096, 0.01712076])),
            ('letter', 0, array([0.97538519, 0.00681078, 0.01780402])),
            ('with', 0, array([0.9730261, 0.00481166, 0.02216221])),
            ('the', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('notice', 1, array([0.00486542, 0.9818131, 0.04332148])),
            ('period', 0, array([0.97400716, 0.0075533, 0.01843956])),
            ('.', 0, array([0.97274315, 0.01215939, 0.01509754])),
            ('If', 0, array([0.96470869, 0.0205418, 0.0147495])),
            ('the', 0, array([0.96488369, 0.02097801, 0.01413829])),
            ('authorities', 0, array([0.97198874, 0.00554673, 0.02246451])),
            ('thing', 0, array([0.96913761, 0.00376531, 0.02709704])),
            ('it', 0, array([0.97268265, 0.0044065, 0.02291081])),
            ('is', 0, array([0.93735614, 0.01033455, 0.05230937]))
        ]
        expected = [
            ('The', 0),
            ('authoritites', 0),
            ('specified', 2),
            ('in', 2),
            ('paragraph', 2),
            ('2', 2),
            ('must', 0),
            ('write', 0),
            ('a', 0),
            ('letter', 0),
            ('with', 0),
            ('the', 1),
            ('notice', 1),
            ('period', 0),
            ('.', 0),
            ('If', 0),
            ('the', 0),
            ('authorities', 0),
            ('thing', 0),
            ('it', 0),
            ('is', 0)
        ]
        result = take_single_entity_for_each_class(predictions)
        self.assertListEqual(result, expected)

        predictions = [
            ('The', 0, array([0.93692135, 0.01944068, 0.043638])),
            ('authoritites', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('specified', 2, array([0.04215204, 0.00930096, 0.94854707])),
            ('in', 2, array([0.05586903, 0.0064729, 0.93765807])),
            ('paragraph', 2, array([0.06311218, 0.00647875, 0.93040907])),
            ('2', 2, array([0.04869084, 0.00783127, 0.94347787])),
            ('must', 0, array([0.9694128, 0.00716435, 0.02342291])),
            ('write', 0, array([0.97327912, 0.00752228, 0.01919858])),
            ('a', 0, array([0.9745782, 0.00830096, 0.01712076])),
            ('letter', 0, array([0.97538519, 0.00681078, 0.01780402])),
            ('with', 0, array([0.9730261, 0.00481166, 0.02216221])),
            ('the', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('period', 0, array([0.97400716, 0.0075533, 0.01843956])),
            ('notice', 1, array([0.00486542, 0.9818131, 0.04332148])),
            ('.', 0, array([0.97274315, 0.01215939, 0.01509754])),
            ('If', 0, array([0.96470869, 0.0205418, 0.0147495])),
            ('the', 0, array([0.96488369, 0.02097801, 0.01413829])),
            ('authorities', 0, array([0.97198874, 0.00554673, 0.02246451])),
            ('thing', 0, array([0.96913761, 0.00376531, 0.02709704])),
            ('it', 0, array([0.97268265, 0.0044065, 0.02291081])),
            ('is', 0, array([0.93735614, 0.01033455, 0.05230937]))
        ]
        expected = [
            ('The', 0),
            ('authoritites', 0),
            ('specified', 2),
            ('in', 2),
            ('paragraph', 2),
            ('2', 2),
            ('must', 0),
            ('write', 0),
            ('a', 0),
            ('letter', 0),
            ('with', 0),
            ('the', 0),
            ('period', 0),
            ('notice', 1),
            ('.', 0),
            ('If', 0),
            ('the', 0),
            ('authorities', 0),
            ('thing', 0),
            ('it', 0),
            ('is', 0)
        ]
        result = take_single_entity_for_each_class(predictions)
        self.assertListEqual(result, expected)

        predictions = [
            ('The', 0, array([0.93692135, 0.01944068, 0.043638])),
            ('authoritites', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('specified', 2, array([0.04215204, 0.00930096, 0.94854707])),
            ('in', 2, array([0.05586903, 0.0064729, 0.93765807])),
            ('paragraph', 2, array([0.06311218, 0.00647875, 0.93040907])),
            ('2', 2, array([0.04869084, 0.00783127, 0.94347787])),
            ('must', 0, array([0.9694128, 0.00716435, 0.02342291])),
            ('write', 0, array([0.97327912, 0.00752228, 0.01919858])),
            ('a', 0, array([0.9745782, 0.00830096, 0.01712076])),
            ('letter', 0, array([0.97538519, 0.00681078, 0.01780402])),
            ('with', 0, array([0.9730261, 0.00481166, 0.02216221])),
            ('the', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('notice', 1, array([0.00486542, 0.9818131, 0.04332148])),
            ('period', 0, array([0.97400716, 0.0075533, 0.01843956])),
            ('.', 0, array([0.97274315, 0.01215939, 0.01509754])),
            ('If', 0, array([0.96470869, 0.0205418, 0.0147495])),
            ('the', 0, array([0.96488369, 0.02097801, 0.01413829])),
            ('authorities', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('thing', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('it', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('is', 0, array([0.93735614, 0.01033455, 0.05230937]))
        ]
        expected = [
            ('The', 0),
            ('authoritites', 0),
            ('specified', 2),
            ('in', 2),
            ('paragraph', 2),
            ('2', 2),
            ('must', 0),
            ('write', 0),
            ('a', 0),
            ('letter', 0),
            ('with', 0),
            ('the', 1),
            ('notice', 1),
            ('period', 0),
            ('.', 0),
            ('If', 0),
            ('the', 0),
            ('authorities', 0),
            ('thing', 0),
            ('it', 0),
            ('is', 0)
        ]
        result = take_single_entity_for_each_class(predictions)
        self.assertListEqual(result, expected)

        predictions = [
            ('The', 0, array([0.93692135, 0.01944068, 0.043638])),
            ('authoritites', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('specified', 2, array([0.04215204, 0.00930096, 0.94854707])),
            ('in', 2, array([0.05586903, 0.0064729, 0.96765807])),
            ('paragraph', 2, array([0.06311218, 0.00647875, 0.96040907])),
            ('2', 2, array([0.04869084, 0.00783127, 0.96347787])),
            ('must', 0, array([0.9694128, 0.00716435, 0.02342291])),
            ('write', 0, array([0.97327912, 0.00752228, 0.01919858])),
            ('a', 0, array([0.9745782, 0.00830096, 0.01712076])),
            ('letter', 0, array([0.97538519, 0.00681078, 0.01780402])),
            ('with', 0, array([0.9730261, 0.00481166, 0.02216221])),
            ('the', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('notice', 1, array([0.00486542, 0.9818131, 0.04332148])),
            ('period', 0, array([0.97400716, 0.0075533, 0.01843956])),
            ('.', 2, array([0.04215204, 0.00930096, 0.94854707])),
            ('If', 2, array([0.04215204, 0.00930096, 0.94854707])),
            ('the', 0, array([0.96488369, 0.02097801, 0.01413829])),
            ('authorities', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('thing', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('it', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('is', 0, array([0.93735614, 0.01033455, 0.05230937]))
        ]
        expected = [
            ('The', 0),
            ('authoritites', 0),
            ('specified', 2),
            ('in', 2),
            ('paragraph', 2),
            ('2', 2),
            ('must', 0),
            ('write', 0),
            ('a', 0),
            ('letter', 0),
            ('with', 0),
            ('the', 1),
            ('notice', 1),
            ('period', 0),
            ('.', 0),
            ('If', 0),
            ('the', 0),
            ('authorities', 0),
            ('thing', 0),
            ('it', 0),
            ('is', 0)
        ]
        result = take_single_entity_for_each_class(predictions)
        self.assertListEqual(result, expected)

        predictions = [
            ('The', 0, array([0.93692135, 0.01944068, 0.043638])),
            ('authoritites', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('specified', 2, array([0.04215204, 0.00930096, 0.94854707])),
            ('in', 2, array([0.05586903, 0.0064729, 0.93765807])),
            ('paragraph', 2, array([0.06311218, 0.00647875, 0.93040907])),
            ('2', 2, array([0.04869084, 0.00783127, 0.94347787])),
            ('must', 0, array([0.9694128, 0.00716435, 0.02342291])),
            ('write', 0, array([0.97327912, 0.00752228, 0.01919858])),
            ('a', 0, array([0.9745782, 0.00830096, 0.01712076])),
            ('letter', 0, array([0.97538519, 0.00681078, 0.01780402])),
            ('with', 0, array([0.9730261, 0.00481166, 0.02216221])),
            ('the', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('notice', 1, array([0.00486542, 0.9818131, 0.04332148])),
            ('period', 0, array([0.97400716, 0.0075533, 0.01843956])),
            ('.', 2, array([0.04215204, 0.00930096, 0.98854707])),
            ('If', 2, array([0.04215204, 0.00930096, 0.98854707])),
            ('the', 0, array([0.96488369, 0.02097801, 0.01413829])),
            ('authorities', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('thing', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('it', 1, array([0.00486542, 0.9518131, 0.04332148])),
            ('is', 0, array([0.93735614, 0.01033455, 0.05230937]))
        ]
        expected = [
            ('The', 0),
            ('authoritites', 0),
            ('specified', 0),
            ('in', 0),
            ('paragraph', 0),
            ('2', 0),
            ('must', 0),
            ('write', 0),
            ('a', 0),
            ('letter', 0),
            ('with', 0),
            ('the', 1),
            ('notice', 1),
            ('period', 0),
            ('.', 2),
            ('If', 2),
            ('the', 0),
            ('authorities', 0),
            ('thing', 0),
            ('it', 0),
            ('is', 0)
        ]
        result = take_single_entity_for_each_class(predictions)
        self.assertListEqual(result, expected)
