from abc import abstractmethod
from typing import List, Any, Set

import numpy as np


class GroundTruthItem:
    def __init__(self, *, clazz: str, location: Any = None) -> None:
        """
        Args:
            clazz: the class of the item.
            location: the location of the item.
                In the case of detection, this is the image id where the box come from.
        """
        self.clazz = clazz
        self.location = location


class PredictedItem:
    def __init__(self, *, clazz: str, score: float, location: Any = None) -> None:
        """
        Args:
            clazz: the class of the item.
            score: the score of the item for the class.
            location: the location of the item.
                In the case of detection, this is the image id where the box come from.
        """
        self.clazz = clazz
        self.score = score
        self.location = location


class MeanAveragePrecision:
    def __init__(self, gt_items: Set[GroundTruthItem], predicted_items: Set[PredictedItem]):
        self.gt_items = gt_items
        self.predicted_items = predicted_items


    def mAP(self, **kwargs) -> float:
        """
        Code modified from https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
        """
        all_average_precisions = []
        all_classes = {b.clazz for b in self.gt_items.union(self.predicted_items)}

        for c in all_classes:
            pred_items_single_class = [d for d in self.predicted_items if d.clazz == c]
            ground_truths_single_class = [g for g in self.gt_items if g.clazz == c]

            if not ground_truths_single_class and pred_items_single_class:
                average_precision = 0.0
            elif ground_truths_single_class and not pred_items_single_class:
                average_precision = 0.0
            elif not ground_truths_single_class and not pred_items_single_class:
                average_precision = 1.0
            else:
                pred_items_single_class = sorted(pred_items_single_class, key=lambda d: d.score, reverse=True)
                prediction_is_correct = self.assign(pred_items_single_class, ground_truths_single_class, **kwargs)

                acc_TP = np.cumsum(prediction_is_correct)
                acc_FP = np.cumsum(1 - prediction_is_correct)
                rec = list(acc_TP / len(ground_truths_single_class))
                prec = list(acc_TP / (acc_FP + acc_TP))
                average_precision = self._average_precision(rec, prec)

            all_average_precisions.append(average_precision)

        return np.mean(all_average_precisions)

    @abstractmethod
    def assign(self, predicted_items_single_class, gt_items_single_class, **kwargs) -> np.ndarray:
        """
        Args:
            predicted_items_single_class: sorted list of PredictedItem of a single class.
            gt_items_single_class: sorted list of GroundTruthItem of a single class.
        Return:
            A 1-d np.ndarray of the same length as predicted_items_single_class, with each value 
            being either 0 or 1, indicating whether the corresponding predicted item is correct or not.
        """
        pass
    
    @staticmethod
    def _average_precision(rec: List[float], prec: List[float]) -> float:
        recall_intervals = [r for r in [0] + rec]
        precision_intervals = [p for p in [0] + prec]

        average_precision = 0
        for i in range(len(recall_intervals) - 1):
            average_precision += (recall_intervals[i + 1] - recall_intervals[i]) * max(
                precision_intervals[i + 1 :]
            )

        return average_precision