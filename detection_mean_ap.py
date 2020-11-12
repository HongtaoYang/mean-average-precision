from typing import Union
from collections import Counter

import numpy as np

from base import GroundTruthItem, PredictedItem, MeanAveragePrecision


class GroundTruthBoundingBox(GroundTruthItem):
    def __init__(self, *, clazz: str, image_id: Union[int, str], box: np.ndarray) -> None:
        super().__init__(clazz=clazz, location=image_id)
        self.box = box

class PredictedBoundingBox(PredictedItem):
    def __init__(
        self, *, clazz: str, score: float, image_id: Union[int, str], box: np.ndarray
    ) -> None:
        super().__init__(clazz=clazz, location=image_id, score=score)
        self.box = box


class BoxMeanAveragePrecision(MeanAveragePrecision):
    def assign(self, predicted_items_single_class, gt_items_single_class, iou_threshold=0.5):
        gt_covered = {}
        for img_id, count in Counter([bb.location for bb in gt_items_single_class]).items():
            gt_covered[img_id] = np.zeros(count)

        detection_is_correct = np.zeros(len(predicted_items_single_class))
        for i, det_box in enumerate(predicted_items_single_class):
            ground_truths_on_same_image = [
                g for g in self.gt_items if g.location == det_box.location
            ]

            max_iou = -1
            for j, gt_box in enumerate(ground_truths_on_same_image):
                iou = calculate_iou(det_box.box, np.expand_dims(gt_box.box, axis=0))
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = j

            if max_iou >= iou_threshold:
                if gt_covered[det_box.location][max_iou_idx] == 0:  # if flag not set
                    detection_is_correct[i] = 1  # count as true positive
                    gt_covered[det_box.location][max_iou_idx] = 1  # set flag as already 'seen'
                else:
                    detection_is_correct[i] = 0  # count as false positive
            else:
                detection_is_correct[i] = 0  # count as false positive

        return detection_is_correct


def calculate_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Calculate IoU between a single target box and a batch of boxes.
    Every box is in the format [xmin, ymin, xmax, ymax].
    Args:
        box: np.ndarray with shape (4,).
        boxes: np.ndarray with shape (n, 4), where n is number of boxes.
    """
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    union = box_area + boxes_area - inter
    union = union.astype(float)
    union[union <= 1e-8] = 1e-8
    return inter / union