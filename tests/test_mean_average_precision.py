import math
import pytest
import numpy as np

from detection_mean_ap import (
    BoxMeanAveragePrecision,
    PredictedBoundingBox,
    GroundTruthBoundingBox,
)
from utils import string_to_boundingbox


def test_mean_average_precision():
    """
    Two correct predictions for one ground truths.
    In this case even though there should only be one prediction for one ground truth,
    the AP should still be 1.0 because:
        precision is 1.0 @ recall=1.0

    """
    gt = GroundTruthBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="1")
    pred1 = PredictedBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="1", score=0.9)
    pred2 = PredictedBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="1", score=0.9)

    mAP = BoxMeanAveragePrecision({gt}, {pred1, pred2}).mAP(iou_threshold=0.5)
    assert mAP == 1.0


def test_mean_average_precision_2():
    """
    One correct box and one wrong box which got the class wrong
    In this case AP should be 0.5 because:
        for class "a" AP is 1.0
        for class "b" AP is 0.0
    """
    gt = GroundTruthBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="1")
    pred1 = PredictedBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="1", score=0.9)
    pred2 = PredictedBoundingBox(clazz="b", box=np.array([10, 10, 20, 20]), image_id="1", score=0.9)

    mAP = BoxMeanAveragePrecision({gt}, {pred1, pred2}).mAP(iou_threshold=0.5)
    assert mAP == 0.5


def test_mean_average_precision_3():
    """
    No correct predictions because the class were wrong.
    AP should be 0.
    """
    gt = GroundTruthBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="1")
    pred1 = PredictedBoundingBox(clazz="c", box=np.array([10, 10, 20, 20]), image_id="1", score=0.9)
    pred2 = PredictedBoundingBox(clazz="b", box=np.array([10, 10, 20, 20]), image_id="1", score=0.9)

    mAP = BoxMeanAveragePrecision({gt}, {pred1, pred2}).mAP(iou_threshold=0.5)
    assert mAP == 0.0


def test_mean_average_precision_4():
    """
    One wrong prediction because of wrong location (image_id), but has lower score.
    In this case AP is still 1.0 because the wrong prediction has lower score
    """
    gt = GroundTruthBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="1")
    pred1 = PredictedBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="1", score=0.9)
    pred2 = PredictedBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="2", score=0.8)

    mAP = BoxMeanAveragePrecision({gt}, {pred1, pred2}).mAP(iou_threshold=0.5)
    assert mAP == 1.0


def test_mean_average_precision_5():
    """
    One wrong prediction because of wrong location (image_id), but has higher score.
    In this case AP should be the precision value @recall=1.0
    """
    gt = GroundTruthBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="1")
    pred1 = PredictedBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="1", score=0.8)
    pred2 = PredictedBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="2", score=0.9)

    mAP = BoxMeanAveragePrecision({gt}, {pred1, pred2}).mAP(iou_threshold=0.5)
    assert mAP == 0.5


def test_mean_average_precision_6():
    """
    Multiple wrong prediction because of wrong location (box coordinates), but all with higher scores.
    In this case AP should be the precision value @recall=1.0
    """
    gt = GroundTruthBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="1")
    pred1 = PredictedBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="1", score=0.8)
    pred2 = PredictedBoundingBox(
        clazz="a", box=np.array([50, 50, 60, 60]), image_id="1", score=0.85
    )
    pred3 = PredictedBoundingBox(
        clazz="a", box=np.array([50, 50, 60, 60]), image_id="1", score=0.86
    )
    pred4 = PredictedBoundingBox(
        clazz="a", box=np.array([50, 50, 60, 60]), image_id="1", score=0.87
    )
    pred5 = PredictedBoundingBox(
        clazz="a", box=np.array([50, 50, 60, 60]), image_id="1", score=0.88
    )
    pred6 = PredictedBoundingBox(
        clazz="a", box=np.array([50, 50, 60, 60]), image_id="1", score=0.89
    )
    pred7 = PredictedBoundingBox(clazz="a", box=np.array([50, 50, 60, 60]), image_id="1", score=0.9)

    mAP = BoxMeanAveragePrecision({gt}, {pred1, pred2, pred3, pred4, pred5, pred6, pred7}).mAP(
        iou_threshold=0.5
    )
    assert math.isclose(mAP, 1 / 7)


def test_mean_average_precision_7():
    gt = GroundTruthBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="1")
    pred1 = PredictedBoundingBox(clazz="a", box=np.array([15, 15, 25, 25]), image_id="1", score=0.9)
    pred2 = PredictedBoundingBox(clazz="b", box=np.array([10, 10, 20, 20]), image_id="1", score=0.9)

    mAP = BoxMeanAveragePrecision({gt}, {pred1, pred2}).mAP(iou_threshold=0.5)
    assert mAP == 0.0


def test_mean_average_precision_8():
    gt = GroundTruthBoundingBox(clazz="a", box=np.array([10, 10, 20, 20]), image_id="1")
    pred1 = PredictedBoundingBox(clazz="a", box=np.array([11, 11, 20, 20]), image_id="1", score=0.9)
    pred2 = PredictedBoundingBox(clazz="b", box=np.array([10, 10, 20, 20]), image_id="1", score=0.9)

    mAP = BoxMeanAveragePrecision({gt}, {pred1, pred2}).mAP(iou_threshold=0.5)
    assert mAP == 0.5



def mAP_from_file(gt_path, prediction_path, str_format, iou_threshold):
    with open(prediction_path) as fh:
        pred_boxes = {
            string_to_boundingbox(l, "pred", str_format="xywh") for l in fh.readlines()
        }

    with open(gt_path) as fh:
        gt_boxes = {
            string_to_boundingbox(l, "gt", str_format="xywh") for l in fh.readlines()
        }

    return BoxMeanAveragePrecision(gt_boxes, pred_boxes).mAP(iou_threshold=iou_threshold)


@pytest.mark.parametrize(
    "iou_threshold, expected_mAP", [(0.3, 0.2253968), (0.5, 0.0222222), (0.7, 0.0)]
)
def test_mean_average_precision_real_example(iou_threshold, expected_mAP):
    gt_detection_txt_path = "tests/data/gt_boxes_xywh.txt"
    pred_detection_txt_path = "tests/data/predicted_boxes_xywh.txt"
    mAP = mAP_from_file(
        gt_detection_txt_path,
        pred_detection_txt_path,
        str_format="xywh",
        iou_threshold=iou_threshold,
    )
    assert math.isclose(mAP, expected_mAP, abs_tol=1e-6)
