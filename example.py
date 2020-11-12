from detection_mean_ap import BoxMeanAveragePrecision
from utils import string_to_boundingbox


def main():
    with open("tests/data/predicted_boxes_xywh.txt") as fh:
        pred_boxes = {
            string_to_boundingbox(l, "pred", str_format="xywh") for l in fh.readlines()
        }

    with open("tests/data/gt_boxes_xywh.txt") as fh:
        gt_boxes = {
            string_to_boundingbox(l, "gt", str_format="xywh") for l in fh.readlines()
        }

    mAP = BoxMeanAveragePrecision(gt_boxes, pred_boxes).mAP(iou_threshold=0.5)
    print(mAP)


if __name__ == "__main__":
    main()