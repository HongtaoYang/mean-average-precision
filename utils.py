from typing import Union

import numpy as np

from detection_mean_ap import PredictedBoundingBox, GroundTruthBoundingBox


def string_to_boundingbox(
    string: str, box_type: str, str_format: str = "xyxy"
) -> Union[GroundTruthBoundingBox, PredictedBoundingBox]:
    """
    Args:
        string: a string representation of a bounding box with format "class image_id xmin ymin xmax ymax score"
        box_type: whether the bounding box is a predicted one or a ground truth.
        str_format: the bounding box format, one of ["xyxy", "xxyy", "xywh"].
    Return:
        Either a GTBoundingBox or a PredictedBoundingBox object
    """
    # TODO: support normalized boxes, i.e. value between 0, 1.
    if str_format == "xyxy":
        clazz, image_id, score, xmin, ymin, xmax, ymax = string.split(" ")
    elif str_format == "xxyy":
        clazz, image_id, score, xmin, xmax, ymin, ymax = string.split(" ")
    elif str_format == "xywh":
        clazz, image_id, score, xmin, ymin, w, h = string.split(" ")
        xmax = int(xmin) + int(w)
        ymax = int(ymin) + int(h)
    else:
        raise ValueError(f"Unknown string format: {str_format}")

    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)
    score = float(score)

    if box_type == "pred":
        return PredictedBoundingBox(
            clazz=str(clazz),
            image_id=str(image_id),
            score=score,
            box=np.array([xmin, ymin, xmax, ymax]),
        )
    elif box_type == "gt":
        return GroundTruthBoundingBox(
            clazz=str(clazz), image_id=str(image_id), box=np.array([xmin, ymin, xmax, ymax]),
        )
    else:
        raise ValueError(f"Unknown bounding box type: {box_type}.")