Mean average precision is a widly used evaluation metric in detection and other tasks. However, I didn't find a solution that fits my requirements, which are easy to understand, extensible and clean.

This repo is just some code snippets that calculates mAP in my prefered way.
The calculation follows PASCAL VOC convention where precision is averaged over all recall values (as opposed to 11-points interpoation.)

This repo was inspired by https://github.com/rafaelpadilla/Object-Detection-Metrics#interpolating-all-points, where the author gives a detailed tutorial on mAP. But if you just want something that just work, and is more extensible, cleaner and more pythonic, then this repo is a good fit.

## General Design
Mean Average Precision is not just for object detection, other tasks such as recognition can also use this metirc, with sightly different implementation. That's why I want to make the code easily extensible.

There is a base `MeanAveragePrecisoin` class in `base.py` that lays out the skeleton of mAP calculation, which is also the fixed part that is common to every tasks. A speciic task, like objection detection, needs to override the `assign` method of the class. An implementation is provided in `detection_mean_ap.py`, because objection detection is the most common use case for mAP.

## Usage:
Basically you need to have a set of ground truth items and a set of predicted items, pass them to the `MeanAveragePrecisoin` class and call `mAP` method.

See `example.py` for an example, where 
    1. gt bounding boxes and predicted bounding boxes are wirtten to txt files;
    2. I load them and covert the format;
    3. calculate mAP.

I could provide a cli in the future, but this repo is meant to provide useful code snippets that you can copy to use in your own projects.


## Contribution
Welcome any imporvements, bugs fixes etx.

Konwn issue:
    1. Some stability issue. The last test in `tests/test_mean_average_precision.py` will pass sometimes and fail othertimes. Which means my function is not entirly deterministic. I have yet to identify the issue.
