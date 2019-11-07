# roitools

Tools and helpers for planning and analyzing regions of interest (ROIs) in video files.

Author contact: tim.gebensleben@gmail.com

This software consists of three main parts:

- an easy to use wrapper/facade for the `cv2.VideoCapture` class called `PyCap` (`vidtools.py`)
- an observable subclass `RoiCap` to which regions of interest can be added as observers (`roitools.py`)
- a graphical user interface for planning ROIs (`roi_planner_gui.py`).

The following sections give a short overview of the essential features. Examples for functions building upon the tools provided are located in the `examples` directory.

## `PyCap` and `Frame` objects

The `PyCap` class is a wrapper for `cv2.VideoCapture` and acts as a video from which frames can be read.

It offers three main features:

- accessing and setting of attributes via the dot notation
- instances of `PyCap` are iterable
- iteration yields `Frame` objects which are `numpy` arrays with useful methods.

The following code snippets highlight the difference between a `VideoCapture` and a `PyCap`. They investigate a sample video between second ten and twelve and show each frame.

Using a `VideoCapture`:

```python
import cv2

video = cv2.VideoCapture('sample.avi')
video.set(cv2.CAP_PROP_POS_MSEC, 10000)

while True:
    success, frame = video.read()

    if not success:
        break

    msec = video.get(cv2.CAP_PROP_POS_MSEC)

    if msec > 12000:
        break

    # perform some operation on the frame here
    # ...

    cv2.imshow('sample.avi', frame)
    cv2.waitKey(25)
```

Equivalent code using a `PyCap`:

```python
import vidtools

video = vidtools.PyCap('sample.avi')
video.pos_msec = 10000

for frame in video:
    if video.pos_msec > 12000:
        break

    # perform some operation on the frame here
    # ...

    frame.show()
    vidtools.wait(25)
```

If no operation needs to be performed on the frame and the goal is to simply play the video, the second snippet can be written even simpler.

```python
import vidtools

video = vidtools.PyCap('sample.avi')
video.play(delay=25, start=10000, stop=12000)
```

## `RoiCap` and ROI objects

TODO

## ROI Planning

TODO
