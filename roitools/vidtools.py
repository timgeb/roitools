# pythonic interface for some cv2 functionality

# TODO port to Python 3 ...

from collections import namedtuple, OrderedDict
import os

import cv2
import numpy as np

# gather capture properties defined by cv2
# and create mapping property name -> property id
# (uses OrderedDict for consistent output from Frame.info)
_cv2version = int(cv2.__version__[0])

if _cv2version <= 2:
    _prop_prefix = 'CV_CAP_PROP_'
    _consts_loc = cv2.cv
else:
    _prop_prefix = 'CAP_PROP_'
    _consts_loc = cv2

_prop_names = sorted(
    x for x in _consts_loc.__dict__ if x.startswith(_prop_prefix))

_prop2id = OrderedDict(
    (name[len(_prop_prefix):].lower(), getattr(_consts_loc, name))
    for name in _prop_names)

# gather events defined by cv2
# and create mapping name -> event id
_ev_prefix = 'EVENT_'
_ev_names = [x for x in cv2.__dict__ if x.startswith(_ev_prefix)]
_ev2id = {ev[len(_ev_prefix):].lower() : getattr(cv2, ev) for ev in _ev_names}
_id2ev = {id_:event for event, id_ in _ev2id.iteritems()}

# module level functions
imwrite = cv2.imwrite

def wait(delay):
    'wait delay milliseconds, to be used after showing a frame'

    # cv2.waitKey with a delay <= 0 pauses, we suppress this behavior
    # and have a pause function instead
    if delay > 0:
        cv2.waitKey(delay)

def pause():
    'wait indefinitely, to be used after showing a frame'
    cv2.waitKey(0)

class PyCap(object):
    'Python adapter/facade for cv2.VideoCapture'

    def __init__(self, video, title=None):
        'PyCap(path) -> PyCap object'
        if not os.path.isfile(video):
            raise IOError("file not found: '{}'".format(video))

        self._cv2cap = cv2.VideoCapture(video)
        self.video = video
        self.title = str(video) if title is None else title
        self.open()

    def open(self):
        'open the capture if not already opened'
        if not self._cv2cap.isOpened():
            self._cv2cap.open(self.video)

    def __getattr__(self, attr):
        # allow accessing capture properties with dot notation, e.g.
        # cap.buffersize instead of cap.get(cv2.cv.CV_CAP_PROP_BUFFERSIZE)
        if hasattr(self, '_cv2cap') and attr in _prop2id:
            return self._cv2cap.get(_prop2id[attr])

        msg = "'{}' has no attribute '{}'".format(type(self).__name__, attr)
        raise AttributeError(msg)

    def __setattr__(self, attr, value):
        # allow setting capture properties with dot notation, e.g.
        # cap.pos_msec = 2000 vs cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, 2000)
        if hasattr(self, '_cv2cap') and attr in _prop2id:
            success = self._cv2cap.set(_prop2id[attr], value)

            # VideoCapture.set returns False if the property cannot be set
            if not success:
                raise AttributeError('read only or unknown property')
        else:
            super(PyCap, self).__setattr__(attr, value)

    def info(self):
        'return video information'
        template = '{{: <{}}} {{}}\n'.format(len(max(_prop2id, key=len)))
        info = ''.join(
            [template.format(prop, getattr(self, prop)) for prop in _prop2id])

        return info + template.format(
            'len/s', self.frame_count/(self.fps or np.nan))

    def __iter__(self):
        return self

    def next(self):
        'get next frame'
        success, array = self._cv2cap.read()
        if not success:
            raise StopIteration
        return Frame(array, self.title)

    def release(self):
        self._cv2cap.release()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release() # 'rewind and close'
        self.open()

    def play(self, delay=None, start=None, stop=None, rewind=False):
        '''play the video from current position

        delay: wait delay milliseconds after showing a frame;
        if delay is None, waits int(ceil(1000.0/fps)) ms if fps
        is available, else waits 1 ms
        start: play from start pos_msec or from current position
        if start is None
        stop: play to stop pos_msec or play to last frame and rewind
        if stop is None
        rewind: if True, release and reopen capture if played to end'''
        if delay is None:
            has_fps = not (np.isnan(self.fps) or self.fps == 0)
            delay = int(np.ceil(1000.0/self.fps)) if has_fps else 1

        if start is not None:
            self.pos_msec = start

        if stop is None:
            stop = np.nan

        self.open()

        for frame in self:
            # if the video lies about its true fps, i.e. fps has been upscaled
            # by repeat-last-frame instructions, the self.pos_msec > stop
            # check will overshoot the destination time
            if self.pos_msec > stop:
                break

            if delay > 0:
                frame.show()
                wait(delay)
        else:
            # played to end -> "rewind and reopen"
            if rewind:
                self._cv2cap.release()
                self.open()

# classes for self-documenting drawing
Point = namedtuple('Point', 'x y')
Color = namedtuple('Color', 'blue green red') # BGR!

# for the intricacies of sublassing ndarray, see
# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
class Frame(np.ndarray):
    # TODO: does currently not play nice with some ufuncs,
    # e.g. np.copy(Frame object) returns ndarray instance
    # and converting to masked array masks properties and methods
    # -> needs __array_wrap__ implemented
    '''Frame(array, title) -> Frame object

    encapsulates common frame operations, e.g. frame.show() instead of
    cv2.imshow(title, frame) or frame.blue instead of frame[:, :, 0]'''

    def __new__(cls, input_array, title='unnamed'):
        '''Frame(numpy array, title) -> Frame

        array must be two-dimensional (grayscale image) or three-dimensional
        with array.shape[2] == 3 (bgr image) and of type np.uint8'''
        # view calls ndarray.__new__ which calls Frame.__array_finalize__
        # with input_array as the second parameter (from_array)
        self = input_array.view(cls)

        # set title in case Frame was instantiated explicitly through
        # the constructor, override the value just set in __array_finalize__
        self.title = title
        return self

    def __array_finalize__(self, from_array):
        # from_array: template array (e.g. from slicing) or viewed array
        #
        # if from_array is None, for some reason ndarray.__new__(Frame,
        # ((10,))) or similar got called, i.e. __new__ on ndarray was invoked
        # directly in order to instantiate a Frame; I don't know why
        # this would happen
        #
        # when instantiating a Frame directly (i.e. Frame.__new__ is
        # called) or constructing through a view or template, from_array cannot
        # be None
        if from_array is not None:
            # when constructing a Frame directly, the fallback getattr value
            # will be overriden once __array_finalize__ returns to __new__,
            # without a fallback value getattr would raise an AttributeError
            # in this scenario
            #
            # when constructing a Frame from view casting of a non-Frame
            # e.g. arange(3).view(Frame), Frame.__new__ is never called
            # so the fallback value from this method will be used
            #
            # if creating a new instance/view from an existing Frame (i.e.
            # title is set) we just forward the value to the new instance
            self.title = getattr(from_array, 'title', 'unnamed')

    def show(self):
        '''shows the frame

        use module level function wait(delay) after call to show to
        slow playback when iterating over a sequence of frames'''
        cv2.imshow(self.title, self)

    def draw_circle(self, center_xy, radius, color=(0, 255, 0), thickness=1,
        linetype=8, shift=0):
        '''draw_circle((x, y), radius[, (blue, green, red)[, thickness[,
        linetype[, shift]]]]) -> None

        draws simple circle on the frame
        set thickness=-1 to fill circle
        see cv2.line documentation for optional linetype and shift arguments'''
        cv2.circle(self, center_xy, radius, color, thickness, linetype, shift)

    def draw_rectangle(self, vertex1, vertex2, color=(0, 255, 0),
        thickness=1, linetype=8, shift=0):
        '''draw_rectangle((x1, y1), (x2, y2)[, (blue, green, red)[, thickness[,
        linetype[, shift]]]]) -> None

        draws simple rectangle on the frame
        vertex2 is the vertex of the rectangle opposite to vertex1
        set thickness=-1 to fill rectangle
        see cv2.line documentation for optional linetype and shift arguments'''
        cv2.rectangle(self, vertex1, vertex2, color, thickness,
            linetype, shift)

    def save(self, filename):
        '''saves image, returns True on success
        examples: frame.save('screenshot.png') -> True
                  frame.save('/home/tim/screenshot.bmp') -> True'''
        return cv2.imwrite(filename, self)

    # TODO: documentation for other drawing methods, make *args explicit
    def put_text(self, *args):
        cv2.putText(self, *args)

    def draw_line(self, *args):
        cv2.line(self, *args)

    def draw_ellipse(self, *args):
        cv2.ellipse(self, *args)

    def draw_polygon(self, *args):
        cv2.polygon(self, *args)

    @property
    def blue(self):
        'blue channel data'
        return self[:, :, 0]

    @blue.setter
    def blue(self, blue):
        self[:, :, 0] = blue

    @property
    def green(self):
        'green channel data'
        return self[:, :, 1]

    @green.setter
    def green(self, green):
        self[:, :, 1] = green

    @property
    def red(self):
        'red channel data'
        return self[:, :, 2]

    @red.setter
    def red(self, red):
        self[:, :, 2] = red
