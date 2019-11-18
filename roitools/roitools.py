'tools for creating and analyzing regions of interest (ROIs) in a video'

# TODO instead of having a circular ROI, implement a general elliptical ROI
# of which a circular ROI is a special case
# once this is done, support for drawing ellipses can be added in the GUI

import csv
from collections import namedtuple
from datetime import datetime
from itertools import izip, product

import numpy as np

from vidtools import Color, PyCap

# observer pattern:
# ROIs: observers, RoiCap: observable

class RoiCap(PyCap):
    'video capture that supports regions of interest (ROIs)'

    _max_rois = 100

    def __init__(self, video, title=None):
        'RoiCap(path or device id) -> RoiCap object'
        self.rois = {}
        self.latest_frame = None # latest frame read from capture
        super(RoiCap, self).__init__(video, title)

    def add_roi(self, roi):
        'register a new ROI'
        if len(self.rois) < RoiCap._max_rois:
            self.rois[roi._id] = roi
            roi.cap = self
        else:
            template = 'cannot have more than {} regions'
            msg = template.format(RoiCap._max_rois)
            raise ValueError(msg)

    def delete_roi(self, roi_id):
        'delete ROI by ID'
        del self.rois[roi_id]

    def _notify_rois(self):
        'notify all ROIs on change'
        for roi in self.rois.viewvalues():
            roi.notified()

        # drawing must take place after all ROIs collected data
        self.draw_rois()

    def draw_rois(self):
        'draw all rois'
        # can't be done by the ROIs themselves because all ROIs have to be
        # notified and collect data before the frame may be drawn on
        for roi in self.rois.viewvalues():
            roi.draw()

        # remember to show the frame AFTER drawing, remember not to
        # draw on the frame before notifying all observers

    def next(self):
        'get next frame, notify ROIs, draw ROIs'
        self.latest_frame = super(RoiCap, self).next()
        self._notify_rois()
        return self.latest_frame

# the observer
class BaseRoi(object):
    'base class to derive ROIs from'

    # next unique ID a ROI will get
    _next_id = 1

    # default colors - can be overridden in childclasses or individual instances
    active_color = Color(green=255, red=0, blue=0)
    passive_color = Color(green=200, red=200, blue=200)

    def __init__(self, vertex1, vertex2, description='N/A'):
        '''BaseRoi((x1, y1), (x2, y2)[, description])
        -> BaseRoi object

        vertex1 and vertex2 are opposite vertices of rectangular ROI

        description: optional description for ROI'''
        self._id = BaseRoi._next_id
        BaseRoi._next_id += 1

        self.deaf = False # flag used to ignore notifications
        self.cap = None # set when registered as an observer via RoiCap.add_roi
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.description = description
        self.collected = []

        # compute center of rectangle for drawing and
        # start and end indices for slicing
        self.st_x, self.end_x = sorted((vertex1[0], vertex2[0]))
        self.st_y, self.end_y = sorted((vertex1[1], vertex2[1]))
        center_x = self.st_x + (self.end_x - self.st_x)/2
        center_y = self.st_y + (self.end_y - self.st_y)/2
        self.center = (center_x, center_y)

        # increment ends by one to get correct values for slicing
        self.end_x += 1
        self.end_y += 1

        # compute the mask
        self.mask = self.compute_mask()

    def compute_mask(self):
        '''returns boolean mask (or None) used to customize shape of region
        (see CircRoi.compute_mask for an example)'''
        return None

    @property
    def roicolor(self):
        return self.passive_color if self.deaf else self.active_color

    def __repr__(self):
        return 'BaseRoi({}, {}, {}, {})'.format(self.vertex1, self.vertex2,
            self.description, str(self.mask).replace('\n', ','))

    def __str__(self):
        return 'BaseRoi(vertex1={}, vertex2={}, ...)'.format(vertex1, vertex2)

    def notified(self):
        'called on notfication from observable'
        if self.deaf:
            return

        collected = self.collect()
        if collected is not None:
            self.collected.append(collected)

    def info(self):
        'get ROI info as string'
        now = datetime.now()
        date = now.date().isoformat()
        time = now.time().strftime('%H:%M:%S')

        info = 'ROI_ID {}\nROI_TYPE {}\nROI_DESCRIPTION {}\nDATE {}\nTIME {}'
        info = info.format(self._id, self, self.description, date, time)

        if self.cap is not None:
            capinfo = 'CAP_FILE {}\nCAP_TITLE {}\nCAP_FPS {}'
            title, video, fps = self.cap.video, self.cap.title, self.cap.fps
            capinfo = capinfo.format(title, video, fps)
            info += '\n' + capinfo

        return info + '\n'

    def get_rect(self):
        'slice rectangular ROI from observed frame (does not apply mask)'
        # NOTE: applying boolean mask to 2D array flattens array!
        frame = self.cap.latest_frame
        return frame[self.st_y:self.end_y, self.st_x:self.end_x]

    def collect(self):
        '''collect data from observed capture or return None if nothing of
        interest has been observed'''
        return None

    def draw(self):
        'draw ROI with id on observed frame'
        self.draw_id()
        self.draw_outline()

    def draw_id(self):
        'draws ROI id on observed frame at center of rectangle'
        # not exactly flexible, but these parameters look ok-ish for now...
        font = 1 # HERSHEY_PLAIN
        font_scale = 0.9
        pos = self.center[0] - 4, self.center[1] + 4
        self.cap.latest_frame.put_text(
            str(self._id), pos, font, font_scale, self.roicolor)

    def draw_outline(self):
        'outline the actual ROI area (respecting masks) on observed frame'
        raise NotImplementedError

    def contains(self, point):
        'contains((x, y)) -> bool'
        raise NotImplementedError

class RectRoi(BaseRoi):
    'simple rectangular region of interest'

    def __init__(self, vertex1, vertex2, description='N/A'):
        # the reason for overriding this method was in order to adjust
        # the docstring - is there a better way?
        '''RectRoi((x1, y1), (x2, y2)[, description]) -> RectRoi object

        vertex1 and vertex2 are opposite vertices of the rectangle
        description: optional description for ROI'''
        super(RectRoi, self).__init__(vertex1, vertex2, description)

    def __repr__(self):
        return "RectRoi({}, {}, '{}')".format(
            self.vertex1, self.vertex2, self.description)

    def __str__(self):
        return 'RectRoi(vertex1={}, vertex2={})'.format(
            self.vertex1, self.vertex2)

    def draw_outline(self):
        'draw rectangular ROI area on observed frame'
        frame = self.cap.latest_frame
        frame.draw_rectangle(self.vertex1, self.vertex2, self.roicolor)

    def contains(self, point):
        'contains((x, y)) -> bool'
        x, y = point
        return self.st_x <= x < self.end_x and self.st_y <= y < self.end_y

class CircRoi(BaseRoi):
    'circular region of interest'
    # TODO support partial circles

    def __init__(self, center, radius, description='N/A'):
        '''CircRoi((x, y), radius[, description]) -> CircRoi object

        description: optional description for ROI'''
        # calling superclass-init does a few lines of redudant computation
        # such as recomputing the center (big whoop)
        v1 = (center[0] - radius, center[1] - radius)
        v2 = (center[0] + radius, center[1] + radius)
        self.radius = radius
        self.radius_sq = radius**2

        super(CircRoi, self).__init__(v1, v2, description)

    def compute_mask(self):
        dx = self.end_x - self.st_x
        dy = self.end_y - self.st_y

        mask = np.zeros((dx, dy), dtype=np.bool)
        macenter = (dx/2, dy/2)

        # NOTE: this takes long if radius is large
        for x, y in product(xrange(dx), xrange(dy)):
            if (x - macenter[0])**2 + (y - macenter[1])**2 <= self.radius_sq:
                mask[y][x] = True

        return mask

    def __str__(self):
        return 'CircRoi(center={}, radius={})'.format(self.center, self.radius)

    def __repr__(self):
        return "CircRoi({}, {}, '{}')".format(self.center, self.radius,
            self.description)

    def draw_outline(self):
        'draw circular ROI area on observed frame'
        frame = self.cap.latest_frame
        frame.draw_circle(self.center, self.radius, self.roicolor)

    def contains(self, point):
        'contains((x, y)) -> bool'
        x, y = point
        dx = x - self.center[0]
        dy = y - self.center[1]
        return dx**2 + dy**2 <= self.radius_sq

# set up specialized example ROIs that MeanCircRoi and
# MeanRectRoi that collect mean channel data
MeanBgrRecord = namedtuple('MeanBgrRecord', 'pos_frames pos_msec color')

def _collect_mean_bgr(instance, use_mask):
    '''helper for MeanCircRoi.collect and MeanRectRoi.collect -
    returns MeanBgrRecord or None'''
    rect = instance.get_rect()

    blue, green, red = [
        getattr(rect, color) for color in ('blue', 'green', 'red')]

    if use_mask:
        blue, green, red = [
            array[instance.mask] for array in (blue, green, red)]

    blue_avg, green_avg, red_avg = [
        float(np.mean(array)) for array in (blue, green, red)]

    col = Color(blue_avg, green_avg, red_avg)

    # test if col should be ignored
    if instance.collected:
        lastcol = instance.collected[-1].color
        if all(abs(x - y) < instance.delta_ig for x, y in izip(col, lastcol)):
            return None

    return MeanBgrRecord(instance.cap.pos_frames, instance.cap.pos_msec, col)

def _to_file_mean_bgr(instance, datafile, maskfile):
    'helper for MeanCircRoi.to_file and MeanRectRoi.to_file'
    if maskfile:
        np.save(maskfile, instance.mask)

    datafile.write(instance.info() + '\n')
    writer = csv.writer(datafile)
    writer.writerow(('pos_frames', 'pos_msec', 'blue_avg',
        'green_avg', 'red_avg'))
    datafile.write('HEADER_END\n')

    rows = ((fp, ms, b, g, r) for fp, ms, (b, g, r) in instance.collected)
    writer.writerows(rows)

class MeanCircRoi(CircRoi):
    '''circular region of interest that collects the mean blue, green and red
    intensities in the observed region for each frame'''

    def __init__(self, center, radius, description='N/A', delta_ig=0):
        '''MeanCircRoi((x, y), radius[, description[, delta_ig]])
        -> MeanCircRoi object

        description: optional description for ROI

        delta_ig: in order to conserve memory, ignore any observed mean
        color that differs less than delta_ig from the last collected color
        for each channel'''

        super(MeanCircRoi, self).__init__(center, radius, description)
        self.delta_ig = delta_ig

    def __repr__(self):
        return "MeanCircRoi({}, {}, '{}', {})".format(
            self.center, self.radius, self.description, self.delta_ig)

    def __str__(self):
        return 'MeanCircRoi(center={}, radius={}, delta_ig={})'.format(
            self.center, self.radius, self.delta_ig)

    def collect(self):
        'collect() -> MeanBgrRecord object or None'
        return _collect_mean_bgr(instance=self, use_mask=True)

    def to_file(self, datafile, maskfile=None):
        'write info and collected data to datafile, save mask to maskfile'
        _to_file_mean_bgr(instance=self, datafile=datafile, maskfile=maskfile)

class MeanRectRoi(RectRoi):
    '''rectangular region of interest that collects the mean blue, green and red
    intensities in the observed region for each frame'''

    def __init__(self, vertex1, vertex2, description='N/A', delta_ig=0):
        super(MeanRectRoi, self).__init__(vertex1, vertex2, description)
        self.delta_ig = delta_ig

    def __repr__(self):
        return "MeanRectRoi({}, {}, '{}', {})".format(
            self.vertex1, self.vertex2, self.description, self.delta_ig)

    def __str__(self):
        return 'MeanRectRoi(vertex1={}, vertex2={}, delta_ig={})'.format(
            self.vertex1, self.vertex2, self.delta_ig)

    def collect(self):
        'collect() -> MeanBgrRecord object or None'
        return _collect_mean_bgr(instance=self, use_mask=False)

    def to_file(self, datafile):
        'write info and collected data to datafile'
        _to_file_mean_bgr(instance=self, datafile=datafile, maskfile=None)
