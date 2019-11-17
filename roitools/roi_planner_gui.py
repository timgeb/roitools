'a GUI for planning regions of interest in a video'

# NOTE: naming conventions for global variables
# - Tkinter widgets/frames: preceded by tk_
# example: tk_export_button = ttk.Button(...)
# - Tkinter mutable variables: preceded by tkvar_
# example: tkvar_allow_all = tk.BooleanVar(...)
# - functions used as callbacks: preceded by cb_
# example: cb_select_video
# - variables holding styling information/parameters: preceded by st_
# example: st_button_border
# - important non-widget globals/constants used in lots of places: uppercase
# example: PLAYER = VideoPlayer(...)
# - single character unicode strings used as symbols: preceded by sym_
# example: sym_white_circle = u'\u25CB'

# TODO: refactoring -> RoiBox class for listboxes? Move ROI related helpers?

# TODO: (option to) show preview of ROI w.r.t current parameters
# on video-mouseover?

# TODO: coupling between the player, its canvas and its video is too tight
# - both player and canvas unidirectionally interact with the video
# - player and canvas interact bidirectionally
# (this is not optimal, but not hard to disentangle)

# standard library imports
from __future__ import division
from collections import OrderedDict
import csv
from datetime import timedelta
from functools import partial
import math
import os
import re
from ScrolledText import ScrolledText
from timeit import default_timer
import tkColorChooser
import tkFileDialog
import Tkinter as tk
import tkMessageBox
import ttk

# third party imports
import cv2
import numpy as np
from PIL import ImageTk, Image

# local application imports
import roitools
import vidtools

# --- GUI ROOT ---
# (must exist before creating StringVars and setting styles)
tk_root = tk.Tk()
# ----------------

# --- STYLE SET UP ---
st_font_name = 'Helvetica'
st_med_pt = 12
st_big_pt = 14
st_med_font = (st_font_name, st_med_pt)
st_big_font = (st_font_name, st_big_pt)
st_big_bold_font = (st_font_name, st_big_pt, 'bold')

# the global ttk style
st_global = ttk.Style(tk_root)
#st_global.theme_use('default')

# TODO: find nice theme?
# https://wiki.tcl-lang.org/page/List+of+ttk+Themes

# bordered label with big font
st_global.configure(
    'Bordered.TLabel', borderwidth=1, foreground='black',
    font=st_big_font, relief='solid')

# bordered button
st_button_border = dict(borderwidth=4, relief='raised')
st_global.configure(
    'Generic.TButton', font=st_med_font, **st_button_border)

# bordered button for single symbols (play, stop, ...)
st_global.configure(
    'SingleSymbol.TButton', font=st_big_bold_font,
    width=3, **st_button_border)
# --------------------

# --- CLASSES/FUNCTIONS FOR CREATING WIDGETS ---

# NOTE input validation has been implemented via tracing - another option
# is to use the validatecommand keyword argument of entry widgets

class TracedVar(tk.StringVar, object):
    'base class for a string variable with "live" value restriction'

    def __init__(self, allow_empty=True, **kwargs):
        super(TracedVar, self).__init__(**kwargs)
        self._allow_empty = allow_empty
        self.trace('w', self._validate_kill_args)
        self._last_value = kwargs.get('value', '')

    def check(self, value):
        '''returns True if value is OK, False otherwise
        (override this method in child classes)'''
        return True

    # for user
    def get(self):
        return self._last_value

    # for internal use
    def _get(self):
        return super(TracedVar, self).get()

    def _accept(self, value):
        self._last_value = value

    def _deny(self):
        self.set(self._last_value)

    def _validate_kill_args(self, *_):
        self._validate()

    def _validate(self):
        value = self._get()
        value_empty = not value

        if value_empty:
            if self._allow_empty:
                self._accept(value)
            else:
                self._deny()
        elif self.check(value):
            self._accept(value)
        else:
            self._deny()

class TracedMaxLen(TracedVar):
    'holds string with a maximum length'

    def __init__(self, max_len=np.inf, **kwargs):
        super(TracedMaxLen, self).__init__(**kwargs)
        self._max_len = max_len

    def check(self, value):
        return (
           super(TracedMaxLen, self).check(value) and
           len(value) <= self._max_len)

class TracedNumber(TracedMaxLen):
    'holds string representing a number'
    # NOTE: remember that the user must be able to enter strings that start
    # with . or - or -. (leading + is disallowed for simplicity)

    # strings that are treated as '0' when encountered on their own
    # (when the user calls get and inside the check method)
    _allowed_prefixes = {'.', '-', '-.'}

    def prefix_conv(self, value):
        # convert values such as '.' to '0' when encountered on their own
        return '0' if value in self._allowed_prefixes else value

    def get(self):
        return self.prefix_conv(super(TracedNumber, self).get())

    def check(self, value):
        success = False

        if super(TracedNumber, self).check(value):
            try:
                float(self.prefix_conv(value))
            except ValueError:
                pass
            else:
                # prevent useless leading zeros
                if not re.match('-?0\d', value):
                    success = True

        return success

class TracedMinMaxValue(TracedNumber):
    # TODO: snapping to min/max value would be nice (if allowed by max length)
    'holds string representing a number with a maximum and minimum value'

    def __init__(self, min_value=-np.inf, max_value=np.inf, **kwargs):
        super(TracedMinMaxValue, self).__init__(**kwargs)

        if min_value >= 0:
            self._allowed_prefixes = ('.',)

        self._min_value = min_value
        self._max_value = max_value

    def check(self, value):
        conv = self.prefix_conv(value)
        return (
            super(TracedMinMaxValue, self).check(value) and
            self._min_value <= float(conv) <= self._max_value)

class TracedInt(TracedMinMaxValue):
    'holds string representing an integer'

    def __init__(self, **kwargs):
        super(TracedInt, self).__init__(**kwargs)

        if self._min_value < 0:
            self._allowed_prefixes = ('-',)
        else:
            self._allowed_prefixes = tuple()

    def check(self, value):
        # NOTE: checking float(value).is_integer() would allow trailing
        # zeros after the decimal
        return (
            super(TracedInt, self).check(value) and
            self.prefix_conv(value).lstrip('+').lstrip('-').isdigit())

class DeleGetEntry(ttk.Entry, object):
    '''intended for use with traced variables:
    the entry delegates get calls to its textvariable and resets to
    latest acceptable value on losing focus'''
    # (in effect, this resets strings such as . or -. to '0' in this program)

    def __init__(self, master, textvariable, *args, **kwargs):
        self._textvariable = textvariable
        super(DeleGetEntry, self).__init__(
            master, *args, textvariable=textvariable, **kwargs)

        self.bind('<FocusOut>', self._reset)

    def get(self, *args, **kwargs):
        return(self._textvariable.get())

    def _reset(self, *_):
        value = self.get()
        self.delete(0, tk.END)
        self.insert(0, value)

    def configure(self, *args, **kwargs):
        self._textvariable = kwargs.get('textvariable', self._textvariable)
        return super(DeleGetEntry, self).configure(*args, **kwargs)

    def __setitem__(self, item, value):
        if item == 'textvariable':
            self._textvariable = value
        super(DeleGetEntry, self).__setitem__(item, value)

class Log(ScrolledText, object):
    'scrolled textbox for usage as log'

    def __init__(
        self, master=None,
        warning_tag_kws=dict(background='#FFFF00'), # yellow
        error_tag_kws=dict(background='#FF9EAB'), # rosy red
        success_tag_kws=dict(background='#9AFC58'), # light green
        **kwargs):

        if not ('background' in kwargs or 'bg' in kwargs):
            kwargs['bg'] = '#EFF0F1' # light gray

        super(Log, self).__init__(master, **kwargs)

        self['state'] = 'disabled'
        self.tag_config('warning', **warning_tag_kws)
        self.tag_config('error', **error_tag_kws)
        self.tag_config('success', **success_tag_kws)

    def _empty(self):
        return self.get('1.0', tk.END) == '\n'

    def append(self, text, tag=None, prefix=None):
        'appends text to the log'
        self['state'] = tk.NORMAL

        # start a new line if the log is not empty
        if not self._empty():
            self.insert(tk.END, '\n')

        self.insert(tk.END, '{}{}'.format(prefix or '', text), tag)
        self['state'] = tk.DISABLED
        self.see(tk.END)

    def warning(self, text, prefix='WARNING: '):
        self.append(text, 'warning', prefix)

    def error(self, text, prefix='ERROR: '):
        self.append(text, 'error', prefix)

    def success(self, text, prefix=None):
        self.append(text, 'success', prefix)

def make_scrolled_listbox(
    frame_master, header=None, vertical=True, horizontal=True,
    make_labelframe=False, **kws):
    '''set up scolled ListBox inside a fresh frame, return (frame, box)
    kws are passed through to Listbox constructor'''
    # TODO: (option to) hide scrollbars when not needed

    # set up box in frame
    boxrow = 0
    if make_labelframe:
        frame = ttk.Labelframe(frame_master, text=header or '')
    else:
        frame = ttk.Frame(frame_master)
        if header:
            ttk.Label(frame, text=header).grid(row=0, column=0)
            boxrow = 1

    box = tk.Listbox(frame, **(kws or {}))
    box.grid(row=boxrow, column=0)

    # set up scrollbars
    if vertical:
        vbar = ttk.Scrollbar(frame, command=box.yview, orient=tk.VERTICAL)
        box['yscrollcommand'] = vbar.set
        vbar.grid(row=boxrow, column=1, sticky=tk.NS)

    if horizontal:
        hbar = ttk.Scrollbar(frame, command=box.xview, orient=tk.HORIZONTAL)
        box['xscrollcommand'] = hbar.set
        hbar.grid(row=boxrow + 1, column=0, sticky=tk.EW)

    # let <Control-a> select everything
    box.bind('<Control-a>', lambda _: partial(box.selection_set, 0, tk.END)())

    return frame, box
# ------------------------------------------------------------------

# --- MISCELLANEOUS HELPERS ---
def bgr_to_hex(b, g, r):
    '''bgr color intensities to hex rgb notation
    example: bgr_to_hex(16, 32, 64) -> #402010'''
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def walk_widgets(root):
    'yield all descendants of a frame'
    for child in root.winfo_children():
        yield child
        if isinstance(child, (ttk.Frame, ttk.Labelframe)):
            for grandchild in walk_widgets(child):
                yield grandchild

def time_str(timespan, unit, rounding=None):
    'turns timespan in unit to human readable string'
    if rounding is not None:
        timespan = round(timespan, rounding)
    td = timedelta(**{unit:timespan})
    return re.sub('(\.\d+?)0+$', r'\1', str(td)) # strip unnecessary zeros

def is_avi(path):
    'checks first twelve bytes of file for valid avi header'
    with open(path, 'rb') as f:
        head = f.read(12)

    RIFF_signature = '\x52\x49\x46\x46'
    type_avi = '\x41\x56\x49\x20'

    return (
        len(head) == 12 and
        head[:4] == RIFF_signature and
        head[-4:] == type_avi)

def check_empty(value, description=None):
    'issues warning and raises ValueError if value is the empty string'
    if value == '':
        msg = 'missing {} value'.format(description or '')
        raise ValueError(msg)
# -----------------------------

# -- ROI RELATED HELPERS ---
class DummyCircRoi(roitools.CircRoi):
    '''dummy circular ROI that does not compute a mask on construction -
    mask computation can be expensive for large radius values and is
    not needed for planning ROIs'''

    def compute_mask(self):
        'does nothing'
        pass

def construct_roi_from_params(x, y):
    'construct a ROI with center (x, y) and w.r.t. current region parameters'
    rtype = tkvar_roitype.get()

    if rtype == RECTANGULAR:
        # get width and height
        width_str, height_str = tkvar_width.get(), tkvar_height.get()
        check_empty(width_str, 'region width')
        check_empty(height_str, 'region height')
        width, height = int(width_str), int(height_str)

        # construct vertices such that (x, y) is region center
        vertex1 = (
            x - width//2,
            y - height//2)
        vertex2 = (
            int(x + math.ceil(width/2.0)),
            int(y + math.ceil(height/2.0)))

        roi = roitools.RectRoi(vertex1, vertex2)

    elif rtype == CIRCULAR:
        radius_str = tkvar_radius.get()
        check_empty(radius_str, 'region radius')
        radius = int(radius_str)

        # check if this ROI can fit
        # TODO I really need to start supporting partial circular ROIs
        # in order to get rid of this mess
        in_bounds = not (
            x - radius < 0 or
            y - radius < 0 or
            x + radius >= PLAYER.video.frame_width or
            y + radius >= PLAYER.video.frame_height)

        if in_bounds:
            roi = DummyCircRoi((x, y), radius)
        else:
            msg = 'region out of frame - partial circles are not supported yet'
            raise ValueError(msg)

    return roi

def roi_info_line(roi):
    '''returns simplified region info for display in listbox
    the line always starts with the ROI id'''
    circ = isinstance(roi, roitools.CircRoi)
    symbol = sym_white_circle if circ else sym_ballot_box
    unit = 'seconds'
    rounding = 3

    start = time_str(roi.start_pos_msec/1000.0, unit, rounding)
    line = u'{}  {}  {}'.format(roi._id, symbol, start)

    if hasattr(roi, 'end_pos_msec'):
        end = time_str(roi.end_pos_msec/1000.0, unit, rounding)
        line += u' {} {}'.format(sym_arrow_right, end)

    return line

def roi_info_dict(roi):
    'creates ordered dict with information on roi'
    # keys of info dict
    keys = (
        'type', 'radius', 'center_x', 'center_y', 'vertex1_x', 'vertex1_y',
        'vertex2_x', 'vertex2_y', 'start_pos_msec', 'end_pos_msec',
        'start_pos_frames', 'end_pos_frames', 'constructor')

    # keys with identical ROI attribute names
    attrs = [
        'start_pos_msec', 'end_pos_msec',
        'start_pos_frames', 'end_pos_frames']

    info = OrderedDict.fromkeys(keys, '')
    type_ = CIRCULAR if isinstance(roi, roitools.CircRoi) else RECTANGULAR
    info['type'] = type_
    info['constructor'] = repr(roi)

    if type_ == CIRCULAR:
        attrs.append('radius')
        info['center_x'] = roi.center[0]
        info['center_y'] = roi.center[1]
    else:
        info['vertex1_x'] = roi.vertex1[0]
        info['vertex1_y'] = roi.vertex1[1]
        info['vertex2_x'] = roi.vertex2[0]
        info['vertex2_y'] = roi.vertex2[1]

    for attr in attrs:
        info[attr] = getattr(roi, attr, '')

    return info
# -----------------------------

# --- CLASSES FOR PLAYBACK ---
class RoiVideo(roitools.RoiCap):
    '''modified/extended RoiCap for use in this GUI
    (keeps track of a brightness offset and yields RGB frames)'''

    # NOTE: I'd like to keep this from initiating interactions with widgets,
    # if possible

    def __init__(self, *args, **kwargs):
        self._brightness_offset = 0
        super(RoiVideo, self).__init__(*args, **kwargs)

    def add_roi(self, roi):
        'register new ROI and add starting timestamps to ROI on success'
        super(RoiVideo, self).add_roi(roi)
        roi.start_pos_msec = self.pos_msec
        roi.start_pos_frames = self.pos_frames

    def _bgr_to_rgb(self, frame):
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)

    def draw_active_rois(self):
        for roi in self.rois.viewvalues():
            if not hasattr(roi, 'end_pos_frames'):
                roi.draw()

    def redraw(self):
        'apply brightness offset to current frame and redraw rois'
        # NOTE: the order is important here!

        # get copy of unmodified frame (BGR) as read by VideoCapture
        frame = self.latest_frame_original.copy()

        # apply brightness offset before drawing ROIs
        frame.adjust_brightness(self._brightness_offset)

        # redraw rois before converting from BGR to RGB
        self.latest_frame = frame
        self.draw_active_rois()

        # convert to RGB
        self._bgr_to_rgb(frame)

    @property
    def brightness_offset(self):
        return self._brightness_offset

    @brightness_offset.setter
    def brightness_offset(self, value):
        # NOTE: currently, ROIs are not notified again if the brightness of a
        # frame that has already been read changes - this has no effect for
        # CircRoi and RectRoi which don't collect data

        # when the offset is set, it needs to be applied to a copy of the
        # unmodified current frame, else information can be lost
        self._brightness_offset = np.clip(value, -255, 255)
        self.redraw()

    def next(self):
        # NOTE: ROIs are not notified (like in the parent class), only
        # redrawn, and only if active - this is fine for simple ROI planning
        # with dummy ROIs that don't collect any data
        # -> adjust code as needed if program is extended to collect ROI data

        self.latest_frame_original = vidtools.PyCap.next(self)
        self.redraw()

        return self.latest_frame

class VideoCaptureError(Exception):
    'custom exception for video loading problems'
    pass

class VideoPlayer(object):
    'class that mimics a video player'
    # NOTE: I would like the player to be decoupled from widgets as strictly
    # as possible, i.e. no initiation of interactions with widgets from here
    # (I could not reasonably prevent the canvas from doing it, though)

    # NOTE: methods intended to be called from outside (not preceded by _)
    # should perform actions that could reasonably be imagined to be found
    # on a video player
    # (it's OK to get attributes of the loaded video such as fps or number of
    # frames through the video instance variable, because these are not
    # properties of the player itself)

    def _re_init(self):
        'reinitialize'
        self.video = None # RoiVideo instance
        self.videoname = None # base name of current video
        self._paused = None # flag for pausing
        self._backwards = False # flag for backwards play
        self._target_delay = None # optimal delay between frames when playing
        self._sch_play_delay = None # current scheduling delay when playing
        self._last_play_show = None # last time of frame shown during playback
        self._after_handle = None # return value of self.tk_root.after
        self._canvas = None # canvas that shows image
        self._window = None # window that holds canvas

        self.pause()
        self._update_pos_vars()

    def __init__(self, tk_root=tk_root):
        self.tk_root = tk_root
        self._idle_delay = 50 # approximate delay between redraws when paused
        self.tkvar_pos_frames = tk.DoubleVar() # current frame
        self.tkvar_pos_time = tk.StringVar() # current time as string
        self._re_init()

    def _update_pos_vars(self):
        if self.video:
            frame = self.video.pos_frames
            msec = self.video.pos_msec
        else:
            frame = 1
            msec = 0

        self.tkvar_pos_frames.set(frame)
        self.tkvar_pos_time.set(time_str(msec/1000.0, 'seconds', rounding=0))

    def next(self):
        frame = next(self.video)
        self._update_pos_vars()
        return frame

    def set_position(self, frame):
        'set video to specific frame position and update display'
        # NOTE: the check whether the video has more than one frame should not
        # be necessary, but testing revealed that cv2.VideoCapture.read seems
        # to have a bug for single frame videos:
        # after reading the frame once and resetting the capture's position to
        # zero, subsequent calls to read fail to return the frame

        if self.video.frame_count > 1:
            self.video.pos_frames = frame
            self.video.pos_frames -= 1
            next(self)

    def refresh(self):
        self.video.redraw()

    def pause(self):
        self._paused = True

    def play(self):
        'resume playback in the current direction'
        if self._paused:
            now = default_timer()
            self._last_play_show = now
        self._paused = False

    def toggle_play(self):
        'pauses or resumes playback'
        if self._paused:
            self.play()
        else:
            self.pause()

    def play_backwards(self):
        self._backwards = True
        self.play()

    def play_forwards(self):
        self._backwards = False
        self.play()

    def stop(self):
        self.pause()
        self.set_position(0)

    def brightness_adjust(self, increment):
        self.video.brightness_offset += increment

    def brightness_reset(self):
        self.video.brightness_offset = 0

    def load_video(self, path):
        'load video from filename - expects to be called from unloaded state'
        # try to load the video
        try:
            video = RoiVideo(path)
            next(video)
        except (StopIteration, TypeError):
            # TODO: what else can be raised by cv2.VideoCapture(path)
            raise VideoCaptureError
        else:
            # missing critical attributes? (zero or None)
            criticals = (
                video.fps, video.frame_count,
                video.frame_width, video.frame_height)

            if not all(criticals):
                raise VideoCaptureError
            else:
                # video seems to be fine
                # TODO consider maximum delay for videos with very low fps
                self._target_delay = max(int(1000.0/video.fps), 1) # msec
                self._sch_play_delay = self._target_delay # initial value
                self.video = video
                self.videoname = os.path.basename(path)
                self._set_up_window()
                self._schedule_next_frame()

    def unload_video(self):
        'unload video - expects to be called from loaded state'
        self.tk_root.after_cancel(self._after_handle)
        self.video.release()
        self._window.destroy()
        self._re_init()

    def _set_up_window(self):
        'sets up the display window and canvas'
        no_border = dict(borderwidth=0, highlightthickness=0)

        # set up window
        self._window = tk.Toplevel(tk_root, **no_border)
        self._window.title(self.videoname)
        self._window.resizable(False, False)

        # set up canvas
        self._canvas = TouchScreen(
            player=self, master=self._window, height=self.video.frame_height,
            width=self.video.frame_width, **no_border)
        self._canvas.grid(row=0, column=0)

        # let close eject video
        self._window.protocol("WM_DELETE_WINDOW", cb_eject_video)

        # let space bar pause/resume
        self._window.bind('<space>', lambda _: self.toggle_play())

    def _manage_playback_scheduling(self):
        '''to be called directly after showing a frame during playback!
        performs actions to recalculate the current _sch_play_delay'''
        # NOTE: the scheduling mechanism is not very sophisticated and
        # constrained by the fact that Tkinter's after method only accepts
        # integer millisecond delays - that said, for lots of videos a
        # (forward) playback speed roughly equal to native speed is achieved
        # (which is fine for planning regions of interest - maybe use another
        # software to watch the latest blockbuster)

        now = default_timer()
        actual_delay = (now - self._last_play_show)*1000.0 # msec
        self._last_play_show = now

        adjusted = (self._target_delay/actual_delay)*self._sch_play_delay
        self._sch_play_delay = max(int(round(adjusted)), 1)

    def _schedule_next_frame(self):
        'pause or play the video by scheduling the next frame to show'
        delay = self._idle_delay # when unloaded or paused

        if self.video:
            # if playing, get the next frame and set
            # a roughly appropriate waiting time between frames
            if not self._paused:
                if self._backwards:
                    self.video.pos_frames -= 2 # clips to zero

                try:
                    next(self)
                except StopIteration:
                    self.pause()

                # reached beginning of video while playing backwards? -> stop
                if self._backwards and self.video.pos_frames == 1:
                    self.pause()

            # update canvas
            # (even when paused because ROIs/brightness may have changed)
            self._show_frame(self.video.latest_frame)

            # adjust delay (needs to be done after showing the frame!)
            # NOTE: time _manage_playback_scheduling takes itself is ignored
            if not self._paused:
                self._manage_playback_scheduling()
                delay = self._sch_play_delay

        # reschedule
        self._after_handle = self.tk_root.after(
            delay, self._schedule_next_frame)

    def _show_frame(self, frame):
        'convert frame to PhotoImage and put on canvas'
        image = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self._image = image # prevent garbage collection!
        self._canvas.create_image(0, 0, anchor=tk.NW, image=image)

class TouchScreen(tk.Canvas, object):
    '''a canvas for use with a VideoPlayer instance
    which supports mouse actions ("touching") to add ROIs
    
    right click -> add ROI with current region parameters
    left click + drag -> add rectangular ROI at selected area'''

    def _re_init(self):
        # keep track of the starting position of a mouse rectangle selection
        self._start_x = None
        self._start_y = None
    
        # delete last selection rectangle
        if getattr(self, '_rect', None) is not None:
            self.delete(self._rect)
        self._rect = None

    def __init__(self, player, master=None, **kwargs):
        super(TouchScreen, self).__init__(master, **kwargs)
        self._player = player
        self._re_init()

        # right click -> add ROI with current region parameters
        self.bind('<Button-3>', self._rclick_add_roi)

        # left click and drag -> add ROI at area of selected rectangle
        self.bind('<Button-1>', self._lclick)
        self.bind('<B1-Motion>', self._ldrag)
        self.bind('<ButtonRelease-1>', self._lrelease_add_roi)

    def _canvas_xy(self, event):
        'convert event x, y coordinates to canvas x, y coordinates as integers'
        x = self.canvasx(event.x)
        y = self.canvasx(event.y)
        return int(x), int(y)

    def _rclick_add_roi(self, event):
        'right click adds ROI w.r.t. current region parameters'
        x, y = self._canvas_xy(event)

        try:
            roi = construct_roi_from_params(x, y)
        except ValueError as e:
            tk_log.error(str(e))
        else:
            self._add_roi_helper(roi)

    def _lclick(self, event):
        'freeze the player and create a modifiable rectangle on the canvas'
        # make player leave the canvas alone
        tk_root.after_cancel(self._player._after_handle)

        # create rectangle
        x, y = self._start_x, self._start_y = self._canvas_xy(event)
        color = bgr_to_hex(*roitools.BaseRoi.active_color)
        self._rect = self.create_rectangle(x, y, x, y, outline=color)

    def _ldrag(self, event):
        'expand the current rectangle'
        x, y = self._canvas_xy(event)
        self.coords(self._rect, self._start_x, self._start_y, x, y)

    def _lrelease_add_roi(self, event):
        'add ROI at selected rectangle, unfreeze player'
        x, y = self._canvas_xy(event)

        # add ROI
        vertex1 = (self._start_x, self._start_y)
        vertex2 = (x, y)
        roi = roitools.RectRoi(vertex1, vertex2)
        self._add_roi_helper(roi)

        # cleanup
        self._re_init()

        # unfreeze player
        self._player._schedule_next_frame()

    def _add_roi_helper(self, roi):
        'helper for adding roi to associated player after mouse action'
        # add ROI to video
        try:
            self._player.video.add_roi(roi)
        except ValueError as e:
            # happens when maximum number of regions is exceeded
            tk_log.error(str(e))
            roitools.BaseRoi._next_id -= 1
        else:
            self._player.refresh()
            tk_active_roi_box.insert(tk.END, roi_info_line(roi))
            tk_log.append('started region {}'.format(roi._id))
# ----------------------------

# --- GLOBALS/CONSTANTS ----
# (no widgets)
PLAYER = VideoPlayer()
FRAMES = 'frames'
SECONDS = 'seconds'
CIRCULAR = 'circular'
RECTANGULAR = 'rectangular'
ttk_DISABLED = 'disabled'
ttk_NORMAL = '!{}'.format(ttk_DISABLED)
# --------------------------

# --- SYMBOLS ---
sym_eject = u'\u23CF'
sym_triangle_right = u'\u23F5'
sym_triangle_left = u'\u23F4'
sym_double_vbar = u'\u23F8'
sym_black_square = u'\u23F9'
sym_white_circle = u'\u25CB'
sym_ballot_box = u'\u2610'
sym_double_triangle_left = u'\u23EA'
sym_double_triangle_right = u'\u23E9'
sym_arrow_up = u'\u2B06'
sym_arrow_down = u'\u2B07'
sym_arrow_right = u'\u2192'
sym_stroked_o = u'\u00D8'
# ---------------

# --- WIDGET CALLBACKS ---
# (and helpers)

# INTERACTION WITH MAIN WINDOW
lose_unsaved_warning = 'All unsaved regions will be lost.'

def cb_confirm_quit():
    'if a video is loaded, ask before closing the program'
    msg = 'Exit program?\n{}'.format(lose_unsaved_warning)
    if not PLAYER.video or tkMessageBox.askyesno('Exit?', msg):
        tk_root.destroy()

# VIDEO EJECTION
def cb_eject_video(confirm=True):
    'unloads video through PLAYER and updates GUI'
    if confirm:
        msg = 'Eject video?\n{}'.format(lose_unsaved_warning)
        if not tkMessageBox.askyesno('Eject?', msg):
            return

    # clear listboxes
    for box in (tk_active_roi_box, tk_finished_roi_box):
        box.delete(0, tk.END)

    set_widget_states_unloaded()

    videoname = PLAYER.videoname
    PLAYER.unload_video()
    tkvar_videoname.set('')
    tk_log.append('ejected "{}"'.format(os.path.basename(videoname)))

# VIDEO SELECTION
def cb_select_video():
    'ejects previous video and loads video through PLAYER, then updates GUI'

    # get file from user
    path = tkFileDialog.askopenfilename()
    basename = os.path.basename(path) if path else ''
    if not basename:
        return

    if PLAYER.video:
        cb_eject_video(confirm=False)

    # currently, we want avi files for best OpenCV support, see
    # https://stackoverflow.com/a/20865122
    if not (tkvar_allow_all.get() or is_avi(path)):
        msg = '"{}" not an avi - currently only avi files are supported'
        msg = msg.format(basename)
        tk_log.error(msg)
        return

    # try to load video, set up widgets, log
    try:
        PLAYER.load_video(path)
    except VideoCaptureError:
        msg = 'can\'t process "{}" as video (corrupt file?)'.format(basename)
        tk_log.error(msg)
    else:
        roitools.BaseRoi._next_id = 1
        tk_bar['to'] = PLAYER.video.frame_count
        set_widget_states_loaded()

        # set video name label
        width = tk_videoname_label['width']
        label_text = (basename if len(basename) <= width else
                      basename[:max(width - 3, 0)] + '...')
        tkvar_videoname.set(label_text)

        # log
        num_frames = int(PLAYER.video.frame_count)
        msg = 'loaded "{}" ({} fps, {} frame{})'.format(
            basename, PLAYER.video.fps, num_frames, 's'*(num_frames > 1))
        tk_log.success(msg)

        # TODO MANUAL
        manual = (
            'ADD A ROI by drawing a rectangle (left mouse)\n'
            'regions with fixed parameters are added with right click')
        tk_log.append(manual)

# VIDEO PROGRESS
def cb_bar_set_frame(event):
    'sets video position according to clicked position on progress bar'
    # NOTE: the problem here is that BETWEEN the mousebutton-release-event
    # and the execution of this function, the bar can be re-set by reading
    # from PLAYER.tkvar_pos_frames - which means calling tk_bar.get will
    # not return the position the bar was just clicked to
    #
    # solution:
    # 1. the bar is disabled by default
    # (events are still registered, this function is still executed)
    # 2. in this callback: enable bar
    # 3. generate click event with same x, y coordinates -> bar moves
    # 4. disable bar again
    # 5. NOW get the bar's position and set the video frame

    if PLAYER.video:
        tk_bar.state([ttk_NORMAL])
        tk_bar.event_generate('<Button-1>', x=event.x, y=event.y)
        tk_bar.state([ttk_DISABLED])

        frame = round(tk_bar.get())
        PLAYER.set_position(frame)

# CLASSICAL CONTROLS (play, pause, stop)
cb_play_forwards = PLAYER.play_forwards
cb_play_backwards = PLAYER.play_backwards
cb_pause = PLAYER.pause
cb_stop = PLAYER.stop

# JUMPING
def secs_to_frames(seconds):
    '''helper for jumping
    converts seconds to nearest possible value in frames and logs adjustment'''
    frames = seconds*PLAYER.video.fps
    frames_adjusted = int(round(frames))

    # issue warning if adjusting changed input value
    if frames != frames_adjusted:
        secs_adjusted = frames_adjusted/PLAYER.video.fps
        template = 'adjusted jump value to {} {} ({} {})'
        insert = (secs_adjusted, SECONDS, frames_adjusted, FRAMES)
        tk_log.warning(template.format(*insert))

    return frames_adjusted

def cb_jump(back=False):
    'changes video position according to jump step field'
    step = tk_step_entry.get()
    unit = tkvar_unit.get()

    # empty?
    try:
        check_empty(step, description=unit)
    except ValueError as e:
        tk_log.error(str(e))

    # get step in frames
    if unit == SECONDS:
        jump_frames = secs_to_frames(float(step))
    else:
        jump_frames = int(step)

    if back:
        jump_frames *= -1

    # work to do?
    if jump_frames == 0:
        return

    # perform the jump (always in frames for exactness)
    before_pos_frames = PLAYER.video.pos_frames
    before_pos_msec = PLAYER.video.pos_msec

    PLAYER.set_position(before_pos_frames + jump_frames)

    # log jump
    delta_frames = int(PLAYER.video.pos_frames - before_pos_frames)
    delta_msec = PLAYER.video.pos_msec - before_pos_msec
    delta_sec = delta_msec/1000.0

    template = 'jumped {} {} ({} {})'
    if unit == SECONDS:
        insert = (delta_sec, SECONDS, delta_frames, FRAMES)
    else:
        insert = (delta_frames, FRAMES, delta_sec, SECONDS)

    msg = template.format(*insert)
    tk_log.append(msg)

    # issue warning if target not met
    # NOTE: if the target has not been met AND we're not at the beginning
    # or end of the video, that's a bug! :)
    if abs(delta_frames) < abs(jump_frames):
        if back:
            where = 'beginning'
        else:
            where = 'end'
        tk_log.warning('reached {} of video'.format(where))

cb_jump_forward = partial(cb_jump, back=False)
cb_jump_back = partial(cb_jump, back=True)

# BRIGHTNESS

# the brightness increment value should give the user reasonably fine grained
# control over the video brightness without larger adjustments being tiresome
# - ten seems to be a good value
brightness_increment = 10
cb_brightness_up = partial(PLAYER.brightness_adjust, brightness_increment)
cb_brightness_down = partial(PLAYER.brightness_adjust, -brightness_increment)
cb_brightness_reset = PLAYER.brightness_reset

# ROI DELETION
def selected_ids(listbox):
    # NOTE: parsing ROI ids from the beginning of listbox strings is straight
    # forward but mixes presentation and model -> on next refactoring consider
    # Listbox subclass internally keeping track of ROIs being held
    'returns list of ROI ids corresponding to selected listbox items'
    lines = (listbox.get(i) for i in listbox.curselection())
    return [int(line.split(' ', 1)[0]) for line in lines]

def del_rois(listbox):
    'delete selected ROIs from listbox'
    ids = selected_ids(listbox)

    # no selection but box not empty?
    if not ids and listbox.get(0, tk.END):
        msg = 'No regions selected. Delete all?'
        if tkMessageBox.askyesno('Delete all?', msg):
            listbox.selection_set(0, tk.END)
            ids = selected_ids(listbox)

    # no work -> abort
    if not ids:
        return

    # delete selected lines in listbox
    for line_index in reversed(listbox.curselection()):
        listbox.delete(line_index)

    # delete from capture
    for id_ in ids:
        del PLAYER.video.rois[id_]

    # log
    msg = 'deleted region{} {}'.format(
        's'*(len(ids) > 1), ', '.join(map(str, ids)))
    tk_log.append(msg)

def cb_del_active_rois():
    del_rois(tk_active_roi_box)
    PLAYER.refresh()

def cb_del_finished_rois():
    del_rois(tk_finished_roi_box)

# ROI FINALIZATION
def cb_finish_rois():
    'move selected active ROIs to finished ROIs'
    # TODO: structure in parts similar to cb_delete_rois function -> refactor?
    # (also, this function got pretty long)
    ids = selected_ids(tk_active_roi_box)

    # no selection but box not empty? -> ask to finish all
    if not ids and tk_active_roi_box.get(0, tk.END):
        msg = 'No regions selected. Finish all?'
        if tkMessageBox.askyesno('Finish all?', msg):
            tk_active_roi_box.selection_set(0, tk.END)
            ids = selected_ids(tk_active_roi_box)

    # no work -> abort
    if not ids:
        return

    # delete selected lines in active roi listbox
    for i in reversed(tk_active_roi_box.curselection()):
        tk_active_roi_box.delete(i)

    # mark ROIs as finished by giving them ending timestamps
    end_msec = PLAYER.video.pos_msec
    end_frame = PLAYER.video.pos_frames
    swap = tkvar_swap_times.get()

    for id_ in ids:
        roi = PLAYER.video.rois[id_]
        roi.end_pos_msec = end_msec
        roi.end_pos_frames = end_frame

        # switch start/end if option set
        if swap and roi.end_pos_frames < roi.start_pos_frames:
            roi.end_pos_msec, roi.start_pos_msec = (
                roi.start_pos_msec, roi.end_pos_msec)
            roi.end_pos_frames, roi.start_pos_frames = (
                roi.start_pos_frames, roi.end_pos_frames)

    # insert lines into finished roi box
    for id_ in ids:
        roi = PLAYER.video.rois[id_]
        tk_finished_roi_box.insert(tk.END, roi_info_line(roi))

    # log
    msg = 'finished region{} {}'.format(
        's'*(len(ids) > 1), ', '.join(map(str, ids)))
    tk_log.append(msg)

    # redraw frame
    PLAYER.refresh()

# ROI EXPORTING
def cb_export():
    'export finished ROIs'
    ids = selected_ids(tk_finished_roi_box)

    # no selection but box not empty? -> ask to finish all
    if not ids and tk_finished_roi_box.get(0, tk.END):
        msg = 'No regions selected. Export all?'
        if tkMessageBox.askyesno('Export all?', msg):
            tk_finished_roi_box.selection_set(0, tk.END)
            ids = selected_ids(tk_finished_roi_box)

    # no work -> abort
    if not ids:
        return

    # ask the user for a filename
    ext = '.csv'
    filename = tkFileDialog.asksaveasfilename(
        title='Export region information', defaultextension=ext,
        initialfile='regions_export{}'.format(ext),
        filetypes=(('spreadsheet', '*{}'.format(ext)),))

    if not filename:
        return

    # build lines for csv file
    row_dicts = [roi_info_dict(PLAYER.video.rois[id_]) for id_ in ids]

    # write csv file
    try:
        with open(filename, 'w') as out:
            writer = csv.DictWriter(out, fieldnames=row_dicts[0].keys())
            writer.writeheader()
            writer.writerows(row_dicts)
    except EnvironmentError as e:
        tk_log.error(str(e))
    else:
        # log
        msg = 'exported region{} {}'.format(
            's'*(len(ids) > 1), ', '.join(map(str, ids)))
        tk_log.success(msg)

        # delete if option set
        if tkvar_del_exported.get():
            del_rois(tk_finished_roi_box)

# COLOR CHOOSING
def cb_select_color():
    'set display color for all regions'
    current_bgr = roitools.BaseRoi.active_color
    current_rgb = current_bgr[::-1]

    new_rgb, _ = tkColorChooser.askcolor(current_rgb)

    if new_rgb:
        new_bgr = vidtools.Color(*new_rgb[::-1])
        roitools.BaseRoi.active_color = new_bgr

        if PLAYER.video:
            PLAYER.refresh()
# ------------------------

# --- WIDGET CREATION, PLACEMENT, CALLBACK BINDING ---
# (from top to bottom, left to right)

tk_root.title('ROI Planner')
tk_root.protocol('WM_DELETE_WINDOW', cb_confirm_quit)

# parent frame for all widgets
tk_root_frame = ttk.Frame(tk_root)
tk_root_frame.grid(row=0, column=0)

# SET UP MENU
tk_menubar = tk.Menu(tk_root)
tk_options_menu = tk.Menu(tk_menubar)
tk_menubar.add_cascade(label='Options', menu=tk_options_menu)
tk_root['menu'] = tk_menubar

# option to globally switch ROI color
tk_options_menu.add_command(
    label='set region color...', command=cb_select_color)

# separator before checkbuttons
tk_options_menu.add_separator()

# option to automatically delete exported videos
tkvar_del_exported = tk.BooleanVar(value=True)
tk_options_menu.add_checkbutton(
    label='delete exported regions',
    offvalue=False, onvalue=True, variable=tkvar_del_exported)

# option to switch start and end times for rois with start > end
tkvar_swap_times = tk.BooleanVar(value=False)
label = (
    'swap timestamps when regions finish '
    'with end time earlier than start time')
tk_options_menu.add_checkbutton(
    label=label, offvalue=False, onvalue=True, variable=tkvar_swap_times)

# option to allow non-avi videos
tkvar_allow_all = tk.BooleanVar(value=False)
tk_options_menu.add_checkbutton(
    label='allow non-avi files (unstable)',
    offvalue=False, onvalue=True, variable=tkvar_allow_all)

# SET UP FILE PICKER
tk_select_frame = ttk.Frame(tk_root_frame)
tk_select_frame.grid(row=0, column=0)
tkvar_videoname = tk.StringVar()

# select video button
tk_select_button = ttk.Button(
    tk_select_frame, text='select video',
    style='Generic.TButton', command=cb_select_video)
tk_select_button.grid(row=0, column=0)

# video name label
# TODO: make label a scrolled readonly entry, hide the
# scrollbar whenever the entry fits
tk_videoname_label = ttk.Label(
    tk_select_frame, width=57,
    textvariable=tkvar_videoname, style='Bordered.TLabel')
tk_videoname_label.bind('<Button-1>', lambda *_: cb_select_video())
tk_videoname_label.grid(row=0, column=2)

# eject video button
tk_eject_button = ttk.Button(
    tk_select_frame, text=sym_eject, style='SingleSymbol.TButton',
    command=cb_eject_video)
tk_eject_button.grid(row=0, column=1)

# separator under file picker
ttk.Separator(tk_root_frame, orient=tk.HORIZONTAL).grid(row=1, column=0)

# SET UP PROGRESS INFO
tk_progress_frame = ttk.Frame(tk_root_frame)
tk_progress_frame.grid(row=2, column=0)

# progress bar
tk_bar = ttk.Scale(
    tk_progress_frame, orient=tk.HORIZONTAL, from_=1,
    takefocus=False, variable=PLAYER.tkvar_pos_frames)

# jump bar with left click in addition to right click
# -> make bar think a '<Button-1>' event is a '<Button-3>' event
tk_bar.bind(
    '<Button-1>',
    lambda event: tk_bar.event_generate('<Button-3>', x=event.x, y=event.y))

# jump to clicked position at mousebutton-release
for button in range(1, 6):
    event = '<ButtonRelease-{}>'.format(button)
    tk_bar.bind(event, cb_bar_set_frame)

tk_bar.grid(row=0, column=0)

# time label next to progress bar
tk_timelabel = ttk.Label(tk_progress_frame, textvariable=PLAYER.tkvar_pos_time)
tk_timelabel.grid(row=0, column=1)

# SET UP PLAYER CONTROLS
tk_control_frame = ttk.Frame(tk_root_frame)
tk_control_frame.grid(row=3, column=0)

tk_pauseplay_frame = ttk.Labelframe(tk_control_frame, text='player controls')
tk_pauseplay_frame.grid(row=1, column=0)

# play button
tk_play_button = ttk.Button(
    tk_pauseplay_frame, text=sym_triangle_right,
    style='SingleSymbol.TButton', command=cb_play_forwards)
tk_play_button.grid(row=0, column=0)

# play backwards button
tk_play_backwards_button = ttk.Button(
    tk_pauseplay_frame, text=sym_triangle_left,
    style='SingleSymbol.TButton', command=cb_play_backwards)
tk_play_backwards_button.grid(row=0, column=1)

# pause button
tk_pause_button = ttk.Button(
    tk_pauseplay_frame, text=sym_double_vbar,
    style='SingleSymbol.TButton', command=cb_pause)
tk_pause_button.grid(row=0, column=2)

# stop button
tk_stop_button = ttk.Button(
    tk_pauseplay_frame, text=sym_black_square,
    style='SingleSymbol.TButton', command=cb_stop)
tk_stop_button.grid(row=0, column=3)

# SET UP JUMP CONTROLS
tk_jump_frame = ttk.Labelframe(tk_control_frame, text='jump controls')
tk_jump_frame.grid(row=1, column=1)

# jump back button
tk_jump_back_button = ttk.Button(
    tk_jump_frame, text=sym_double_triangle_left,
    style='SingleSymbol.TButton', command=cb_jump_back)
tk_jump_back_button.grid(row=0, column=0)

# jump step entry
val_restrictions = dict(min_value=-1000, max_value=99999, max_len=10)
tk_timeentry_frame = ttk.Frame(tk_jump_frame)
tk_timeentry_frame.grid(row=0, column=1)
tkvar_seconds = TracedMinMaxValue(**val_restrictions)

tkvar_frames = TracedInt(**val_restrictions)
tk_step_entry = DeleGetEntry(
    tk_timeentry_frame, width=7, justify=tk.CENTER, textvariable=None)
tk_step_entry.grid(row=0, column=0)

# jump step unit label
tkvar_unit = tk.StringVar()
tk_unit_label = ttk.Label(
    tk_timeentry_frame, anchor=tk.CENTER, textvariable=tkvar_unit)
tk_unit_label.grid(row=1, column=0)

# jump forward button
tk_jump_forward_button = ttk.Button(
    tk_jump_frame, text=sym_double_triangle_right,
    style='SingleSymbol.TButton', command=cb_jump_forward)
tk_jump_forward_button.grid(row=0, column=2)

# group of radiobuttons to select unit for jump step (seconds or frames)
# select seconds: tie jump entry to TracedMinMaxValue (tkvar_seconds)
# select frames: tie jump entry to TracedInt (tkvar_frames)
tk_selectunit_frame = ttk.Frame(tk_jump_frame)
tk_selectunit_frame.grid(row=0, column=3)
tk_seconds_radiobutton = ttk.Radiobutton(
    tk_selectunit_frame, text=SECONDS, value=SECONDS, variable=tkvar_unit,
    command=partial(tk_step_entry.configure, textvariable=tkvar_seconds))
tk_seconds_radiobutton.invoke() # set initial selection, trigger command
tk_seconds_radiobutton.grid(row=0, column=0)

tk_frames_radiobutton = ttk.Radiobutton(
    tk_selectunit_frame, text=FRAMES, value=FRAMES, variable=tkvar_unit,
    command=partial(tk_step_entry.configure, textvariable=tkvar_frames))
tk_frames_radiobutton.grid(row=1, column=0)

# SET UP BRIGHTNESS CONTROLS
tk_bright_frame = ttk.Labelframe(
    tk_control_frame, text='brightness controls')
tk_bright_frame.grid(row=1, column=2)

# brightness up button
tk_brightup_button = ttk.Button(
    tk_bright_frame, text=sym_arrow_up,
    style='SingleSymbol.TButton', command=cb_brightness_up)
tk_brightup_button.grid(row=0, column=0)

# brightness down button
tk_brightdown_button = ttk.Button(
    tk_bright_frame, text=sym_arrow_down,
    style='SingleSymbol.TButton', command=cb_brightness_down)
tk_brightdown_button.grid(row=0, column=1)

# brightness reset button
tk_brightreset_button = ttk.Button(
    tk_bright_frame, text=sym_stroked_o,
    style='SingleSymbol.TButton', command=cb_brightness_reset)
tk_brightreset_button.grid(row=0, column=2)

# separator under controls
ttk.Separator(tk_root_frame, orient=tk.HORIZONTAL).grid(row=4, column=0)

# SET UP FRAMES FOR CONTROLING ACTIVE AND FINISHED ROIS
tk_roi_frame = ttk.Frame(tk_root_frame)
tk_roi_frame.grid(row=6, column=0)

# frame and box for active ROIs
box_kws = dict(
    frame_master=tk_roi_frame, make_labelframe=True, horizontal=False,
    selectmode=tk.MULTIPLE, exportselection=False, height=8)
tk_active_roi_frame, tk_active_roi_box = make_scrolled_listbox(
    header='active regions - start', **box_kws)
tk_active_roi_box['width'] = 25
tk_active_roi_frame.grid(row=0, column=0)

# frame and box for finished ROIs
tk_finished_roi_frame, tk_finished_roi_box = make_scrolled_listbox(
    header='finished regions - lifetime', **box_kws)
tk_finished_roi_box['width'] = 35
tk_finished_roi_frame.grid(row=0, column=1)

# buttons for roi deletion
del_text = 'delete'
del_button_kws = dict(
    text=del_text, width=len(del_text), style='Generic.TButton')
del_grid_kws = dict(row=3, column=0)

# delete active ROIs button
tk_del_active_button = ttk.Button(
    tk_active_roi_frame, command=cb_del_active_rois, **del_button_kws)
tk_del_active_button.grid(**del_grid_kws)

# delete finished ROIs button
tk_del_finished_button = ttk.Button(
    tk_finished_roi_frame, command=cb_del_finished_rois, **del_button_kws)
tk_del_finished_button.grid(**del_grid_kws)

# finish active ROIs button
fin_text = 'finish'
tk_finish_button = ttk.Button(
    tk_active_roi_frame, text=fin_text, width=len(fin_text),
    style='Generic.TButton', command=cb_finish_rois)
tk_finish_button.grid(row=3, column=0)

# export finished ROIs button
exp_text = 'export'
tk_export_button = ttk.Button(
    tk_finished_roi_frame, text=exp_text, width=len(exp_text),
    command=cb_export, style='Generic.TButton')
tk_export_button.grid(row=3, column=0)

# SET UP ROI PARAMETER SELECTION FRAME
# TODO: section might need refactoring
tk_roi_select_frame = ttk.Labelframe(tk_roi_frame, text='region parameters')
tk_roi_select_frame.grid(row=0, column=2)

tkvar_roitype = tk.StringVar(value=RECTANGULAR)

roi_entry_params = dict(master=tk_roi_select_frame, width=5, justify=tk.CENTER)
tkvar_width = TracedInt(**val_restrictions)
tkvar_height = TracedInt(**val_restrictions)

tkvar_radius = TracedInt(**val_restrictions)
tk_width_entry, tk_height_entry, tk_radius_entry = [
    DeleGetEntry(textvariable=var, **roi_entry_params)
    for var in (tkvar_width, tkvar_height, tkvar_radius)]

tk_rectangular_radiobutton = ttk.Radiobutton(
    tk_roi_select_frame, value=RECTANGULAR, variable=tkvar_roitype)
tk_rectangular_radiobutton.grid(row=0, column=0)
ttk.Label(tk_roi_select_frame, text=RECTANGULAR).grid(
    row=0, column=1, columnspan=2)

tk_width_entry.grid(row=1, column=1)
ttk.Label(tk_roi_select_frame, text='px width').grid(row=1, column=2)

tk_height_entry.grid(row=3, column=1)
ttk.Label(tk_roi_select_frame, text='px height').grid(row=3, column=2)

tk_circular_radiobutton = ttk.Radiobutton(
    tk_roi_select_frame, value=CIRCULAR, variable=tkvar_roitype)
tk_circular_radiobutton.grid(row=4, column=0)
tk_circular_label = ttk.Label(tk_roi_select_frame, text=CIRCULAR)
tk_circular_label.grid(row=4, column=1, columnspan=2)
tk_radius_entry.grid(row=5, column=1)
ttk.Label(tk_roi_select_frame, text='px radius').grid(row=5, column=2)

# SET UP LOG
tk_log = Log(tk_root_frame, height=10)
tk_log.grid(row=7, column=0)
# ----------------------------------------------------

# --- GRID CONFIGURATON ---

# NOTE getting all widgets here assumes they are static - if widgets are added
# or removed at runtime, walk_widgets needs to be called at runtime as needed
ALL_WIDGETS = set(walk_widgets(tk_root_frame))
ALL_WIDGETS.add(tk_root_frame)

# this is a (crude?) implementation of cascading grid configuration:
# the three dictionaries map widget classes, styles and individual widgets to
# parameter dictionaries - each widget in the GUI is configured with the most
# specific parameters that are available

# parameters will likely need adjustment if the GUI changes!

# map widget class to grid parameters
class_grid_params = {
    ttk.Frame: dict(sticky=tk.EW),
    ttk.Label: dict(sticky=tk.W),
    ttk.Separator: dict(pady=20, sticky=tk.EW),
    ttk.Scale: dict(sticky=tk.EW),
    ttk.Radiobutton: dict(sticky=tk.W),
}

# map widget style to grid parameters
style_grid_params = {
    'Generic.TButton': dict(pady=(5, 0)),
    'SingleSymbol.TButton': dict(padx=1),
}

# map individual widgets to grid parameters
widget_grid_params = {
    tk_root_frame: dict(padx=10, pady=10),
    tk_select_button: dict(sticky=tk.NS, padx=(0, 1), pady=0),
    tk_videoname_label: dict(sticky=(tk.W, tk.NS), padx=(10, 0)),
    tk_progress_frame: dict(pady=(0, 15)),
    tk_timeentry_frame: dict(padx=5),
    tk_unit_label: dict(sticky=tk.EW),
    tk_selectunit_frame: dict(padx=(5, 0)),
    tk_roi_frame: dict(pady=(0, 15)),
    tk_bright_frame: dict(sticky=tk.E),
    tk_del_active_button: dict(sticky=tk.W),
    tk_finish_button: dict(sticky=tk.E),
    tk_log.frame: dict(sticky=tk.EW),
    tk_roi_select_frame: dict(sticky=tk.N),
    tk_circular_radiobutton: dict(pady=(5,0)),
}

widget_grid_params[tk_export_button] = (
    widget_grid_params[tk_finish_button])

widget_grid_params[tk_del_finished_button] = (
    widget_grid_params[tk_del_active_button])

widget_grid_params[tk_circular_label] = (
    widget_grid_params[tk_circular_radiobutton])

# apply grid parameters
for widget in ALL_WIDGETS:
    try:
        style = widget['style']
    except tk.TclError:
        style = None

    kwargs = class_grid_params.get(widget.__class__, {}).copy()
    kwargs.update(style_grid_params.get(style, {}))
    kwargs.update(widget_grid_params.get(widget, {}))

    widget.grid_configure(**kwargs)

# additional row/column configuration
tk_progress_frame.columnconfigure(0, weight=1) # expand progress bar
tk_control_frame.columnconfigure(1, pad=40)
tk_roi_frame.columnconfigure(1, pad=60)

# pad all Labelframes
# NOTE: can't use ipadx/ipady through Labelframe.configure because internal
# padding is applied to frame before widgets are placed
for widget in ALL_WIDGETS:
    if isinstance(widget, ttk.Labelframe):
        widget['padding'] = 5

# pad left/right of video label
tk_videoname_label['padding'] = (10, 0)

# miscellaneous
tk_root.resizable(False, False)
# -------------------------

# --- ENABLING/DISABLING WIDGETS ---
def set_widget_states(widgets, state=tk.NORMAL):
    '''given an iterable of widgets, tries to set state to tk.NORMAL or
    tk.DISABLED and suppresses errors if that's not possible'''
    for widget in widgets:
        try:
            widget['state'] = state
        except tk.TclError:
            # deal with ttk widgets
            ttk_state = ttk_NORMAL if state == tk.NORMAL else ttk_DISABLED

            try:
                widget.state([ttk_state])
            except (AttributeError, tk.TclError):
                pass

# function for state: no video
set_widget_states_unloaded = partial(
    set_widget_states, ALL_WIDGETS - {tk_select_button}, tk.DISABLED)

# function for state: video loaded
# NOTE disabling the bar is a kludge to make setting the video position work
# properly while playing - see cb_bar_set_frame for details
set_widget_states_loaded = partial(
    set_widget_states, ALL_WIDGETS - {tk_bar}, tk.NORMAL)
# ----------------------------------

# --- TEST SECTION ---
if __name__ == '__main__':
    set_widget_states_unloaded()
    tk_root.mainloop()
