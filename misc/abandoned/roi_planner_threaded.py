# --- ABANDONED ---
# threaded version of the ROI planner GUI - abandoned when I realized
# how to do this properly without multiple threads

# NOTE: naming conventions for *global* variables
# - Tkinter widgets/frames: preceded by tk_
# example: tk_export_button = ttk.Button(...)
# - Tkinter mutable variables: preceded by tkvar_
# example: tkvar_allow_all = tk.BooleanVar(...)
# - functions used as callbacks: preceded by cb_, functions run via the
# after_idle/after methods are preceded by cb_after_
# examples: cb_pause, cb_after_update_bar
# - variables holding styling information/parameters: preceded by st_
# example: st_warning_style
# - important non-widget globals/constants used in lots of places: uppercase
# example: RUNNER = CapRunner(...)
# - single character unicode strings used as symbols: preceded by sym_
# example: sym_white_circle = u'\u25CB'

# NOTE: locks have been applied pretty generously with the locked decorator,
# more fine grained locking is certainly possible

# NOTE: race conditions (and/or AttributeErrors due to RUNNER.cap being None)
# can be eliminated by locking callbacks or disabling interface elements right
# before they become unnecessary
# (e.g. brightness buttons when no video is loaded)

# TODO: with correctly adjusted frame_delay_msec the video can play at
# (close to) correct speed!

from __future__ import division, print_function

from collections import OrderedDict
import csv
from datetime import timedelta
from functools import partial, wraps
import os.path
import re
from ScrolledText import ScrolledText
import time
import threading
import Tkinter as tk
import tkFileDialog
import tkMessageBox
import ttk

import cv2
import numpy as np
import mttkinter # make tkinter thread safe

import roitools
import vidtools

# --- DEBUG HELPERS ---
def whichthread(say=''):
    t = threading.current_thread()
    print('{}{}'.format(say, t))
# ---------------------

# --- GUI ROOT ---
# (must exist before creating StringVars and setting styles)
tk_root = tk.Tk()
# ----------------

# --- STYLE OPTIONS ---
# (fonts, colors, etc.)
st_font_name = 'Helvetica'
st_med_pt = 12
st_big_pt = 14
st_med_font = (st_font_name, st_med_pt)
st_big_font = (st_font_name, st_big_pt)
st_big_bold_font = (st_font_name, st_big_pt, 'bold')
st_color_lightgray = '#eff0f1'
st_color_rosyred = '#ff9eab'
st_color_lightgreen = '#9afc58'

# styles for colored lines
st_warning_style = dict(background='yellow')
st_error_style = dict(background=st_color_rosyred)
st_success_style = dict(background=st_color_lightgreen)

# the global ttk style
st_global = ttk.Style(tk_root)
#st_global.theme_use('default')

# TODO: find nice theme?
# https://wiki.tcl-lang.org/page/List+of+ttk+Themes

# bordered label
st_global.configure(
    'Bordered.TLabel', foreground='black', borderwidth=1, relief='solid',
    font=st_big_font)

# bordered buttons
st_button_border_style = dict(borderwidth=4, relief='raised')
st_global.configure(
    'Generic.TButton', font=st_med_font, **st_button_border_style)
st_global.configure(
    'SingleSymbol.TButton', font=st_big_bold_font,
    width=3, **st_button_border_style)
# --------------

# --- CLASSES/FUNCTIONS FOR CREATING WIDGETS ---
class TracedVar(tk.StringVar, object):
    'base class for a string variable with "live" value restriction'

    def __init__(self, allow_empty=True, **kwargs):
        super(TracedVar, self).__init__(**kwargs)
        self.allow_empty = allow_empty
        self.trace('w', self._validate_kill_args)
        self.last_value = kwargs.get('value', '')

    def check(self, value):
        '''returns True if value is OK, False otherwise
        (override this method in child classes)'''
        return True

    def get(self): # for user
        return self.last_value

    def _get(self): # for internal use
        return super(TracedVar, self).get()

    def _deny(self):
        self.set(self.last_value)

    def _accept(self, value):
        self.last_value = value

    def _validate_kill_args(self, *_):
        self._validate()

    def _validate(self):
        value = self._get()
        value_empty = not value

        if value_empty:
            if self.allow_empty:
                self._accept(value)
            else:
                self._deny()
        elif self.check(value):
            self._accept(value)
        else:
            self._deny()

class TracedNumber(TracedVar):
    'holds string representing a number'

    # NOTE: remember that the user must be able to enter strings that start
    # with . or - or -. (leading + is disallowed for simplicity)

    # strings that are treated as '0' when encountered on their own
    # (when the user calls get and inside the check method)
    _allowed_prefixes = {'.', '-', '-.'}

    def prefix_conv(self, value):
        return '0' if value in self._allowed_prefixes else value

    def get(self):
        return self.prefix_conv(super(TracedNumber, self).get())

    def check(self, value):
        success = False

        if super(TracedNumber, self).check(value):
            try:
                float(self.prefix_conv(value))
                success = True
            except ValueError:
                pass

        return success

class TracedMinMaxValue(TracedNumber):
    # TODO: snapping to min/max value would be nice
    'holds string representing a number with a maximum and minimum value'
    def __init__(self, min_value=-np.inf, max_value=np.inf, **kwargs):
        super(TracedMinMaxValue, self).__init__(**kwargs)

        if min_value >= 0:
            self._allowed_prefixes = ('.',)

        self._max_value = max_value
        self._min_value = min_value

    def check(self, value):
        conv = self.prefix_conv(value)
        return (super(TracedMinMaxValue, self).check(value) and
                self._min_value <= float(conv) <= self._max_value)

class TracedInt(TracedMinMaxValue):
    'holds string representing integer with maximum and minimum value'
    def __init__(self, min_value=-np.inf, max_value=np.inf, **kwargs):
        super(TracedInt, self).__init__(min_value, max_value, **kwargs)

        if min_value < 0:
            self._allowed_prefixes = ('-',)
        else:
            self._allowed_prefixes = tuple()

    def check(self, value):
        # NOTE: checking float(value).is_integer() would allow trailing
        # zeros after the decimal

        return (super(TracedInt, self).check(value) and
                self.prefix_conv(value).lstrip('+').lstrip('-').isdigit())


class DeleGetEntry(ttk.Entry, object):
    '''intended for use with traced variables
    the entry delegates get calls to its textvariable and resets to
    latest acceptable value on losing focus'''
    # (in effect, this resets strings such as - or -. to '0' in this program)

    def __init__(self, master, textvariable, *args, **kwargs):
        self.textvariable = textvariable
        super(DeleGetEntry, self).__init__(
            master, *args, textvariable=textvariable, **kwargs)

        self.bind('<FocusOut>', self._reset)

    def _reset(self, *_):
        value = self.get()
        self.delete(0, tk.END)
        self.insert(0, value)

    def configure(self, *args, **kwargs):
        self.textvariable = kwargs.get('textvariable', self.textvariable)
        return super(DeleGetEntry, self).configure(*args, **kwargs)

    def get(self, *args, **kwargs):
        return(self.textvariable.get())

    def __setitem__(self, item, value):
        if item == 'textvariable':
            self.textvariable = value
        super(DeleGetEntry, self).__setitem__(item, value)

class Log(ScrolledText, object):
    'scrolled textbox for usage as log'
    def __init__(self, *args, **kwargs):
        super(Log, self).__init__(*args, **kwargs)
        self['state'] = 'disabled'
        self.tag_config('warning', **st_warning_style)
        self.tag_config('error', **st_error_style)
        self.tag_config('success', **st_success_style)

    def append(self, text, tag=None, prefix=None):
        self['state'] = 'normal'

        # start a new line if the log is not empty
        if self.get('1.0', tk.END).strip():
            self.insert('end', '\n')

        self.insert(tk.END, '{}{}'.format(prefix or '', text), tag)
        self['state'] = 'disabled'
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

# NOTE: functools.wraps does not preserve function signature in Python 2 -
# if signatures must be preserved use non standard library module "wrapt"
# or port to Python 3
def locked(func):
    'decorated function func acquires global lock during execution'
    @wraps(func)
    def func_locked(*args, **kwargs):
        with LOCK:
            func(*args, **kwargs)

    return func_locked

def walk_widgets(root):
    'yield all descendants of root frame'
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

def time_sleep_ms(ms):
    seconds = ms/1000
    time.sleep(seconds)

def roi_info_line(roi):
    '''returns simplified region info for display in listbox
    the line always starts with the ROI id'''
    circ = isinstance(roi, roitools.CircRoi)
    symbol = sym_white_circle if circ else sym_ballot_box
    unit = 'seconds'
    rounding = 3

    start = time_str(roi.start_pos_msec/1000, unit, rounding)
    line = u'{}  {}  {}'.format(roi._id, symbol, start)

    if hasattr(roi, 'end_pos_msec'):
        end = time_str(roi.end_pos_msec/1000, unit, rounding)
        line += u' {} {}'.format(sym_arrow_right, end)

    return line

def roi_info_dict(roi):
    'creates ordered dict with information on finished roi'
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
        info[attr] = getattr(roi, attr)

    return info

def is_avi(path):
    'checks first twelve bytes for valid avi header'
    with open(path, 'rb') as f:
        head = f.read(12)

    RIFF_signature = '\x52\x49\x46\x46'
    type_avi = '\x41\x56\x49\x20'

    return (
        len(head) == 12 and
        head[:4] == RIFF_signature and
        head[-4:] == type_avi)
# -----------------------------

# --- FUNCTIONS/CLASSES FOR PLAYBACK/ROIS ---
class VideoCaptureError(Exception):
    'custom exception for video loading problems'
    pass

class DummyCircRoi(roitools.CircRoi):
    '''dummy circular ROI that does not compute a mask on construction -
    mask computation can be expensive for large radius values and is
    not needed for planning ROIs'''
    def compute_mask(self):
        pass

class GuiRoiCap(roitools.RoiCap):
    'modified/extended RoiCap for use in this GUI'
    def __init__(self, *args, **kwargs):
        self._brightness_offset = 0
        super(GuiRoiCap, self).__init__(*args, **kwargs)

    def _apply_offset(self, frame):
        if self.brightness_offset == 0:
            return

        h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, frame))
        cv2.add(v, self.brightness_offset, v)
        cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR, frame)

    def draw_active_rois(self):
        for roi in self.rois.viewvalues():
            if not hasattr(roi, 'end_pos_frames'):
                roi.draw()

    def redraw(self):
        'apply brightness offset to current frame, redraw rois'
        frame = self.latest_frame_original.copy()
        self._apply_offset(frame)
        self.latest_frame = frame

        # redraw rois
        self.draw_active_rois()

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
        # NOTE: at this point ROIs are not notified like in parent class, only
        # redrawn, and only if active - this is fine for simple ROI planning
        # with dummy ROIs that don't collect any data
        # -> adjust code as needed if program is extended to collect ROI data

        frame = vidtools.PyCap.next(self)

        # the current offset is applied directly to the fresh frame
        # get copy of frame before applying brightness and drawing ROIs
        self.latest_frame_original = frame.copy()
        self._apply_offset(frame)
        self.latest_frame = frame

        self.draw_active_rois()

        return self.latest_frame

class CapRunner(threading.Thread):
    'Thread responsible for displaying video capture in context of this GUI'
    # Keep this simple! This class should not know about widgets.

    def _re_init(self):
        self.cap = None
        self.hold = True
        self.backwards = False
        self.frame_delay_msec = None
        self.num_last_shown_frame = 0

    def __init__(self, windowname='video'):
        super(CapRunner, self).__init__()

        self.daemon = True
        self.windowname = windowname
        self.idle_wait_msec = 50
        self._re_init()

        # NOTE: without the next line, the program has a high chance of
        # crashing when loading a video - I don't know why yet!
        cv2.namedWindow(self.windowname)

    def set_position(self, frame):
        # NOTE: the check whether the cap has more than one frame should not
        # be necessary, but testing revealed that cv2.VideoCapture.read seems
        # to have a bug for single frame videos:
        # after reading the frame once and resetting the capture's position to
        # zero, subsequent calls to read fail to return the frame

        'jump to specific frame and update display'
        if self.cap.frame_count > 1:
            self.cap.pos_frames = frame
            self.cap.pos_frames -= 1
            next(self.cap)


    def unload_video(self):
        'unload video - expects to be called from loaded state'
        self.cap.release()
        self._re_init()

        # destroying the window sometimes takes a few tries, see
        # https://stackoverflow.com/a/25301690
        # https://stackoverflow.com/a/50538883
        # (inconsistent black magic)
        for _ in range(10):
            cv2.destroyWindow(self.windowname)
            cv2.waitKey(1)

    def load_video(self, path):
        'load video - expects to be called in unloaded state'
        try:
            cap = GuiRoiCap(path)
            cap.title = self.windowname # generic name to reuse display window
            next(cap)
        except (StopIteration, TypeError):
            # TODO: what else can be raised by cv2.VideoCapture(path)?
            raise VideoCaptureError
        else:
            # no fps or no frames? (zero or None)
            if not all((cap.fps, cap.frame_count)):
                raise VideoCaptureError
            else:
                # video seems to be fine
                # NOTE: cv2.waitKey can't wait less than a millisecond
                # TODO: consider giving frame_delay_msec an upper limit in
                # order to guard against weird videos with ridiculously low fps
                self.frame_delay_msec = max(int(1000/cap.fps), 1)
                self.cap = cap
                cv2.namedWindow(self.windowname)
                cv2.setMouseCallback(self.windowname, cb_videoclick)

    def run(self):
        while True:
            # set default waiting time when not playing a video
            wait_msec = self.idle_wait_msec
            # let the time module do the waiting if there's no window
            wait_function = time_sleep_ms

            LOCK.acquire()

            if self.cap:
                # let cv2 module do the waiting
                wait_function = cv2.waitKey

                # if not holding, get the next frame and set
                # a roughly appropriate waiting time between frames
                if not self.hold:
                    if self.backwards:
                        self.cap.pos_frames -= 2 # clips to zero

                    try:
                        next(self.cap) # increases pos_frames by 1
                    except StopIteration:
                        self.hold = True
                    else:
                        wait_msec = self.frame_delay_msec

                    # reached beginning of video while playing backwards?
                    if self.backwards and self.cap.pos_frames == 1:
                        self.hold = True

                self.cap.latest_frame.show()
                self.num_last_shown_frame = self.cap.pos_frames

            LOCK.release()

            # wait AFTER the lock is released in order to prevent the thread
            # from immediately trying to acquire it again
            wait_function(wait_msec)

# set up callback for video window
def check_empty(value, description=None):
    if value == '':
        msg = 'missing {} value'.format(description or '')
        tk_log.error(msg)
        raise ValueError

def roi_from_click(click_x, click_y):
    'construct a ROI from click-coordinates and ROI selection parameters'
    rtype = tkvar_roitype.get()

    if rtype == RECTANGULAR:
        width_str = tkvar_width.get()
        check_empty(width_str, 'region width')
        height_str = tkvar_height.get()
        check_empty(height_str, 'region height')
        vertex1 = (click_x, click_y)
        vertex2 = (click_x + int(width_str), click_y + int(height_str))
        return roitools.RectRoi(vertex1, vertex2)
    elif rtype == CIRCULAR:
        radius_str = tkvar_radius.get()
        check_empty(radius_str, 'region radius')
        return DummyCircRoi((click_x, click_y), int(radius_str))

def out_of_frame(roi, width, height):
    'check whether a circular roi is out of bounds'
    x, y = roi.center
    return (
        x - roi.radius < 0 or
        y - roi.radius < 0 or
        x + roi.radius >= width or
        y + roi.radius >= height)

def add_roi(x, y):
    # NOTE this is run in the RUNNER's thread
    # interacting with log and listboxes requires mttkinter to do it's job!

    'create ROI based on x, y coordinates'
    start_ms = RUNNER.cap.pos_msec
    start_frame = RUNNER.cap.pos_frames

    try:
        roi = roi_from_click(x, y)
    except ValueError:
        return

    # position supported?
    circ = isinstance(roi, roitools.CircRoi)
    width, height = RUNNER.cap.frame_width, RUNNER.cap.frame_height
    if circ and out_of_frame(roi, width, height):
        msg = 'region out of frame - partial circles are not supported yet'
        tk_log.error(msg)
        roitools.BaseRoi._next_id -= 1
    else:
        # add roi to frame
        try:
            RUNNER.cap.add_roi(roi)
        except ValueError as e:
            # happens when maximum number of active ROIs is exceeded
            tk_log.error(str(e))
        else:
            roi.start_pos_msec = start_ms
            roi.start_pos_frames = start_frame
            roi.draw()

            # add roi to listbox
            tk_active_roi_box.insert(tk.END, roi_info_line(roi))

            # log
            tk_log.append('started region {}'.format(roi._id))

def cb_videoclick(event, x, y, flags, _):
    event = vidtools._id2ev[event]

    # left click -> add ROI
    # anything else -> do nothing
    if event != 'lbuttondown':
        return

    # catching AttributeError:
    # RUNNER.cap could become None (e.g. due to ejection) during execution of
    # add_roi -> catching the AttributeError is easier than working out the
    # locking logic
    try:
        add_roi(x, y)
    except AttributeError:
        pass

# -------------------------------------------

# --- GLOBALS/CONSTANTS ----
# (no widgets)
RUNNER = CapRunner()
LOCK = threading.Lock()
FRAMES = 'frames'
SECONDS = 'seconds'
CIRCULAR = 'circular'
RECTANGULAR = 'rectangular'
DELTABRIGHT = 10
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
def cb_ask_close():
    msg = 'Exit program?\nAll unsaved regions will be lost.'
    if tkMessageBox.askyesno('Exit?', msg):
        tk_root.destroy()

# VIDEO EJECTION

# TODO/NOTE: rarely, ejecting can freeze the GUI...
# I suspect OpenCV struggling to destroy the window because I'm (sometimes)
# getting gibberish GLib errors in the terminal - investigate!
@locked
def cb_eject_video():
    # TODO: ask before ejecting if there are ROIs in the listboxes
    # (needs to consider that this function is also called by cb_select_video)
    'unloads video through RUNNER and updates GUI'
    set_state_unloaded()

    for box in (tk_active_roi_box, tk_finished_roi_box):
        box.delete(0, tk.END)

    vidname = RUNNER.cap.video
    RUNNER.unload_video()

    tkvar_vidname.set('')
    tk_log.append('ejected "{}"'.format(os.path.basename(vidname)))

    time.sleep(0.05) # may help with freezing issue

# VIDEO SELECTION
def cb_select_video():
    'ejects previous video and loads video through RUNNER and updates GUI'
    path = tkFileDialog.askopenfilename()
    basename = os.path.basename(path) if path else ''
    if not basename:
        return

    if RUNNER.cap:
        cb_eject_video()

    # currently, we want avi files for best OpenCV support, see
    # https://stackoverflow.com/a/20865122
    if not (tkvar_allow_all.get() or is_avi(path)):
        msg = '"{}" not an avi - currently only avi files are supported'
        msg = msg.format(basename)
        tk_log.error(msg)
        return

    try:
        RUNNER.load_video(path)
    except VideoCaptureError:
        msg = 'can\'t process "{}" as video (corrupt file?)'.format(basename)
        tk_log.error(msg)
    else:
        roitools.BaseRoi._next_id = 1
        tk_bar['to'] = RUNNER.cap.frame_count
        set_state_loaded()

        # set video name label
        width = tk_vidlabel['width']
        label_text = (basename if len(basename) <= width else
                      basename[:max(width - 3, 0)] + '...')
        tkvar_vidname.set(label_text)

        # log
        msg = 'loaded "{}" ({} fps, {} frames)'.format(
            basename, RUNNER.cap.fps, int(RUNNER.cap.frame_count))
        tk_log.append(msg)

# VIDEO PROGRESS
@locked
def cb_bar_set_frame(event):
    # NOTE: see explanation why bar is disabled by default
    # at set_state_loaded definition

    if RUNNER.cap:
        # briefly enable
        tk_bar.state([ttk_NORMAL])
        tk_bar.event_generate('<Button-1>', x=event.x, y=event.y)
        tk_bar.state([ttk_DISABLED])

        frame = round(tk_bar.get())
        RUNNER.set_position(frame)

# CLASSICAL CONTROLS (play, pause, stop)
def play():
    RUNNER.hold = False

@locked
def cb_play_forwards():
    RUNNER.backwards = False
    play()

@locked
def cb_play_backwards():
    RUNNER.backwards = True
    play()

@locked
def cb_pause():
    RUNNER.hold = True

@locked
def cb_stop():
    RUNNER.hold = True
    RUNNER.set_position(0)

# JUMPING
def adjust_seconds(seconds):
    '''helper for jumping
    converts seconds to nearest possible value in frames'''
    frames = seconds*RUNNER.cap.fps
    frames_adjusted = int(round(frames))
    secs_adjusted = frames_adjusted/RUNNER.cap.fps

    # issue warning if adjusting changed input value
    if frames != frames_adjusted:
        template = 'adjusted jump value to {} {} ({} {})'
        insert = (secs_adjusted, SECONDS, frames_adjusted, FRAMES)
        tk_log.warning(template.format(*insert))

    return frames_adjusted

@locked
def cb_jump(back=False):
    'changes video position according to jump step field'
    step = tk_step_entry.get()
    unit = tkvar_unit.get()

    # empty?
    try:
        check_empty(step, description=unit)
    except ValueError:
        return

    # seconds need adjustment?
    if unit == SECONDS:
        jump_frames = adjust_seconds(float(step))
    else:
        jump_frames = int(step)

    # work to do?
    if jump_frames == 0:
        return

    # perform the jump (always in frames for exactness)
    before_pos_frames = RUNNER.cap.pos_frames
    before_pos_msec = RUNNER.cap.pos_msec

    if back:
        jump_frames *= -1

    RUNNER.set_position(RUNNER.cap.pos_frames + jump_frames)

    # log jump
    delta_frames = int(RUNNER.cap.pos_frames - before_pos_frames)
    delta_msec = RUNNER.cap.pos_msec - before_pos_msec
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
        tk_log.warning('reached {} of video'.format(where)) # only if overshot

cb_jump_back = partial(cb_jump, back=True)

# BRIGHTNESS
def cb_brightness_up():
    RUNNER.cap.brightness_offset += DELTABRIGHT

def cb_brightness_down():
    RUNNER.cap.brightness_offset -= DELTABRIGHT

def cb_brightness_reset():
    RUNNER.cap.brightness_offset = 0

# ROI DELETION
def selected_ids(listbox):
    # TODO: parsing ROI ids from the beginning of listbox strings is straight
    # forward but mixes presentation and model -> consider Listbox subclass
    # internally keeping track of ROIs being held (id and line)
    'returns list of ROI ids corresponding to selected listbox items'
    lines = (listbox.get(i) for i in listbox.curselection())
    return [int(line.split(' ', 1)[0]) for line in lines]

def del_rois(listbox, redraw=True):
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
        del RUNNER.cap.rois[id_]

    # log
    msg = 'deleted region{} {}'.format(
        's'*(len(ids) > 1), ', '.join(map(str, ids)))
    tk_log.append(msg)

    # redraw frame
    if redraw:
        RUNNER.cap.redraw()

def cb_del_active_rois():
    del_rois(tk_active_roi_box, redraw=True)

def cb_del_finished_rois():
    del_rois(tk_finished_roi_box, redraw=False)

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
    end_msec = RUNNER.cap.pos_msec
    end_frame = RUNNER.cap.pos_frames
    swap = tkvar_swap_times.get()

    for id_ in ids:
        roi = RUNNER.cap.rois[id_]
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
        roi = RUNNER.cap.rois[id_]
        tk_finished_roi_box.insert(tk.END, roi_info_line(roi))

    # log
    msg = 'finished region{} {}'.format(
        's'*(len(ids) > 1), ', '.join(map(str, ids)))
    tk_log.append(msg)

    # redraw frame
    RUNNER.cap.redraw()

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
    row_dicts = [roi_info_dict(RUNNER.cap.rois[id_]) for id_ in ids]

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

# PERIODIC EXECUTION BY MAIN THREAD

# NOTE: there is no RUNNER-updated FloatVar/StringVar bound to the
# progress bar and time label - instead the main thread periodically reads
# RUNNER's instance variables and then sets the widgets.
# Calling the original set/get of DoubleVar/StringVar in locked area of thread
# can cause deadlocks.

# keep progress bar up to date
def cb_after_update_bar():
    # NOTE: see explanation why bar is disabled by default
    # at set_state_loaded definition

    # briefly enable
    tk_bar.state([ttk_NORMAL])
    tk_bar.set(RUNNER.num_last_shown_frame)
    tk_bar.state([ttk_DISABLED])

    tk_root.after_idle(tk_root.after, 1, cb_after_update_bar)

# keep time label up to date
def cb_after_update_timelabel():
    if RUNNER.cap:
        sec = RUNNER.cap.pos_msec/1000
    else:
        sec = 0

    t = time_str(sec, 'seconds', rounding=0)
    tk_timelabel['text'] = t
    tk_root.after_idle(tk_root.after, 1, cb_after_update_timelabel)

# ------------------------

# --- CREATION AND PLACEMENT OF WIDGETS ---
# (from top to bottom, left to right)
tk_root.title('ROI Planner')
tk_root.protocol('WM_DELETE_WINDOW', cb_ask_close)

# parent frame for all widgets
tk_root_frame = ttk.Frame(tk_root)
tk_root_frame.grid(row=0, column=0)

# SET UP MENU
tk_menubar = tk.Menu(tk_root)
tk_options_menu = tk.Menu(tk_menubar)
tk_menubar.add_cascade(label='Options', menu=tk_options_menu)
tk_root['menu'] = tk_menubar

# option to automatically delete exported videos
tkvar_del_exported = tk.BooleanVar(value=True)
tk_options_menu.add_checkbutton(
    label='delete exported regions',
    onvalue=True, offvalue=False, variable=tkvar_del_exported)

# option to switch start and end times for rois with start > end
tkvar_swap_times = tk.BooleanVar(value=False)
label = (
    'swap timestamps when finishing regions '
    'with end time earlier than start time')
tk_options_menu.add_checkbutton(
    label=label, onvalue=True, offvalue=False, variable=tkvar_swap_times)

# option to allow non-avi videos
tkvar_allow_all = tk.BooleanVar(value=False)
tk_options_menu.add_checkbutton(
    label='allow non avi files (unstable)',
    onvalue=True, offvalue=False, variable=tkvar_allow_all)

# SET UP FILE PICKER
tk_select_frame = ttk.Frame(tk_root_frame)
tk_select_frame.grid(row=0, column=0)
tkvar_vidname = tk.StringVar()

# select video button
tk_select_button = ttk.Button(
    tk_select_frame, style='Generic.TButton',
    text='select video', command=cb_select_video)
tk_select_button.grid(row=0, column=0)

# video name label
# TODO: make label a scrolled readonly entry, and hide the
# scrollbar whenever the entry fits
tk_vidlabel = ttk.Label(
    tk_select_frame, textvariable=tkvar_vidname, width=60,
    style='Bordered.TLabel')
tk_vidlabel.bind('<Button-1>', lambda *_: cb_select_video())
tk_vidlabel.grid(row=0, column=2)

# eject video button
tk_eject_button = ttk.Button(
    tk_select_frame, style='SingleSymbol.TButton',
    text=sym_eject, command=cb_eject_video)
tk_eject_button.grid(row=0, column=1)

# separator under file picker
ttk.Separator(tk_root_frame, orient=tk.HORIZONTAL).grid(row=1, column=0)

# SET UP PROGRESS INFO
tk_progress_frame = ttk.Frame(tk_root_frame)
tk_progress_frame.grid(row=2, column=0)

# progress bar
tk_bar = ttk.Scale(
    tk_progress_frame, orient=tk.HORIZONTAL, from_=1, takefocus=False)

# jump bar with left click in addition to right click
# -> make bar think a '<Button-1>' event is a '<Button-3>' event
tk_bar.bind(
    '<Button-1>',
    lambda event: tk_bar.event_generate('<Button-3>', x=event.x, y=event.y))

# jump to clicked position at mousebutton-release
for button in range(1, 6):
    event = '<ButtonRelease-{}>'.format(button)
    tk_bar.bind(event, cb_bar_set_frame)

cb_after_update_bar() # invoke periodic reading of frame position
tk_bar.grid(row=0, column=0)

# time label next to progress bar
tk_timelabel = ttk.Label(tk_progress_frame)
tk_timelabel.grid(row=0, column=1)
cb_after_update_timelabel() # invoke periodic reading of time

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
tk_back_button = ttk.Button(
    tk_jump_frame, text=sym_double_triangle_left,
    style='SingleSymbol.TButton', command=cb_jump_back)
tk_back_button.grid(row=0, column=0)

# jump step entry
pos_val_restrictions = dict(min_value=0, max_value=99999)
tk_timeentry_frame = ttk.Frame(tk_jump_frame)
tk_timeentry_frame.grid(row=0, column=1)
tkvar_seconds = TracedMinMaxValue(**pos_val_restrictions)

tkvar_frames = TracedInt(**pos_val_restrictions)
tk_step_entry = DeleGetEntry(
    tk_timeentry_frame, textvariable=None, width=7, justify='center')
tk_step_entry.grid(row=0, column=0)

# jump step unit label
tkvar_unit = tk.StringVar()
tk_unit_label = ttk.Label(
    tk_timeentry_frame, textvariable=tkvar_unit, anchor='center')
tk_unit_label.grid(row=1, column=0)

# jump forward button
tk_forward_button = ttk.Button(
    tk_jump_frame, text=sym_double_triangle_right,
    style='SingleSymbol.TButton', command=cb_jump)
tk_forward_button.grid(row=0, column=2)

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
text = 'finish'
tk_finish_button = ttk.Button(
    tk_active_roi_frame, text=text, width=len(text),
    style='Generic.TButton', command=cb_finish_rois)
tk_finish_button.grid(row=3, column=0)

# export finished ROIs button
text = 'export'
tk_export_button = ttk.Button(
    tk_finished_roi_frame, text=text, width=len(text),
    command=cb_export, style='Generic.TButton')
tk_export_button.grid(row=3, column=0)

# SET UP ROI PARAMETER SELECTION FRAME
# TODO: section might need refactoring
tk_roi_select_frame = ttk.Labelframe(tk_roi_frame, text='region parameters')
tk_roi_select_frame.grid(row=0, column=2)

# NOTE roitype, width, height and radius must be read from the cv2 video
# callback (cb_videoclick) which runs in the CapRunner's thread!
# -> make TracedVars and access last_value (via TracedVar.get)
tkvar_roitype = TracedVar(value=RECTANGULAR)

roi_entry_params = dict(master=tk_roi_select_frame, width=5, justify='center')
tkvar_width = TracedInt(**pos_val_restrictions)
tkvar_height = TracedInt(**pos_val_restrictions)

tkvar_radius = TracedInt(**pos_val_restrictions)
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
tk_log = Log(tk_root_frame, height=10, bg=st_color_lightgray)
tk_log.grid(row=7, column=0)
# -----------------------------------------

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
    ttk.Radiobutton: dict(sticky=tk.W)
}

# map widget style to grid parameters
style_grid_params = {
    'Generic.TButton': dict(pady=(5, 0)),
    'SingleSymbol.TButton': dict(padx=1),
}

# map individual widgets to grid parameters
widget_grid_params = {
    tk_root_frame: dict(padx=10, pady=10),
    tk_select_button: dict(sticky=tk.NS, pady=0, padx=(0, 1)),
    tk_vidlabel: dict(sticky=(tk.W, tk.NS), padx=(10, 0)),
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

widget_grid_params[tk_export_button] = widget_grid_params[tk_finish_button]
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
tk_select_frame.columnconfigure(2, weight=1) # expand video label
tk_control_frame.columnconfigure(1, pad=40)
tk_roi_frame.columnconfigure(1, pad=60)

# pad all labelframes (should also be possible through grid_configure with
# ipadx/ipady, but for some reason the padding is not applied on all sides)
for w in ALL_WIDGETS:
    if isinstance(w, ttk.Labelframe):
        w['padding'] = 5

# pad left/right of video name
tk_vidlabel['padding'] = (10, 0, 10, 0)

# miscellaneous
tk_root.resizable(False, False)
#tk_select_frame.propagate(False)
# -------------------------

# --- ENABLING/DISABLING WIDGETS ---
def set_widget_states(widgets, state=tk.NORMAL):
    '''given an iterable of widgets, tries to set state to tk.NORMAL or
    tk.DISABLED and suppresses errors if that's not possible'''
    for widget in widgets:
        try:
            widget['state'] = state
        except tk.TclError:
            # deal with ttk widgets such as ttk.Scale
            ttk_state = ttk_NORMAL if state == tk.NORMAL else ttk_DISABLED

            try:
                widget.state([ttk_state])
            except (AttributeError, tk.TclError):
                pass

# function for state: no video
# NOTE: activating the boxes is a little kludge that makes
# deleting from them later a bit easier
set_state_unloaded = partial(
    set_widget_states,
    ALL_WIDGETS - {tk_select_button, tk_active_roi_box, tk_finished_roi_box},
    tk.DISABLED)

# function for state: video loaded
# NOTE: the progress bar (scale) is disabled by default to avoid races between
# clicking it and updating it periodically via cb_after_update_bar.
# Otherwise the bar likely does not respond or flicker when interacted with.
# The click event is still registered in disabled state, but the bar does not
# move. FUNCTIONS THAT WANT TO SET THE BAR MUST BRIEFLY ENABLE IT.
set_state_loaded = partial(
    set_widget_states, ALL_WIDGETS - {tk_bar}, tk.NORMAL)

if __name__ == '__main__':
    set_state_unloaded()
    RUNNER.start()
    tk_root.mainloop()
