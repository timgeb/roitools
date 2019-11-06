# --- ABANDONED ---
# crude command line tool for planning ROIs - abandoned

from os.path import basename
from sys import argv, exit

import cv2
import numpy as np

from vidtools import _id2ev
from roitools import RoiCap, MeanRectRoi, MeanCircRoi

ROISPEC = ('R', 90, 15) # (type, width, height)
BRIGHT_OFFSET = 0

def brighten(frame, offset):
    'brighten/darken a frame in-place by adding offset'
    if not offset:
        return

    cp = np.zeros(frame.shape, dtype=np.int32)
    cp += frame
    cp += offset
    cp = np.clip(cp, a_min=0, a_max=255)
    frame[:] = cp

def spec_from_str(inputstr):
    'construct a ROI spec tuple from an input str'
    spec = inputstr.split()
    if spec[0].lower() == 'c':
        radius = int(spec[1])
        return ('C', radius)
    elif spec[0].lower() == 'r':
        width, height = int(spec[1]), int(spec[2])
        return ('R', width, height)
    else:
        raise ValueError

def roi_from_spec(click_x, click_y, spec):
    '''construct a ROI from click-coordinates and spec,
    expects valid spec'''
    rtype = spec[0]
    if rtype == 'C':
        radius = spec[1]
        return MeanCircRoi((click_x, click_y), radius)
    elif rtype == 'R':
        width, height = spec[1], spec[2]
        return MeanRectRoi((click_x, click_y),
                           (click_x + width, click_y + height))

def _debug_callback(event, x, y, flags):
    'show event infos'
    eventname = _id2ev[event]
    show = (eventname != 'mousemove' and
            'buttonup' not in eventname)

    if show:
        template = 'xy=({}, {}), eventID={}, event={}, flags={}'
        print template.format(x, y, event, eventname, flags)

def callback(event, x, y, flags, cap):
    # TODO disable callbacks during execution, setting a
    # temporary no op callback does not seem to work
    global ROISPEC
    global BRIGHT_OFFSET

    #_debug_callback(event, x, y, flags)

    CTRL = 8
    SHIFT = 16
    NOFLAGS = 0
    event = _id2ev[event]

    # left click: draw ROI
    if flags == NOFLAGS and event == 'lbuttondown':
        roi = roi_from_spec(x, y, ROISPEC)
        cap.add_roi(roi)
        roi.birth = cap.pos_msec
        print ('frame {}, ms {}: add ROI {} {}'
                .format(int(cap.pos_frames), cap.pos_msec, roi._id, repr(roi)))
        roi.draw()
    # right click: delete youngest ROI that was clicked
    elif flags == NOFLAGS and event == 'flag_rbutton':
        for roi_id, roi in sorted(cap.rois.iteritems(), reverse=True):
            if roi.contains((x, y)):
                cap.delete_roi(roi_id)
                msg = 'frame {}, ms {}: del ROI {}, spec ({}, {}, {})'
                print (msg.format(int(cap.pos_frames), cap.pos_msec, roi._id,
                                  repr(roi), roi.birth, cap.pos_msec))
                cap.pos_frames -= 1
                next(cap)
                brighten(cap.latest_frame, BRIGHT_OFFSET)
                break
    # shift + left click: change time
    elif flags == SHIFT and event == 'lbuttondown':
        delta_t = int(raw_input('input delta sec = '))
        new_pos = cap.pos_msec + delta_t*1000
        cap.pos_msec = new_pos
        cap.pos_frames -= 1
        next(cap)
        brighten(cap.latest_frame, BRIGHT_OFFSET)
        print 'changed video position to msec {}'.format(cap.pos_msec)
    # ctrl + right click: adjust roi
    elif flags == CTRL and event == 'flag_rbutton':
        print ('input C <RADIUS> (e.g. C 10) or R <WIDTH> <HEIGHT> (e.g. R 10 20) (current: {})'.
               format(' '.join(map(str, ROISPEC))))
        spec = raw_input('input> ')
        try:
            ROISPEC = spec_from_str(spec)
            print 'changed roi spec to {}'.format(spec)
        except (IndexError, ValueError):
            print 'malformed roi specification'
    # ctrl + left click: brighten/darken
    elif flags == CTRL and event == 'lbuttondown':
        print 'input integer between -255 and 255'
        try:
            BRIGHT_OFFSET = int(raw_input('input> '))
            if not (-255 <= BRIGHT_OFFSET <= 255):
                raise ValueError
        except ValueError:
            print 'invalid brigthen/darken offset (must be between -255 and 255)'
        else:
            brighten(cap.latest_frame, BRIGHT_OFFSET)

try:
    path = argv[1].strip()
except IndexError:
    print 'missing command line argument: filename'
    exit(1)

filename = basename(path)
cap = RoiCap(path)
#print cap.info()
next(cap)

print '---- COMMANDS ----'
print 'left-click: draw ROI'
print 'right-click in ROI: delete ROI'
print 'SHIFT and left-click: go forward/backward'
print 'CTRL and right-click: adjust ROI'
print 'CTRL and left-click: brighten/darken'
print 'CRTL-C: quit\n'

cv2.namedWindow(filename)
cv2.setMouseCallback(filename, callback, cap)

try:
    while True:
        cap.latest_frame.title = filename
        cap.latest_frame.show()
        cv2.waitKey(100)
except KeyboardInterrupt:
    cap._cv2cap.release()
