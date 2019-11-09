'''examples and helpers for vidtools and roitools users

Functions accepting ROIs work with MeanRectRoi or MeanCircRoi instances.

The main data collection functions are get_roidata and collect_plot_export.'''

from collections import deque
from datetime import datetime
from itertools import count
from math import log10, floor
from pprint import pprint as pretty_print
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# (try to) import useful objects for the user if script is executed directly
if __name__ == '__main__' and __package__ is None:
    # add roitools directory to path
    dirname = os.path.dirname

    abs_path = os.path.abspath(sys.argv[0])
    one_up = dirname(dirname(abs_path))
    sys.path.insert(0, one_up)

    # imports
    try:
        import vidtools
        from vidtools import PyCap

        import roitools
        from roitools import RoiCap, MeanBgrRecord, MeanRectRoi, MeanCircRoi
    except ImportError:
        pass

def reset_roi_ids():
    'make new ROIs start at id 1 again'
    BaseRoi._next_id = 1

def save_roi(roi, path):
    '''save ROI as csv file
    if path is a directory, uses filename roi_<id>_out.csv'''
    if os.path.isdir(path):
        filename = 'roi_{}_out.csv'.format(roi._id)
        path = os.path.join(path, filename)

    with open(path, 'w') as outfile:
        roi.to_file(outfile)

def array_to_csv(filepath, array, header=''):
    np.savetxt(filepath, array, delimiter=',', header=header, comments='')

def load_roi(roi_csv_path, header_rows=11):
    'load ROI data as array from csv'
    return np.loadtxt(roi_csv_path, skiprows=header_rows, delimiter=',')

def concat_arrays(arrays):
    'concats arrays vertically'
    return np.vstack(arrays)

def _set_up_result_directory(videopath):
    'helper function to set up a directory for result files'
    dirname = os.path.dirname(videopath)
    videoname = os.path.basename(videopath)

    now = datetime.now()
    date = now.date().isoformat()
    time = now.time().strftime('%Hh%Mm%Ss')

    saveto = os.path.join(
        dirname, '{}_analyzed_{}_{}'.format(videoname, date, time))
    os.mkdir(saveto)

    return saveto

def get_roidata(videopath, roi_specs):
    '''videopath: complete path to video file
    roi_specs: list of (roi_instance, birth_msec, death_msec) tuples, e.g.

    roi_specs = [
        (MeanRectRoi((0, 0), (100, 100)), 2000, 10000), # observe sec 2 -> 10
        (MeanRectRoi((10, 10), (110, 110)), 10000, 20000) # observe sec 10 -> 20
    ]

    lets regions in roi_specs collect data during their lifetime

    will dump all results and region screenshots into a new directory located
    in the same directory as the video, returns path'''

    # set up a directory for results
    saveto = _set_up_result_directory(videopath)

    # save ROI specs
    with open(os.path.join(saveto, 'roi_specs.py'), 'w') as f:
        pretty_print(roi_specs, f)

    # set up queues for insertion/deletion
    insertion_order = deque(
        sorted(roi_specs, key=lambda (roi, birth, death): birth))
    deletion_order = deque(
        sorted(insertion_order, key=lambda (roi, birth, death): death))

    # iterate video
    cap = RoiCap(videopath)

    screenshot_next_frame = False
    for frame in cap:
        # NOTE: fast forwarding to the next birth/death (via cap.play) instead
        # of executing loop body for every frame has been tried and does not
        # seem to yield a noticeable performance boost compared to this simpler
        # version
        # also, skipping via setting attributes seems to be non-exact for some
        # types of videos -> see comments in image_series function
        print '{:.1f} %\r'.format(100*cap.pos_frames/cap.frame_count),
        sys.stdout.flush()

        # save screenshot
        if screenshot_next_frame:
            im_filename = 'frame_{}.png'.format(int(cap.pos_frames))
            frame.save(os.path.join(saveto, im_filename))
            screenshot_next_frame = False

        # add rois
        while insertion_order:
            roi, birth, _ = insertion_order[0]

            if cap.pos_msec >= birth:
                insertion_order.popleft()
                screenshot_next_frame = True
                cap.add_roi(roi)
                # manually notify ROI in order to not miss current frame
                roi.notified()
            else:
                break

        # delete rois
        while deletion_order:
            roi, _, death = deletion_order[0]
            if cap.pos_msec >= death:
                deletion_order.popleft()
                save_roi(roi, saveto)
                cap.delete_roi(roi._id)
            else:
                break

        # abort if no further observing planned
        if not cap.rois and not insertion_order:
            break

    # save leftover rois (with death > video length)
    for roi in cap.rois.itervalues():
        save_roi(roi, saveto)

    # save a screenshot of all rois
    cap.pos_frames -= 1
    for roi, _, _ in roi_specs:
        cap.add_roi(roi)
    last_frame = next(cap)
    last_frame.save(os.path.join(saveto, 'all_rois.png'))

    print '100.0 %'
    return saveto

def plot_roi(array, linetype='-', channels='bgr', title='ROI Plot', ax=None):
    '''creates plot from ROI data array, returns the axes object (the plot)
    (use plt.show() to show the plot)'''
    if len(array.shape) <= 1:
        print 'nothing to plot'
        return

    msec = array[:, 1]
    channels = set(channels.lower()).intersection('bgr')

    if ax is None:
        _, ax = plt.subplots()

    ax.set_xlabel('milliseconds')
    ax.set_ylabel('mean channel intensity')
    ax.set_title(title)

    for c, idx in zip('bgr', xrange(2, 5)):
        if c in channels:
            ax.plot(msec, array[:, idx], linetype + c)

    ax.set_xlim(array[0][1], array[-1][1])
    return ax

def collect_plot_export(videopath, roi_specs):
    '''compound comfort function

    visualizes regions with non-overlapping lifetimes and saves helpful files

    - runs get_roidata with roi_specs (see get_roidata documentation)
    - concatenates all rows of collected data and saves to all_rois.csv
    - plots the result with horizontal lines indicating change of ROI
    - saves the figure as an image

    returns the axes object (the plot)
    (use plt.show() to show the plot)'''

    # get numerical data
    saveto = get_roidata(videopath, roi_specs)

    # concat data from all rois
    region_ids = [roi._id for roi, _, _ in roi_specs]
    roi_arrays = [load_roi(os.path.join(saveto, 'roi_{}_out.csv'.format(id_)))
                  for id_ in region_ids]

    merged = concat_arrays(roi_arrays)
    array_to_csv(os.path.join(saveto, 'all_rois.csv'), merged)

    # plot and add ROI-deaths to plot
    ax = plot_roi(merged)
    for roi, birth, death in roi_specs:
        ax.axvline(death, color='black')

    ax.get_figure().savefig(os.path.join(saveto, 'fig_autosave.png'),
                            dpi=1200, bbox_inches='tight')

    print 'saved data to {}'.format(saveto)
    return ax

def image_series(videopath, start_msec, end_msec, n):
    'save every nth frame between start_msec and end_msec'
    # NOTE: I compared iterating the video frame-by-frame versus skipping ahead
    # n frames by setting the pos_frames attribute. OpenCV sometimes misses
    # the correct frame for non-avi files with the second approach.
    # This can only be seen when taking images from a video that
    # numbers its frames on-screen because the pos_frames and pos_msec
    # attributes report the values you would expect after setting them.

    # NOTE: tested with well behaved 60 fps videos

    # set up a directory for results
    saveto = _set_up_result_directory(videopath)

    # create template to pad filenames with zeros to make them ordered
    cap = PyCap(videopath)
    pad = int(floor(log10(cap.frame_count/float(n)) + 1))
    template = '{{:0>{}d}}_{{}}.png'.format(pad)

    # iterate and save images
    imgcount = count(1)
    t_one_frame = 1000.0/cap.fps
    cap.pos_msec = start_msec

    if cap.pos_msec != start_msec:
        print('adjusted start_msec to {}'.format(cap.pos_msec))
        start_msec = cap.pos_msec

    for i, frame in enumerate(cap):
        # current cap.pos_msec and cap.pos_frames are the values for the NEXT
        # frame that can be retrieved with next(cap)
        t_abs = cap.pos_msec - t_one_frame
        if t_abs > end_msec:
            break

        if i%n == 0:
            t_rel = t_abs - start_msec
            img_fname = template.format(next(imgcount), t_rel)
            frame.save(os.path.join(saveto, img_fname))

    return saveto

def get_frames(videopath, frames):
    '''frames: iterable of frame numbers
    saves screenshots for given frame numbers'''

    # set up a directory for results
    saveto = _set_up_result_directory(videopath)

    frames = set(frames)
    last = max(frames)
    cap = PyCap(videopath)
    template = 'frame_{}.png'

    # we don't jump around with cap.pos_frames directly because it can be
    # inexact for some types of videos (i.e. non-avi files)
    for frame in cap:
        if cap.pos_frames in frames:
            frame.save(os.path.join(saveto, template.format(cap.pos_frames)))
        if cap.pos_frames > last:
            break

    return saveto
