from roitools.examples.examples import collect_plot_export
from roitools.roitools import MeanRectRoi

video = 'testfiles/traffic_light_small.avi'
roi = MeanRectRoi((277, 56), (367, 271))
start = 40
end = 45960

collect_plot_export(video, [(roi, start, end)])
