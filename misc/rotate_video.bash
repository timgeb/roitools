#!/bin/bash

# rotate a video with ffmpeg
# argument 1 is path, argument 2 is angle

set -e

path=$1 # path to video
angle=$2 # angle in degrees (positive: cw, negative: ccw)
crf=$3 # quality (0: highest, 51: lowest) - optional

folder=$(dirname "$path")
video=$(basename "$path")
ext="${video#*.}"
no_ext="${video%.*}"

default_crf=10
if [ -z "$crf" ]; then
    crf=$default_crf
fi

outname="${no_ext}_crf_${crf}_rot_${angle}.${ext}"
cd "$folder"

# debugging
#echo pwd=$(pwd)
#echo folder="$folder"
#echo video="$video"
#echo ext="$ext"
#echo no_ext="$no_ext"
#echo outname="$outname"

ffmpeg -i "$video" -crf "$crf" -vf "rotate=$angle*PI/180" "$outname"
