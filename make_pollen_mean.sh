#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/skong/LargeScalePollenProject
DATA=/home/skong/LargeScalePollenProject
TOOLS=/home/skong/caffeCustom/build/tools

$TOOLS/compute_image_mean $EXAMPLE/train_lmdb \
  $DATA/pollen_mean_withWidth.binaryproto

echo "Done."
