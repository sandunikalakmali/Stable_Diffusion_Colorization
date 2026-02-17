#!/usr/bin/env bash

# Extract training images
mkdir -p imagenet/train
tar -xf ILSVRC2012_img_train.tar -C imagenet/train
cd imagenet/train
for f in *.tar; do
  d="${f%.tar}"
  mkdir "$d"
  tar -xf "$f" -C "$d"
  rm "$f"
done
cd ../../

# Extract validation images
mkdir -p imagenet/val
tar -xf ILSVRC2012_img_val.tar -C imagenet/val
