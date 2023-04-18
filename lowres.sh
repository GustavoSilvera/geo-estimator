#!/bin/bash



IMAGES="dataset/images/"
mkdir -p $IMAGES/lowres
cd $IMAGES

WIDTH=512 # low res
WIDTH=256 # very low res

for f in *.jpg
    ffmpeg -hide_banner -loglevel error -i $f -vf scale="$WIDTH:-1" lowres/$f
done

cd - # back home