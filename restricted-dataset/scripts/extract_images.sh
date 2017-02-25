#!/bin/sh

PREFIX="$1"

if [ -z "$PREFIX" ]; then
    PREFIX="."
fi

run() {
    echo "$@"
    $@
}

run mkdir -p $PREFIX/val_images
run mkdir -p $PREFIX/test_images
run mkdir -p $PREFIX/train_images

run unzip -j $PREFIX/val2014.zip -d $PREFIX/val_images $(cat $PREFIX/val_images_select.txt | xargs)
run unzip -j $PREFIX/test2015.zip -d $PREFIX/test_images $(cat $PREFIX/test_images_select.txt | xargs)
run unzip -j $PREFIX/train2014.zip -d $PREFIX/train_images $(cat $PREFIX/train_images_select.txt | xargs)

