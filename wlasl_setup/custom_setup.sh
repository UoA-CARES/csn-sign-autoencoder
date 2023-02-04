#!/bin/bash
cd ..

# Set parameters
n_classes=10 # classes with the most samples are favoured to create WLASL_{n_classes}
n_workers=14

# Download the dataset
mkdir -p data
cd data
mkdir -p wlasl
cd wlasl
kaggle datasets download -d sttaseen/wlasl2000-resized
mkdir -p wlasl-uncompressed
unzip wlasl2000-resized.zip -d ./wlasl-uncompressed
cd ../..


# Delete missing clips and create a clean annotations file
python tools/wlasl/create_json.py $n_classes data/wlasl/wlasl-uncompressed/wlasl-complete

# Split the videos
python tools/wlasl/split_videos.py "wlasl_${n_classes}.json" data/wlasl/wlasl-uncompressed/wlasl-complete

# Extra raw frames
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-uncompressed/wlasl-complete/test data/wlasl/rawframes/test --ext mp4 --task rgb --level 1 --num-worker $n_workers --out-format jpg --use-opencv
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-uncompressed/wlasl-complete/train data/wlasl/rawframes/train --ext mp4 --task rgb --level 1 --num-worker $n_workers --out-format jpg --use-opencv
python mmaction2/tools/data/build_rawframes.py data/wlasl/wlasl-uncompressed/wlasl-complete/val data/wlasl/rawframes/val --ext mp4 --task rgb --level 1 --num-worker $n_workers --out-format jpg --use-opencv

# Build the labels
python tools/wlasl/build_labels.py "data/wlasl/wlasl-uncompressed/wlasl-complete/wlasl_${n_classes}.json" data/wlasl


