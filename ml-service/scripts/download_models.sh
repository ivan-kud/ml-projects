#!/bin/sh
# Download Mask RCNN model
wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar zxvf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
mkdir ./ml-models
mkdir ./ml-models/mask-rcnn-coco
mv ./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb ./ml-models/mask-rcnn-coco/
rm mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
rm -rf ./mask_rcnn_inception_v2_coco_2018_01_28