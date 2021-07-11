# Person segmentation service
This is my instance segmentation service basen on MaskRCNN model trained on COCO dataset for person class only.

Service is available [here](https://person-segmentation-j6ql7uq6xa-ez.a.run.app/predict). On the page you should choose image file with persons and press 'Submit' button. After awhile you'll see segmented image.

**Technical description:**
1. Model: MaskRCNN based on resnet101;
2. Training dataset: COCO-2017;
3. Library: GluonCV;
4. Web framework: FastAPI;
5. Cloud service: Google Cloud Platform.
