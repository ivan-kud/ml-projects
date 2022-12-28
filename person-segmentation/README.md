# Person segmentation service
This is my instance segmentation service basen on MaskRCNN model trained on COCO dataset for person class only.

Service is available [here](FIXME). On the page you should choose image file with persons and press 'Submit' button. After awhile you'll see segmented image.

**Technical description:**
1. Model: MaskRCNN based on resnet101;
2. Training dataset: COCO-2017;
3. Library: FIXME;
4. Backend: FastAPI;
5. Frontend: HTML, CSS
5. Cloud service: DigitalOcean (GCP was used before 24.02.2022).
