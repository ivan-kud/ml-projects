# My DataScience Projects
Here you can see my educational projects of DataScience.
## Person segmentation service
This is my instance segmentation service basen on MaskRCNN model trained on COCO dataset for person class only.

Service is available [here](https://person-segmentation-j6ql7uq6xa-ez.a.run.app/predict). On the page you should choose image file with persons and press 'Submit' button. After awhile you'll see segmented image.

**Technical description:**
1. Model: MaskRCNN based on resnet101;
2. Training dataset: COCO-2017;
3. Library: GluonCV;
4. Web framework: FastAPI;
5. Cloud service: Google Cloud Platform. 

## Car price prediction. Case 2
**Description:** Prediction of car price by it's characteristics, description and image. Dataset consists of tabular data (6682 rows into train set and 1671 rows into test set) and images.

**Goal:** To identify cars whose price is below market.

**Work contents**
1. EDA.
2. Naive model implementation for tabular data.
3. Model implementation for tabular data based on classic ML algorithm.
4. MLP model implementation for tabular data.
5. NLP model implementation for text data.
6. CV model implementation for image data.
7. NN model implementation for tabular, text and image data based on MLP, NLP and CV models.

## Image classification
**Description:** Pretrained EfficientNet model to classify car model by image. Dataset consists of 15561 images into train set and 6675 images into test set.

## George image classification
**Description:** Classification of Saint George the Victorious by image.

## Recommender system
**Description:** Collaborative filtering model to predict goods rating for user-item pairs. Dataset consists of 857895 rows into train set and 285965 rows into test set.

**Goal:** To recommend the best product for the user.

## Car price prediction. Case 1
**Description:**  Prediction of car prices by their characteristics. There is no training sample with correct answers. It must be found from external sources. The model is a stack of RandomForestRegressor and GradientBoostingRegressor with LinearRegressor meta-model.

**Goal:** To identify cars whose price is below market.

## Credit scoring problem
**Description:** Logistic regression model to predict customer default.

**Goal:** To identify the risk of customer default.

## Restaurant rating prediction
**Description:** Random forest model to predict restaurant rating.

**Goal:** To identify restaurants that inflate their ratings.

## Exploratory data analysis
**Description:** Exploratory data analysis of the dataset.

**Goal:** To track the impact of the living conditions of schoolchildren aged 15 to 22 on their math performance to identify students at risk at an early stage.
