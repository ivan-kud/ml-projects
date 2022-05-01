# My works of Machine Learning

## Hierarchical classification
A lot of new products arrive in the marketplace every day and each of them must be assigned to a specific category in the category tree (there are more than 1000 categories). It takes a lot of effort and time, so I want to learn how to predict a category based on short description and parameters of products.

**Technical description:**
1. Approaches: Local Classifier per Node (LCN) and Local Classifier per Parent Node (LCPN);
2. Base model: Random Forest;
3. Dataset: 354316 instances;
4. Libraries: hiclass, scikit-learn, networkx.

## Person segmentation service
This is my instance segmentation service person class basen on MaskRCNN model.

**Technical description:**
1. Model: MaskRCNN based on resnet101 and pretrained on COCO-2017 dataset;
3. Library: gluoncv;
4. Web framework: FastAPI;
5. Cloud service: Google Cloud Platform. 

## Car price prediction. Case 2
To identify cars whose price is below market, we should predict their price by characteristics, description and image. Dataset consists of tabular data (6682 rows into train set and 1671 rows into test set) and images.

**Work contents**
1. EDA.
2. Naive model implementation for tabular data.
3. Model implementation for tabular data based on classic ML algorithm.
4. MLP model implementation for tabular data.
5. NLP model implementation for text data.
6. CV model implementation for image data.
7. Multi-input NN model implementation (tabular + NLP + CV).

## Image classification
Pretrained EfficientNet model to classify car model by image. Dataset consists of 15561 images into train set and 6675 images into test set.

## George image classification
Classification of Saint George the Victorious by image.

## Recommender system
Collaborative filtering model to predict goods rating for user-item pairs. Dataset consists of 857895 rows into train set and 285965 rows into test set.

## Car price prediction. Case 1
To identify cars whose price is below market, we should predict their price by characteristics. There is no training sample with correct answers. It must be found from external sources. The model is a stack of RandomForestRegressor and GradientBoostingRegressor with LinearRegressor meta-model.

## Credit scoring problem
Logistic regression model to predict customer default.

## Restaurant rating prediction
Random forest model to predict restaurant rating.

## Exploratory data analysis
Exploratory data analysis of the dataset.  To track the impact of the living conditions of schoolchildren aged 15 to 22 on their math performance to identify students at risk at an early stage.
