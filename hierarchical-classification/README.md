# Hierarchical classification
## Project description

A lot of new products arrive in the marketplace every day and each of them must be assigned to a specific category in the category tree (there are more than 1000 categories). It takes a lot of effort and time, so I want to learn how to predict a category based on the names and parameters of products.

## Files

Root directory:
* EDA.ipynb - exploratory data analysis
* baseline.ipynb - main file that was used to get the best result
* lvl1_feature_sel.ipynb - some attempts of feature selection (results from this file was not used to get the best prediction)
* lvl1_clf.ipynb - some attempts to train level_1 classifier (results from this file was not used to get the best prediction)
* myutils.py - module with all necessary functions and classes

Working directory (does not present here):
* Files with pickled models, vectorizers, vectorized features and predictions
* stopwords-ru.txt - downloaded from https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt

## Approaches used

LCN (Local Classifier per Node) and LCPN (Local Classifier per Parent Node) approaches are applied in this hierarchical classification project. Random forest is used as a base classifier for LCN and LCPN.

## Results

Below is the comparative results for various models and parameters. The metric is weighted hierarchical F1 score (whF):
* LCPN, 3-symbol vectorizer, 20 estimators

    * train whF: 0.99929
    * valid whF: 0.95954

* LCPN, 3-symbol vectorizer, 1200 estimators

    * train whF: 1.00000
    * valid whF: 0.96215

* LCPN, 2-symbol vectorizer, 20 estimators

    * train whF: 0.99932
    * valid whF: 0.95847

* LCN, 3-symbol vectorizer, 20 estimators

    * train whF: 0.99985
    * valid whF: 0.96605

* LCN, 2-symbol vectorizer, 10 estimators

    * train whF: 0.99937
    * valid whF: 0.96389

* LCN, 2-symbol vectorizer, 20 estimators

    * train whF: 0.99986
    * valid whF: 0.96601

* LCN, 2-symbol vectorizer, 40 estimators

    * train whF: 0.99999
    * valid whF: 0.96703

## What's next

To improve the results, we can also use:
* 'short_description' and 'name_value_characteristics' features
* Other approaches for vectorization
* Feature selection
* Hyperparameters tuning
* Various base estimators for LCN and LCPN
* Unique estimators for each hierarchy level or for each node
* Error analysis

## References

* Carlos N. Silla Jr., Alex A. Freitas. A Survey of Hierarchical Classification Across Different Application Domains. *Data Mining and Knowledge Discovery. 2011.*
* Fábio M. Miranda, Niklas Köehnecke, Bernhard Y. Renard. HiClass: a Python library for local hierarchical classification compatible with scikit-learn. *2021.*
* Liran Shen, Meng Joo Er, Qingbo Yin. The classification for High-dimension low-sample size data. *2020.*
* Noa Weiss. The Hitchhiker’s Guide to Hierarchical Classification. *Towards Data Science. 2019.*
* Noa Weiss. Hierarchical Classification with Local Classifiers: Down the Rabbit Hole. *Towards Data Science. 2020.*
* Noa Weiss. Hierarchical Classification by Local Classifiers: Your Must-Know Tweaks & Tricks. *Towards Data Science. 2020.*
* Noa Weiss. Hierarchical Performance Metrics and Where to Find Them. *Towards Data Science. 2020.*
