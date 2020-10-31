# Arvato-Customer-Segmentation
_Customer segmentation report using machine learning for arvato financials_

- Blog Post: [Medium](https://aakriti01as.medium.com/arvato-bertelsmann-customer-segmentation-5ff58d891e12)
- Kaggle Leaderboard: [Kaggle](https://www.kaggle.com/c/udacity-arvato-identify-customers/leaderboard)

## Problem Statment
The goal of this project is to help a mail-order sales company in Germany to identify segments of the general population to target with their marketing to grow. The company has provided us with the demographic data of their current customers and the general population. We’ve to build a customer-segmentation machine learning model for the company, which correctly categorizes customers into groups and identifies the customers that the company should target. 
The project is divided into 3 parts:
- **Data Explorationg and Cleaning**
- **Unsupervised learning**: Grouping the population into clusters using K-means Clustering algorithm
- **Supervised learning**: Predicting the response of individual customers towards the marketing campaign using classification algorithms and make a submission to the kaggle competition.

## Datasets and Inputs

There are four data files associated with this project:
- Udacity_AZDIAS_052018.csv: Demographics data for the general population of
Germany; 891 211 persons (rows) x 366 features (columns).
- Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order
company; 191 652 persons (rows) x 369 features (columns).
- Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were
targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
- Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were
targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

The data required for the project has been provided by Bertelsmann Arvato Analytics for the completion of Machine Learning Nanodegree capstone project. The data can be accessed from udacity platform only.

## Installation
Besided Anaconda Python 3.7 distribution, following libraries will require installement:
- ```LGBM```
- ```XGBoost```

## Authors
[Aakriti Sharma](https://github.com/itirkaa)

## License
[MIT](https://choosealicense.com/licenses/mit/)

## References
- Dirk Van den Poel, 2003[Predicting Mail-Order Repeat Buying: Which Variables Matter?](http://ebc.ie.nthu.edu.tw/km/MI/crm/papper/wp_03_191.pdf)
- Selva Prabhakaran, [Principal Component Analysis[PCA] – better explained](https://www.machinelearningplus.com/machine-learning/principal-components-analysis-pca-better-explained/)
- [Analytics Vidya](https://www.analyticsvidhya.com/blog/2020/10/a-simple-explanation-of-k-means-clustering/)
- [LGBM Wikipedia](https://en.wikipedia.org/wiki/LightGBM)
- [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)
