# RentHop Projct Repo for CSP571
#### Star using RProject

## Folders:
### 1.Feature Engineering
        a. read_data.R contains the code to read the json and transform it to the CSVs ready to be consumed by the models.
        b. read_test.R reads test.json
        c. Dan_description_feature_extraction.R processed and cleaned data for doing feature engineering on descriptions, output stored into          /processed_data/train_description_tfidf_new.csv
        d. Dan_description_EDA.R did the feature selection and sentimental test on train_description.csv which is cleaned by us
        e. Dan_price_EDA.R did the EDA on price
        f. enhanced_feature.R processed the geographic data and stored the cleaned data into /processed_data/baseline11_v2.csv
        g. json_to_DF.R transformed json file to dataframe
 ### 2.Models
        a. baseline_models folder contains baseline_EDA.R gives all the plots stored in Plots folder, and baseline_accuracy.R predicted the accurancy based on our baseline model
        b. DD_model_DT_RF.R did the Decision Tree Algorithm
        c. MoreFeatures_LR.R did the Logistic Regression
        d. RF_forKaggle.R and randomForestdid the RandomForest Algorithm for Kaggle submission
        e. RP_XGB_Knn.R and XGB_4Kaggle.R did the XGBoost and KNN for Kaggle submission
 ### 3.Plots
        contains all the plots we have generated
 ### 4.Processed Data
        contains all the ready to use transform datasets
        train_moreFeat36.csv and test_moreFeat36.csv were used as Kaggle submission

## To Run:
Please set the working directory to ./Models


