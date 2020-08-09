# TITANIC : MACHINE LEARNING FROM DISASTER(Kaggle)

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

## WORKFLOW
1. Perform a statistical analysis of the data by looking over each feature's characteristics such as data type of columns, number of instances, correlation of each attribute with the output variable, finding mean and other information about data, correlation matrix etc;. and segregating them to a new DataFrame.
2. Make a list of all machine learning algorithms that can give good prediction results and spot check each one of them (apply each one of them on the dataset) to find which one is better for prediction
3. Make some of the good performing algorithms and perform a grid search/ randomised search over it's hyperparameters to find the optimal hyperparameters for the prediction task. Ensure that the optimal hyperparameters do not overfit the data.

## FIle Description
test.csv - test set provided by kaggle in csv format.

train.csv - train data provided by kaggle in csv format.

Titanic.ipynb - ipython notebook.

### Algorithm Performance
RnadomForest Classifier -- 80.7%

XGBoost                 -- 82.9% 

DecisionTree Classifier --  81.6%

#### References
Titanic: Machine Learning from Disaster (https://www.kaggle.com/c/titanic)
