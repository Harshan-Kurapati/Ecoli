**Abstract**

The dataset ecoli.csv will be used to predict whether the has cell communication or not based on the expression level and gene function. The dataset contains 1500 data points that were formed by micro-array expression data and functional data of 1500 genes of E. coli. E. coli is a bacterium that is commonly found in the lower intestine of warm-blooded organisms. In total the dataset consists of 107 columns, in which the first 103 columns are the numerical features describing the expression level and the next 3 columns describe the gene function. The last column is the target column. In this column, the positive class is denoted as 1 which represents the gene that has the function of cell communication, and the negative class is denoted as 0 represents there is no cell communication for that gene. 

1	Pre-processing Techniques 
Pre-processing is a technique that is used to transform the raw data into a useful and efficient format. 

**Outlier detection**

An outlier is a data object that behaves differently and dramatically deviates from the other data objects. To overcome this type of problem outliers are to be identified and handled. So, to identify the outliers there are different methods which are density-based technique, model-based technique, distance-based technique, cluster-based technique, and isolation-based technique. Working individually with all these techniques and the method that is more accurate will be selected. 

**Normalization** 

The process of arranging relational data in accordance with a variety of normal forms to reduce data redundancy and increase data integrity is known as normalisation. Since the data ranges from 781.5653 to -498.29 and the mean was nearly zero, normalisation is advised. Initially, min-max normalisation for this project was chosen. later with the aid of cross-validation and choose the one that produces a decent result.

**Imputation**

It is observed from the provided dataset, that ecoli.csv contains a significant number of missing or NA values. These values should be handled to work with the dataset. There were numerous methods for handling the NA values.
Deleting the rows that consist of missing values, is not an ideal process for handling the data because it could lead to a loss of information. 
Imputation of the missing numerical value by the average of the features of all values. This method will give good results compared to the earlier method but in some cases, it can be biased. 
Imputation of the missing numerical value by the average of the feature’s class-specific values will help to solve the bias situation. 

**Cross-validation**

The simplest and most crucial type of cross-validation technique is the k-fold approach. Following a random data shuffle, K-Unique Datasets are formed, often with an equal number of samples. The model is created iteratively on any (K-1) Datasets (Folds), verifying it on the dataset that was excluded, and maintaining track of the previous evaluation metrics of the built models. By considering the evaluation measures, it can determine which methodology was successful, allowing one to choose the best approach.

**Applying cross-validation to the current data**

The data is first divided into training and testing sets, with the test set being placed aside while the training data is worked on. The training dataset will be partitioned into k files at random, each containing roughly the same number of instances. One of the k folders will serve as our validation at each stage, while the other k folders will serve as training data. the validation score is calculated for each fold using various k-values, with the validation set as the ground truth values. The k with the best-averaged evaluation outcomes by CV is selected. 
2	Classification Techniques
Based on a training set of data that includes observations and whose category membership is known, classification is the problem of determining which of a set of categories, a new observation belongs to. 

**Decision tree**

When constructing a decision tree for the data, the dataset is to be split into train-test. Now consider two variables Dv and Av, where Dv represents the whole training data and Av represents the list of all the features. Now find the initial splitting feature or root node it can be anyone from the which helps in selecting the optimal splitting feature they are information gain, gain ratio and Gini index. For the information gain and gain ratio, the score should be the highest for a feature to be an optimal splitting feature whereas in the case of the Gini index It should be the lowest. After selecting the splitting feature, now consider the stopping criteria.
For the stopping criteria, the following conditions are to be followed. all examples in Dv have the same values on Av, and all examples in Dv belong to the same class C, if Dv is empty and if Av is empty. If any of the mentioned conditions satisfy, then stop growing the tree and mark the non-root node a leaf node leading to the decision of the majority class in Dv here in the dataset class feature is the target col107. Else continue splitting using the optimal splitting metrics. To overcome the problem of overfitting there is a technique called pruning. Pruning is a data compression technique that helps to reduce the size of the decision tree by removing the sections of the tree that are redundant to classifying instances.  

**Random forest**

The random forest classifier is the combination of multiple decision trees it is because it is known that decision trees are highly sensitive to training data which results in high variance. To perform random forest, build sample subsets from the dataset which is ecoli.csv then randomly sample k features among all the features. Now split the node by selecting the optimal splitting among the sampled k features. repeat these for many subsets and finally, select the class that has majority voting. In simple words, create multiple decision trees from different subsets of the dataset and during the prediction pass the value through all the decision trees and select the class with the majority voting since it is a classification problem. In this way, it reduces the overfitting problem in decision trees and reduces the variance and therefore improves the accuracy.

**Naive Bayes**

The naive Bayes classifier was named following the naive attribute conditional independence. It uses the Bayesian method to predict the output based on input features. To predict the class given input features will be the product of the probability of one class with the probability of all the features with the same class and repeat this with all the classes. Then the probability scores that were obtained from those the class that gives the highest probability score will be selected.

**Ensemble**

Ensemble methods is a machine learning technique that combines several base models to produce one optimal predictive model. This means the predictions were drawn from different models and the majority class will be considered as an output value. If it is a regression problem, then consider the mean of all the results as the result. This way it will be more generalised overcome the problem of overfitting and will be more accurate.
3	Evaluating the model with cross-validation
To evaluate the model that gives high accuracy the dataset will be randomly divided into k folders with an almost equal number of instances in each folder. Data can be randomly selected in each fold or stratified. All folds are used to train the model except one, which is used for validation. That validation fold should be rotated until all folds have become validation folds once and only once. Now the different validation folds result in different scores. Now repeat this with other models then a set of validation scores for each model will be obtained. From these scores, the performance of each model can be evaluated and the model that gives more accurate results can be selected. This way model can be evaluated using cross-validation.

**Metric to measure the classification performance**

There are several metrics to measure the performance of a classifier they are Accuracy, Confusion matrix, AUC/ROC (Receiver Operating Characteristics), Precision, Recall, F1 score, kappa, and MCC (Matthews Correlation Coefficient). Now consider which metric to select among accuracy, precision, recall and F1 score. Looking at the distribution of our dataset ecoli.csv it can be observed that the overall dataset contains 1500 data samples in those only 161 samples are of class ‘1’ and the remaining 1339 are of class ‘0’ this clearly shows that the data was imbalanced. Whenever the data is imbalanced accuracy metric may not be the most appropriate metric. Now a very small precision or recall will result in a lower overall score. Thus, the F1 score can help balance the metric across positive/negative samples. Therefore, for this dataset considering the F1 score as the metric will be appropriate.
