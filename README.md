# Epileptic-Seizure-Recognition
Our project was based on the dataset of epileptic seizure recognition.

Objective:
Our aim is to create a machine learning model that can most accurately classify eeg data under epileptic or non epileptic brain activity using EEG readings. 
Our Machine learning problem is Classification, classifying whether the 178 datapoints in that 1 second are detected as an epileptic seizure activity.

Data structuring/cleaning:
For the first part of our project, we had to do some minor data structuring and cleaning. As the bulk of the data in the dataset are already EEG readings, we only removed the first column of the dataset as it was not used in building our models or for exploratory analysis.
We also split up the dataset based on different classes. This allows us to perform various exploratory analysis on the different classes of EEG readings to further understand the data. This helped us visualise the difference between epileptic seizure brain activity vs non-epileptic brain activity.
We Conducted our project with different subsets of the data, using 2 classes of data each time. Class 1, which are EEG readings of epileptic seizure plus another nonepileptic seizure data.
This allows us to identify which type of brain activity is optimal in helping to build a model to correctly classify seizure activity.
Reduce biases as the total number of non-epileptic data is much more than the epileptic data provided by the dataset. 



Exploratory analysis:
In our exploratory analysis, we found out that there are 11500 rows, 179 columns are most importantly, there are no missing values.
In the dataset, X1-X178 represents the EEG recording at a different point in time while y contains the category of the 178-dimensional input vector.
The number of instances for each category are the same, with non-epileptic seizure data: epileptic seizure data having a ratio of 4 :1
After plotting histograms for all 178 variables (X1-X178), we can observe that the EEG values are normally distributed.
And when plotting boxplots for all the attributes (X1-X178), it seems like there are a lot of outliers.
However, these are maximum and minimum values of the recorded voltages and they represent relevant information to determine the degree of seizure.
Furthermore, these data points are well distributed within the curve. Thus, they should not be treated as real outliers.
We then proceed to plot line graphs using mean values and you can see that class 1 has much larger fluctuations compared to other classes.
When we plot the line graphs using standard deviation, there is a huge disparity between std dev of the EEG values for class 1 and the other classes, clearly distinguishing the seizure class from the rest. 
For the next line graph, we used maximum values. We can see that the max EEG values for class 1 and 2 are much higher than the other classes Interestingly, although both class 1 and 2 had similar peak maximum EEG values of just above 2000microvolts, class 2 showed greater fluctuations than class 1.
We went on to plot the line graph using minimum values. It is clear that class 1 has the lowest minimum EEG value. Although the minimum value for class 2 is not as low as class 1, it showed greater fluctuations than all the other classes.


Models:
We experimented with multiple machine learning models, such as the decision tree, random forest and neural networks.
For our hypothesis, we think that the model of neural network will be the most accurate model in detecting and correctly classifying epileptic seizure activity based on EEG readings.

Decision Tree:
The first model we did was decision tree, it analyses EEG readings from a training dataset, which a model is then created. This model is trained to classify EEG readings under 2 different groups, epileptic or nonepileptic. We used this model as it was intuitive and simple to understand. However, the accuracy of our decision tree model is only about 92-95%.

Random Forest:
It is made up of multiple decision trees bagged together, hence the term "forest".
Random forest uses random sampling in creating individual decision trees. Additionally, random features are considered when splitting nodes within the trees. The final classification is
derived from averaging out each individual tree.

It is similar to a decision tree, but is able to yield a more accurate model compared to a singular decision tree, as our results show us, yielding an accuracy of 97-99%.
Overall, the dataframe containing class1 and class 3 EEG readings produce the best model to classify epileptic seizure activities in both decision trees and random forest models with high accuracy.


Neural Network:
From the models mentioned previously, data from class 2 and class 3 concatenated with the seizure data eventually led to the building of a better model with higher test accuracy, and thus we will be exploring new models with these two classes. 
Model 1
Exploring multilayer perceptron model, It is a feedforward neural network model which maps sets of input data onto a set of outputs by going through many layers of nodes in a directed graph. However neural network models have a tendency to overfit, but this can be resolved with adding Gaussian Dropout layer, a regularisation technique: Generates noise during training for better generalisation of data and prevents the issue of overfitting. This shrank the gap between our test and train accuracy and generated a test accuracy of 0.95%, which is almost the same as decision tree models.
Model 2
Thus we have decided to build a better neural network model more suited to our time series sequential data. Presenting our Long Short-term memory network model.  They are a type of recurrent neural network that are able to learn and remember over long sequences of input data which is appropriate for our project. The model does not require domain expertise for data preparation and can learn to extract features from sequences of observations. They can store information about previous values and exploit the time dependencies between our datapoints. The results are the best among all the models so far, producing a good 98-99% test accuracy!
LSTM neural network model is the best at classification of EEG data. However the disadvantage in training a long short-term memory neural network is that it takes a long time and takes up a huge memory space, which are factors to be considered when choosing a model. 

K fold cross validation:
Our group has found out that the simple train/test split to evaluate our models may not be the least biased method to evaluate model effectiveness.
Thus we also used k-fold cross validation, a resampling procedure. It results in a less biased & less optimistic estimate of the model effectiveness. Due to the large dataset that we are dealing with, we have decided to use 10-fold cross validation as it has been shown to yield test error rate estimates that are not bias. 
We computed the mean validation scores across 10 folds and these are our results. The neural network model achieved the highest score and is proven once again to be the best model in classification of this dataset. 

Conclusion: 
Using our random forest model, we also computed the feature importance of every datapoint X1-X178 when considering if we should conduct feature extractions. Ultimately we came to a conclusion that all of the features should be equally important and thus included in training of the models, since they are all EEG measurements. Theoretically, there isn't any single one millisecond that is more important or predictive than any other millisecond in terms of determining whether a patient is having an epileptic seizure or not.
Lastly, We discovered that epileptic seizure eeg readings have higher fluctuations compared to normal brain activity readings. With that in mind, we were able to build models that could classify epileptic and non-epileptic activities using those eeg readings and achieve our initial objective. LSTM Neural Network model led to the most accurate detection of epileptic and non epileptic activities, and the best variable in building an accurate model is Class 3 data as it is consistently shown the highest test accuracy and validation score through all our models compared to other classes. 

