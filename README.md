# Churn Prediction

# Introduction and Summary of the Project
Churn is a metric that shows customers who stop doing business with the company and by following this metric most of the business could try to understand the reason behind churn numbers and make strategies to tackle the customer churn. Most of the companies focus largely on customer acquisition than on customer retention. This however, can cost companies five times more to attract a new customer than it does to retain an existing one.The reasons that could lead to the customers leaving the service with the company can be numerous but it is of high importance for the companies to identify these factors and take actions quickly and efficiently prior to customer leaving the bank. Increasing customer retention can lead to a significant increase in the profits of the company.

Therefore the main objective of this project is to work on the anonymous bank customer churn dataset obtained from Kaggle and identify and visualize which factors contribute to the customer churn from bank as well as to build a prediction model that will perform the classification task and classify whether a customer is going to churn or not and based on the model performance, choose a model that will attach a probability to the churn to make it easier for the customer service to target low hanging fruits in their efforts to prevent churn. This dataset contains total 1000 customers and 14 attributes related to their demographics,credit score, bank balance, loaction, time stayed with the bank, number of products purchased and whether they have credit card or not and whether they are active members or not. The target variable is Exited feature.

There are six continuous variables - CreditScore,Age,Tenure,Balance,NumOfProducts and EstimatedSalary(where balance, credit score and estimated salary are in US dollars and Tenure and Age in years)

There are five categorical variables - Geography, Gender, HasCrCard,IsActiveMember and Exited

The analysis have been performed in the below three steps:

# -Exploratory Analysis

# -Data Preprocessing and Preparation

# -Training,tuning and evaluating machine learning models

# Exploratory Analysis:

In order to identify patterns that can yield to customer churn, exploratory analysis was performed on the dataset. The bar graphs, histograms and boxplots were made and some useful insights were drawn and explained in the notebook below. I created two new attributes Age_Type and Tenure_Group by bucketing the data of age groups into 18-34(Young Adults), 35-54(Mid-Age),55-71(Seniors 1) and 72-92(Seniors 2) and for tenure groups into 0-2,3-5,6-8,9-10 in order to identify the bucket of customers leaving the bank in Age and Tenure. This gives insights into the type of customers which are currently existing with the bank and which have higher likelyhood of churning.

# Data Preprocessing and Preparation:

The missing data was found in CreditScore, Balance and EstimatedSalary attributes. These missing data was replaced with the mean of the column using Simple Imputer, in oreder to have a clean dataset for churn prediction.

The variables such as RowNumber, CustomerID, Surname did not have any predictive power as these were unuique to the customers and therefore were removed from the analysis. Similarly the additional attributes Age_Type and Tenure_Group which were created for exploratory analysis were also removed since they did not have any predictive power and Age and Tenure were already present in the data in the form of continuous variables, so to avoid the duplicacy of the data, these new attributes were removed.

Since the data was imbalanced because out of total entries, 7963 customers were with the bank and 2031 customers churned, I have balanced the data using random over sampling through imblearn library and adding new samples in the minority class to make more accurate predictions. This prevents getting pretty high accuracy just by predicting the majority class and failing to capture the minority class.

The categorical variables were dummyfied using the label encoding and gone hot encoding methods. The correlation matrix was formed in order to understand if there are any variables which are high colliear. From the matrix it was observed that there was no collinearity issues in the dataset as no two variables were highly correlated because there correlation coefficients were not 0.95 and above. Similarly, I did not observe any variable which was highly collinear with the target variable.

While exploring the data it was observed that there were some outliers in the CreditScore and Age attributes, so these outliers were removed from the training dataset for better predictions.

The training and testing data was then standardised to ensure that each input variable has the same range so that the effect on the model is similar and the variable such as Estimated Salary that has greater ranges does not have larger influence on the model's results

After this, the feature selection process was performed using Lasso and RFE methods to select the predictors which play a significant role in explaining the variation between X and Y better. The features that came out to be less useful for the prediction purpose were "CreditScore","Tenure","HasCrCard","EstimatedSalary","Geography_Spain","Geography_France" as based on their RFE rankings and Lasso cofficients(zero). Therefore, for the modelling purpose these useless predictors were removed from the analysis.

# Training,tuning and evaluating machine learning models:

For the churn problem, the ideal metric to be used is Recall. Recall answers the following question: what proportion of actual positives was identified correctly? In that case, Recall measures the percentage of churns that were correctly classified out of the total churns, which is what I have looked for to analyze the performance of ML classifiers. As an example, consider a re-engagement campaign which provides homes loans and higher returns on customer savings. We'd likely want to ensure that the precision for this offering is high. In other words, we would want to minimize the number of happy users who will receive the offering, and instead have this offering hitting almost exclusively users in danger of churning.

I applied three different ML algorithms (Logistic Regression, Random Forest Classifier and Gradient Boosting Classifier) to analyze and compare the Recall, Precision, Accuracy and F1 scores obtained by each of them.To improve the overall performance of the model, I tuned the algorithms hyperparameters using GridSearchCV.

Models were built on the selected features and after the application of various models, it was found that Gradient Boosting Model generated the highest accuracy score of approx. 87% with optimal hyperparameters as compared to other models.It also has a highest recall of 92% and precision rate of 85%. This illustrates that GBT model correctly identifies if the curstomers are churning or not, when the customers are actually churning from the bank with 92% probability and with a precision of 85%. In terms of false positives, the rate is lowest in GBT model which is 16% as compared to Random Forest Model with 17% and Logistic Regression Model with 28%. This illustrates that only 16% of the satisfied customers can be incorrectly predicted as churning from GBT model making the model efficient.(These results can be seen in the heatmap matrix shown in this notebook in the modelling section).

The GBT model also highlighted three most important features when it comes to churn prediction of customers in a bank. Balance plays an important role, where if the balance in customer's account is below a certain threshold, there is a possibility that the customer may churn in the near future. Similarly, Age is the second important feature so the Middle Age people are more prone to churning due to high appetite for risk due to higher earnings and better living standards that other age groups and they may get better opportunities to invest their money or have premium services elsewhere.Lastly, the number of products also plays an important role in customer churning and the lesser the quality products and services are offered more will be the chances of customer being unsatisfied and leaving the bank.

# Business Benefit: 
The Gradient Boosting Model built in this project would proof very helpful to the bank as they will be able to predict the churning of customers with 88% accuracy, 85% precision and 92% recall which means that algorithm is efficient in identifying the customers that are actually churning with the probability of 92%. This would ensure that the company is able to build strategies on retaining the existing customers by offering better interest rates, varied products and services based on the customer requirements and different demographics. More customers being retained by the bank means higher profitability for the firm and more loyalty points added to their reputation of the bank in the market.
