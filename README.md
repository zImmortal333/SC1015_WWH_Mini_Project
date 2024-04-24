# SC1015_Mini_Project
# Fraudulent_Job_Postings_Classification
## Pratical Motivation 
The significant loss of $135.7 million to job scams in Singapore during 2023 is a concerning issue, particularly for university students who are soon entering the workforce. Singapore's status as an international business hub attracts job seekers from diverse backgrounds, including these students. However, the prevalence of fraudulent job schemes has left them deeply worried about their prospects. Our objective is to address and mitigate the impact of job scams.
## The Dataset
The dataset of real and fake job postings is from Kaggle - https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction 
## Problem Formulation
Our aim is to build a classifier using machine learning models to indentify fraudulent and non-fraudulent jobs.

After filtering irrelevant variables, we will be using three model to analyze textual data, and a model to analyze categorical data. Among three models that classify textual data, we will be deciding the most effective model using accuracy and time metrics.
## Exploratory Data Analysis
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/Number%20of%20job%20postings.png)
* Not fraudulent: 17014
* Fraudulent: 866

There are noticeably less fraudulent job postings than non-fraudulent ones.
Taking 'fraudulent' as a response variable, we construct a relationship of each variable with the response variable.
## Boolean Variables vs Fraudulent
### Telecommuting vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/Screenshot%202024-04-24%20231155.png)

Jobs postings that allow for telecommuting are almost twice as likely to be fraudulent.
### Company_Logo vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/Company%20Logo.png)
We can see that jobs postings that have no company logo accounts **is 8 times more** than jobs postings with company logo to be fraudulent.
### Has_Questions vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/Questions.png)
We can infer that jobs postings **without questions has higher likelihood** than jobs postings that have questions to be fraudulent.
## Categorical Variables vs Fraudulent
### Employment Type vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/Employment%20type.png)
we can see that **part-time job postings** has the highest chance of being a fraudulent job posting.
### Required_Experience vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/required%20experience.png)
After looking at this bar chart, we felt that there was not much conclusion that we can draw, as it does not really tell us **the chance of a job** posting being fraudulent from the required experience specified.
### Required_Education vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/required%20education.png)
From this, we can see that majority of the job postings that are fraudulent only require for the applicant to have **some high school coursework** as their required education, as compared to needing some kind of certification or any higher education.
### Industry vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/industry.png)
For 'Industry' we could not draw a conclusion on how it makes a job posting fraudulent or not, as it contains **many categories with only 1 data point**. In this example, ranching only contained 1 data point which was fraudulent. Therefore, it may seem that ranching has a 100% chance of being a fraudulent job posting, but that may not be the case.
### Function vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/Function.png)
Similarly for functions, we could not draw much conclusion for how it contributes to telling if a job is fraudulent or not, as the results shown could be misleading.
### Country vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/Country.png)
From this, we can see that **Malaysia** has the highest number of fraudulent job postings among all countries.
## Conclusion on Boolean and Categorical Data

We listed out useful variables and unuseful variables:

**Useful:**
- telecommuting
- has_company_logo
- has_questions
- employment_type
- required_education
- country

**Not useful:**
- required_experience
- Industry
- Function

## Textual Analytic Visualization
We generate Word Cloud to visualize job postings texts.
### Word Cloud of texts commonly used in fraudulent job postings
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/fraud.png)
### Word Cloud of texts commonly used in non-fraudulent job postings
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/non%20fraud.png)

From this, we can conclude that **"product", "service" and "customer"** are likely to appear in fraudulent job posting texts.
**"Client","team","looking"** are likely to appear in non-fraudulent job postings texts.
**"Work" and "experience"** are commonly used in most of job postings texts.

# Machine Learning Models for Textual Data Classification
## Textual Data Cleaning
Steps of cleaning textual data are as followed:
1.  Change texts to lower case
2.  Remove HTML and URLs
3.  Replace non-Alnum
4.  Tokenizes the cleaned text into individual words
5.  Use NLTK Library to remove stopwords from tokenized text
6.  Joins filtered tokens back into a single string

## Textual Analysis Model
Splitting the data into train and test data. The train test split was done using a 70:30 ratio, where 30% of the data was for testing
### 1. Logistic Regression
A statistical method used for binary classification tasks, where the target variable (fraudulent) has only two possible outcomes, typically represented as 0 and 1. 

Cross-validation scores: [0.96271439 0.96569724 0.96681581 0.96644295 0.96457867]
Average cross-validation score: 0.9652498135719612
Test set accuracy: 0.962192393736018
Test set precision: 1.0
Test set recall: 0.27155172413793105
Test set F1 score: 0.42711864406779665
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/LogisticRegression.png)
### Analysis of textual classification using Logistic Regression
The confusion matrix for the Logistic Regression model indicates that it performs well overall, with a high average cross-validation score of approximately 0.965. However, when looking at the test set performance, the accuracy is slightly lower at 0.962, suggesting that the model may not generalize as well to unseen data. The precision for the test set is 1.0, indicating that when the model predicts a job listing as fraudulent, it is almost always correct. However, the recall is relatively low at 0.272, suggesting that the model misses a significant number of actual fraudulent listings. This trade-off between precision and recall is reflected in the F1 score of 0.427, which considers both metrics and indicates a balance between them. Overall, while the model performs well in terms of precision, it may benefit from improvements in recall to better identify fraudulent job postings.
### 2. MultinomialNb
MultinomialNB uses NLP to calculate the probability of each class given the observed feature values for text classification tasks 

Cross-validation scores: [0.9541387  0.95525727 0.95339299 0.9541387  0.95451156]
Average cross-validation score: 0.9542878448918717
Test set accuracy: 0.9512304250559284
Test set precision: 0.7333333333333333
Test set recall: 0.09482758620689655
Test set F1 score: 0.16793893129770993
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/Binomial.png)
### Analysis of textual classification using MultinomialNB
The Multinomial Naive Bayes model achieved a high average cross-validation score of approximately 0.9543, indicating its effectiveness in classifying fraudulent job postings. However, when evaluated on the test set, the model's accuracy was slightly lower at 0.9512. This suggests that the model's performance may vary slightly when applied to unseen data.

The precision of the model on the test set was 0.7333, indicating that when it predicted a job posting as fraudulent, it was correct approximately 73.33% of the time. The recall, or sensitivity, of the model was low at 0.0948, indicating that the model only identified 9.48% of all fraudulent job postings in the test set. This low recall suggests that the model may have difficulty identifying all instances of fraudulent job postings.

The F1 score, which is the harmonic mean of precision and recall, was 0.1679. This indicates a trade-off between precision and recall, as the F1 score balances these two metrics. Overall, while the Multinomial Naive Bayes model performed well in terms of accuracy and cross-validation scores, its performance in terms of precision, recall, and F1 score suggests that it may benefit from further tuning or the use of additional features to improve its ability to identify fraudulent job postings.
### 3. Support Vector Classification (SVC)
How it works:
1. Plot each data point in an n-dimensional space based on the number of features extracted
2. Value of each feature is the value of a particular coordinate. These come from the values generated by the vectorizer
3. Classify by finding the hyperplane that clearly differentiates the two classes

Cross-validation scores: [0.95824012 0.96047726 0.9601044  0.95861298 0.95973154]
Average cross-validation score: 0.9594332587621178
Test set accuracy: 0.9579418344519016
Test set precision: 1.0
Test set recall: 0.1896551724137931
Test set F1 score: 0.3188405797101449
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/SVC.png)
### Analysis of textual classification using SVC
The confusion matrix analysis for Support Vector Classification (SVC) with the provided cross-validation scores and test set metrics shows a high average cross-validation score of approximately 0.9594, indicating that the model performs well across different subsets of the data during training. However, the test set accuracy, which measures the overall correctness of predictions, is slightly lower at around 0.9579, indicating a small drop in performance on unseen data.

The precision score of 1.0 indicates that when the model predicts a fraudulent job posting, it is always correct. However, the recall score of 0.1897 suggests that the model has a relatively low ability to correctly identify fraudulent job postings out of all actual fraudulent postings. This indicates a high rate of false negatives, where actual fraudulent postings are incorrectly classified as non-fraudulent.

The F1 score, which considers both precision and recall, is 0.3188. This score balances the trade-off between precision and recall, providing a single metric to evaluate the model's overall performance. The relatively low F1 score suggests that there is room for improvement in the model's ability to balance precision and recall for fraudulent job posting detection.

# Random Forest Model for Categorical Data Classification
## Categorical Data Cleaning
Steps of cleaning categorical data are as followed:
1. Change categorical data into object data type
2. Replace Nan value with blank space
3. Convert data back to categorical
4. Convert 'Fraudulent' from categorical data into numeric data
   
## Random Forest Classifier
It is a combination of random sampling and decision trees which combines the output of multiple (randomly created) Decision Trees to generate the final output. The predictions are either fraudulent or non-fraudulent and the random forest classifiers eventually takes the average of these predictions to make its prediction.

We splited the data into train and test data. The train test split was done using a 70:30 ratio, where 30% of the data was for testing

Total execution time: 5.5
Test Set Accuracy: 95.62%
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/randomtree.png)
## Analysis of categorical classification using the random forest classifier
A Test accuracy of 95.62% shows that random forest is a relatively accurate model in predidcting Fraudulent postings.

# Conclusion
Through the use of Logistic Regression, Multinomial Naive Bayes, Support Vector Classification, and Random Forest models for predicting fraudulent job postings, a key learning outcome is the importance of model selection and understanding their strengths and limitations. Logistic Regression offers simplicity and interpretability, making it suitable for initial exploratory analysis. Multinomial Naive Bayes, despite its assumption of feature independence, can be effective in text classification tasks. Support Vector Classification, while powerful, requires careful tuning and may be computationally intensive. Random Forest excels in handling categorical data and capturing non-linear relationships. By combining these models, one can leverage their individual strengths to improve the overall predictive performance and gain deeper insights into fraudulent job posting detection strategies.

# What we learn
1. Textual Data Cleaning
2. Logistic Regression
3. MultinomialNb
4. Support Vector Classifier
5. Categorical Data Cleaning
6. Random Forest Classifier

# Takeaways
**Yee Hong**: I'm in charge of exploring random forest classifier. It helps me understand how to handle non-linear relationships and feature interactions using multiple variables. 

**Waylen**: I learnt that if one model fails to capture certain patterns or makes incorrect assumptions, other models can compensate. Execution of multiple models will potentially lead to more accurate predictions overall. 

**Wei Zhang**: The most interesting machine learning model I have learnt is Support Vector Classification due to its ability to find the optimal hyperplane that maximizes the margin between different classes in high-dimensional spaces.

# References
1. https://levity.ai/blog/text-classifiers-in-machine-learning-a-practical-guide
2. https://www.datacamp.com/tutorial/wordcloud-python
3. https://spotintelligence.com/2023/02/22/logistic-regression-text-classification-python/
4. https://stackabuse.com/removing-stop-words-from-strings-in-python/
5. https://github.com/sajjadirn/Text-Classification-using-random-forest-classifier
6. https://levity.ai/blog/text-classifiers-in-machine-learning-a-practical-guide
