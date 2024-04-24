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
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/Telecommuting.png)
Jobs postings that allow for telecommuting are almost twice as likely to be fraudulent.
### Company_Logo vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/Company%20Logo.png)
We can see that jobs postings that have no company logo accounts for **16% of being fraudulent**, which **has higher likelihood** than jobs postings with company logo to be fraudulent.
### Has_Questions vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/Questions.png)
We can infer that jobs postings **without questions has higher likelihood** than jobs postings that have questions to be fraudulent.
## Categorical Variables vs Fraudulent
### Employment Type vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/Employment%20type.png)
we can see that **part-time job postings** has the highest chance of being a fraudulent job posting.
### Required_Experience vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/required%20experience.png)
From here, we can see that **executive jobs postings has the highest likelihood** of being a fraudulent job posting, followed by **entry level job postings.**
### Required_Education vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/required%20education.png)
From this, we can see that majority of the job postings that are fraudulent only require for the applicant to have **some high school coursework** as their required education, as compared to needing some kind of certification or any higher education.
### Industry vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/industry.png)
From this, we can see that job postings in the industry of **ranching** **are 100% fraudulent**, followed by **50% **of job postings in the **military** industry which are fraudulent.
### Function vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/Function.png)
This shows that **administrative job postings** account for **17%** to be fraudulent. It has highest chance of being fraudulent compared to other functions.
### Country vs Fraudulent
![image](https://github.com/zImmortal333/SC1015_WWH_Mini_Project/blob/Images/Country.png)
From this, we can see that **Malaysia** has the highest number of fraudulent job postings among all countries.
## Conclusion from the Exploratory Data Analysis
Required_Experience is removed because most of the data falls under the 'not applicable' category, which means no requirement for experience. We also dropped Industry and Function because there are a noticeable amount of missing values.
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

### Word Cloud of texts commonly used in non-fraudulent job postings

From this, we can conclude that **"product", "service" and "customer"** are likely to appear in non-fraudulent job posting texts.
**"Client","team","looking"** are likely to appear in fraudulent job postings texts.
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
### 2. MultinomialNb
MultinomialNB uses NLP to calculate the probability of each class given the observed feature values for text classification tasks 
### 3. Support Vector Classification (SVC)
How it works:
1. Plot each data point in an n-dimensional space based on the number of features extracted
2. Value of each feature is the value of a particular coordinate. These come from the values generated by the vectorizer
3. Classify by finding the hyperplane that clearly differentiates the two classes
## Analysis of textual classification using three models
The accuracy and precision of three models can be evaluated and visualized through the confusion matrix. Using the models, we see that most of the non-fraudulent datas are predicted as non-fraudulent. There are nearly o non-fraudulent data being predicted as fraudulent.

However, there are quite a few false negatives. It is better to classify a non-fraudulent job as fraudulent than classifying a fraudulent job as non-fraudulent.

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

## Analysis of categorical classification using the random forest classifier

# Conclusion

# Takeaways
**Yee Hong**: I'm in charge of exploring random forest classifier. It helps me understand how to handle non-linear relationships and feature interactions using multiple variables. 

**Waylen**: I learnt that if one model fails to capture certain patterns or makes incorrect assumptions, other models can compensate. Execution of multiple models will potentially lead to more accurate predictions overall. 

**Wei Zhang**: The most interesting machine learning model I have learnt is Support Vector Classification due to its ability to find the optimal hyperplane that maximizes the margin between different classes in high-dimensional spaces.

# References
