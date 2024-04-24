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



