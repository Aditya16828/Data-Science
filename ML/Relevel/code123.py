import pandas as pd
import numpy as np

"""### Problem Inroduction and Data Ingestion

##### Objective

To predict the house price from various house attributes present in the data. 

This is a Regression problem.

##### General Steps for a ML workflow (Not an exhaustive or standard list )-

1. Data Ingestion
2. Train Test Split
3. Data Cleaning
4. Feature Extraction
5. EDA
6. One hot encoding
7. Feature Scaling
9. Feature Selection
10. Model Training
11. Model Evaluation

In this session, we will mostly focus on Feature Engineering and Feature scaling aspects of the workflow. 
Model training and evaluation will be covered later in this week.

##### Data Ingestion

Importing house_price csv data file using pandas
"""

house_data=pd.read_csv('house_price.csv')
house_data.head()

house_data.shape

"""### Train/Test Split

Always remember there will always be a chance of data leakage so we need to split the data first and then apply feature
Engineering, especially scaling. 
Data cleaning , Null value treatment can still be done on the entire dataset since there won't be signifincant data leakage.
 But it is always a good practice to treat Test and Train data seperately always.

In the interest of time and fo the sake of simplicity, we will be treating missing values together on full data in this notebook.
"""

# Using 75/25 ratio for train test split in this case

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(house_data.loc[:, house_data.columns != 'SalePrice'],
                                               house_data['SalePrice'],test_size=0.25,random_state=100)

X_train.shape, X_test.shape

"""### Treating Missing Values

##### Categorical Features
"""

# Extracting a list of Categorical fields with missing values
# NOTE : Categorical fields will have data type as 'O', as can be observed in the filter condition below
# O means objects
features_with_missingval =[feature for feature in house_data.columns if house_data[feature].isnull().sum()>=1 and house_data[feature].dtypes=='O']
features_with_missingval

# Printing %s of missing values in all cateogrical features

for feature in features_with_missingval:
    print("{0}: {1}% missing values".format(feature,np.round(house_data[feature].isnull().mean()*100,2)))

house_data[features_with_missingval].isnull().sum()

## Replacing missing values with a new label "Not avaialble"

def replace_missing_lables(df,features):
    dataset=df.copy()
    dataset[features]=dataset[features].fillna('Not Avaialable')
    return dataset

house_data=replace_missing_lables(house_data,features_with_missingval)

house_data[features_with_missingval].isnull().sum()  # Prints the final counts of nulls. We can confirm all to be 0

house_data.head()

"""##### Numerical Features"""

## Similarly, lets check for list of features with numerical values
## For numerical features dtype will not be 'O'

numerical_with_nan=[feature for feature in house_data.columns if house_data[feature].isnull().sum()>=1 and house_data[feature].dtypes!='O']

## Printing the the numerical variables and percentage of missing values

for feature in numerical_with_nan:
    print("{}: {}% missing value".format(feature,np.around(house_data[feature].isnull().mean(),2)))

import matplotlib.pyplot as plt
for col in numerical_with_nan:
  house_data[col].plot(kind = 'box')
  plt.show()

## Imputing missing values with median

for feature in numerical_with_nan:
    ## We will replace by using median since there are outliers
    house_data[feature].fillna(house_data[feature].median(),inplace=True)
    
house_data[numerical_with_nan].isnull().sum()

print(house_data.shape)
house_data.head()

house_data.isnull().sum().sum()
#0 means nowhere missing values

"""### Feature Extraction"""

## The below year-based features are not useful for any model
## Hence we can convert them into Duration by sbstracting from Year sold to determine age

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    house_data[feature]=house_data['YrSold']-house_data[feature]

"""We can see below how year based fetures have been converted to duration"""

house_data[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()

"""##### Handling Rare Categorical Labels inside Features

If a label is occupying less than 1% of the data in a feature, then we can remove that since that wont be adding any value to any model.

Drpping that label can improve the performance of a model.
"""

# Generating list of ctegorical variables
categories=[feature for feature in house_data.columns if house_data[feature].dtype=='O']

categories

house_data.RoofStyle.value_counts()

len(house_data)

ll1 = house_data.groupby('MSZoning')['SalePrice'].count()/len(house_data)
ll1[ll1>0.01].index

# Below code snippet imputes all lables having less than 1% weightage in a data to "Insignificant Label'"

for feature in categories:
    label_percentage=house_data.groupby(feature)['SalePrice'].count()/len(house_data)
    label_more_than_1_percent =label_percentage[label_percentage>0.01].index
    house_data[feature]=np.where(house_data[feature].isin(label_more_than_1_percent),house_data[feature],'Insignificant Label')

house_data.groupby('MSZoning')['SalePrice'].count()/len(house_data)
#label_more_than_1_percent =label_percentage[label_percentage>0.01].index

"""We can confirm that 'Insignificant Label' is a part of categorical features as below - """

house_data.Street.unique()

"""### One-hot encoding

Since machine based algorythms can't understand words, we usually need to convert categorical features into binary ones.

It can be done using pd.dummies() function as below -
"""

dd1 = pd.DataFrame({'cat': ['A', 'A', 'A', 'B', 'C', 'B', 'C', 'A'],
                    'val': [12, 45, 78, 56, 89, 12, 56, 9]})
dd1

pd.get_dummies(dd1['cat'],drop_first=True)

house_data[categories[2]].value_counts()

categories_dummy = pd.DataFrame()
for category in categories:
    categories_dummy = pd.concat([categories_dummy,pd.get_dummies(house_data[category],drop_first=True)],axis=1)
categories_dummy.head()

categories_dummy.shape

"""Once the dummies are created, insignifacnt lables can be dropped as below - """

categories_dummy.drop(['Insignificant Label','Not Avaialable'],axis=1,inplace=True)
categories_dummy

"""Before the dummies are merged with main data, all the categorical fields are dropped since those are no longer required"""

for feature in house_data.columns:
    if house_data[feature].dtypes=='O':
        house_data.drop([feature],axis=1,inplace=True) # inplace= true signifies that the actual object is modified by drop method
house_data

"""Finally, dummies are merged with main data"""

house_data = pd.concat([house_data,categories_dummy],axis=1)
house_data

house_data.shape

"""### Feature Scaling"""

## Always remember there way always be a chance of data leakage so we need to split the data first and then apply feature
## Scaling 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(house_data.loc[:, house_data.columns != 'SalePrice'],
                                               house_data['SalePrice'],test_size=0.25,random_state=100)

X_train.shape, X_test.shape

"""Id is not a usefull feature and Saleprice is our target variable. Hence, those are removed from the data to be scaled"""

l1 = list(categories_dummy.columns)
l1.extend(['Id','SalePrice'])

features_to_be_scaled = [feature for feature in X_train.columns if feature not in l1]
# since along with ID and sale price all the onehot encoded variables must be excluded in scaling

features_to_be_scaled

"""##### Normalisation Scaling"""

from sklearn.preprocessing import MinMaxScaler

scaler_minmax = MinMaxScaler()
X_train_normalised = scaler_minmax.fit_transform(X_train[features_to_be_scaled])
X_test_normalised = scaler_minmax.transform(X_test[features_to_be_scaled])

X_train_normalised

X_test_normalised

print("Min value of X_train is : "+str(X_train_normalised.min()))
print("Max value of X_train is : "+str(X_train_normalised.max()))
print("Min value of X_test is : "+str(X_test_normalised.min()))
print("Max value of X_test is : "+str(X_test_normalised.max()))

"""##### Standardisation Scaling"""

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train_standardised =sc.fit_transform(X_train[features_to_be_scaled])
X_test_standardised =sc.transform(X_test[features_to_be_scaled])
#discuss important difference between transform and fit_transform

X_train_standardised

X_test_standardised

print("Mean of X_train is : "+str(X_train_standardised.mean()))
print("Standard Deviation of X_train is : "+str(X_train_standardised.std()))
print("Mean of X_test is : "+str(X_test_standardised.mean()))
print("Standard Deviation of X_test is : "+str(X_test_standardised.std()))
# discuss why not exact 0,1  in X_test, reason is transform has been applied & not fit_transform

# transform the train and test set, and add on the Id and SalePrice variables
X_train = pd.DataFrame(X_train_normalised)

X_test = pd.DataFrame(X_test_normalised)

X_train

X_test

"""##### Next Steps - Model Training and Evaluation. 

##### Those will be covered in the upcoming sessions since today we are just learning the fundamentals

# Feature Engineering Case 2 |  Titanic Data

### Problem Inroduction and Data Ingestion

##### Objective

To predict whether the given person survived in titanic mishap or not.  

This is a very famous example of Classification problem.

##### Data Ingestion

Titanic classification problem data is a standard dataset used for many mock ML case studies. 
It can be loaded directly from Seaborn library as an inbuilt dataset.
"""

import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic.head()

"""### Feature Extraction & EDA"""

# checking if the data has missing values and if yes, which featues

titanic.info()

"""##### Analysing if Priority Class had any impact on Survival Rate"""

print(titanic[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean())

"""Insight - 

Yes, Passengers having high priority class had a better survival rate

##### Analysing if family Size had any impact on Survival Rate

Family size is not available to us directly in the data. But we have Parent/Children adn Sibling/Spouse feature. By adding all we can get family size which can be a useful feature as below -
"""

# sibsp = no of sibling/spouses aboard
# parch = no of parent/children aboard
titanic['FamilySize'] = titanic['sibsp'] + titanic['parch'] + 1
print(titanic[['FamilySize', 'survived']].groupby(['FamilySize'], as_index=False).mean())

"""Insight - 

There seems to be a pattern here with moderately sized families standing a better survival chance.

##### Analysing if Gender had any impact on Survival Rate
"""

print(titanic[['sex', 'survived']].groupby(['sex'], as_index=False).mean())

"""Insight - 

Yes, Females had a much better survival rate as compared to males.

##### Analysing if Base City of passengers had any impact on Survival Rate
"""

titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])
print (titanic[['embarked', 'survived']].groupby(['embarked'], as_index=False).mean())

"""Insight - 

There appears to be a pattern with Passengers belonging to C cateogry having a much better survival rate

##### Analysing if Fare paid by passengers had any impact on Survival Rate

For simpler interpretation and impacful insights, we can convert this numerical feature Fare into a categorical feature. 

In this way we can analyse the survival rates for various class of fair and their survival rates.
It can be done using pd.qcut function 

This function partitions data into N partiotions based on values.
"""

# Adding a new extracted feature CategoricalFare

titanic['fare'] = titanic['fare'].fillna(titanic['fare'].median())
titanic['CategoricalFare'] = pd.qcut(titanic['fare'], 3)
print (titanic[['CategoricalFare', 'survived']].groupby(['CategoricalFare'], as_index=False).mean())

"""Insight - 

Passengers who paid higher fair had better chaces of surviving.

##### Analysing if Age of the passengers had any impact on Survival Rate

We have earlier seen the Age feature has considerable nulls. Hence we can impute the nulls before categorising Age and analysing impact on survival.
"""

age_avg = titanic['age'].mean()
age_std = titanic['age'].std()
age_null_count = titanic['age'].isnull().sum()

#random age imputation so that data is complete
#this process should only be done after problem consideration, its not a std process
age_null_random_list = np.random.randint(age_avg - 3*age_std, age_avg + 3*age_std, size=age_null_count)
#can take 0 as min also
titanic['age'][np.isnan(titanic['age'])] = age_null_random_list
titanic['age'] = titanic['age'].astype(int)

titanic['CategoricalAge'] = pd.cut(titanic['age'], 5)

print(titanic[['CategoricalAge', 'survived']].groupby(['CategoricalAge'], as_index=False).mean())

titanic.age.describe()

#since in the original dataset age has a min of 0, random imputation assigned some negative values
#these should be made 0. OR instead of +-3 std dev +-1 std dev could be taken or min 0 can be specified.

"""Insight - 

Kids had the best chance of survival and senior citizens had a really bad chance

#####  Assigning labels to extracted categorical variables - Label encoding
"""

# for sex

titanic['sex'] = titanic['sex'].map( {'female': 0, 'male': 1} ).astype(int)
    

# for Embarked
titanic['embarked'] = titanic['embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# for fare
titanic.loc[ titanic['fare'] <= 7.91, 'fare']                               = 0
titanic.loc[(titanic['fare'] > 7.91) & (titanic['fare'] <= 14.454), 'fare'] = 1
titanic.loc[(titanic['fare'] > 14.454) & (titanic['fare'] <= 31), 'fare']   = 2
titanic.loc[ titanic['fare'] > 31, 'fare']                                  = 3
titanic['fare'] = titanic['fare'].astype(int)

# for age
titanic.loc[ titanic['age'] <= 16, 'age']                          = 0
titanic.loc[(titanic['age'] > 16) & (titanic['age'] <= 32), 'age'] = 1
titanic.loc[(titanic['age'] > 32) & (titanic['age'] <= 48), 'age'] = 2
titanic.loc[(titanic['age'] > 48) & (titanic['age'] <= 64), 'age'] = 3
titanic.loc[ titanic['age'] > 64, 'age']                           = 4

# Feature Selection
drop_elements = [ 'sibsp', 'parch', 'CategoricalAge', 'CategoricalFare' ] # Dropping insignificant features
titanic = titanic.drop(drop_elements, axis = 1)

print (titanic.head(10))

"""### Test/Train split"""

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(titanic.drop('survived',axis=1),titanic['survived'],test_size=0.25,random_state=100)

X_train

X_test

y_train

y_test

y_test.shape

"""### Performance Indicators

Since we have not yet learnt to develop or train a model, let's assume a random set of predictions for our accuracy check.

Our test target dataset (y) had 223 target variables. Hence, generating an array of 223 random 1s and 0s -
"""

predicted_y= np.random.choice([0, 1], size=(223))
predicted_y

"""##### Confusion matrix"""

# Importing Confusion matrix from sklearn
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test,predicted_y, labels=[1,0] )
matrix

"""##### Type 1 & Type 2 Error"""

Type_1_False_positive = matrix[0][1]
print("Type 1 error ( False Postive) is : " + str(Type_1_False_positive))

Type_2_False_negative = matrix[1][0]
print("Type 2 error ( False Negative) is :  " + str(Type_2_False_negative))

"""##### Accuracy"""

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, predicted_y)
print("Accuracy Score is :  " + str(score*100)+' %')