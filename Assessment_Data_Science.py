#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score


# # Exploratory Data Analysis (EDA):
# 
# Step 1: Loaded the dataset and examined its structure and dimensions.
# 
# Step 2: Checked for missing values and handled them appropriately (e.g., imputation or removal).
# 
# Step 3: Explored the distribution and summary statistics of each feature.
# 
# Step 4: Visualized the relationships between variables using heat map.

# In[2]:


# Load the dataset
data = pd.read_csv(r"C:\Users\Nimisha\OneDrive\Desktop\Assessment\starcraft_player_data.csv")

# Display the first few rows of the dataset
data.head()


# In[3]:


data.info()


# In[4]:


# Check the shape of the dataset
print("Shape of the dataset:", data.shape)

# Check for missing values
print("Missing values:\n", data.isna().sum())

# Summary statistics
print("Summary statistics:\n", data.describe())


# In[5]:


# Check the distribution of the target variable
print("Distribution of the target variable:\n", data['LeagueIndex'].value_counts())


# In[6]:


class_counts = data['LeagueIndex'].value_counts()
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()


# After looking at the dimension and structure of the dataset , I noticed a few important characteristics about the dataset:
# 1. There are 3 columns described as objects and those are  Age, TotalHours and HoursPerWeek. I tried to find the null values in these columns but there are no null values. Instead, they have '?' so it needs to be either removed or imputed. First, we will simply remove all the '?' from the dataset.
# 
# 2. This is a class imbalance problem which we will address later on. As we can see there are very few data points with LeagueIndex 7 and 8.

# Conducted feature selection using correlation analysis and identified relevant features.

# In[7]:


data['Age'].unique()


# In[8]:


data['HoursPerWeek'].unique()


# In[9]:


data['TotalHours'].unique()


# In[10]:


data[data['TotalHours']=='?']


# I checked all the three columns with '?' and figured out that TotalHours has the maximum '?' and if we drop its rows then our 
# issue will be resolved because it combines the '?' rows of the other 2 columns as well.

# In[11]:


data2 = data.drop(data[data['TotalHours'] == '?'].dropna().index)


# In[12]:


data2.head()


# In[13]:


data2.info()


# In[14]:


data2[data2['Age']=='?']


# In[15]:


data2[data2['HoursPerWeek']=='?']


# Then I converted all the 3 columns to integer type to find the correlation between the features.

# In[16]:


#converting them into integer
data2['Age'] = data2['Age'].astype('int64')
data2['HoursPerWeek'] = data2['HoursPerWeek'].astype('int64')
data2['TotalHours'] = data2['TotalHours'].astype('int64')


# In[17]:


data2.isna().sum()


# In[18]:


#Then I checked correlation between columns to understand what impact does the other features have on target variable. 
correl = data2.corr()

trace = go.Heatmap(z=correl.values,
                   x=correl.index.values,
                   y=correl.columns.values)
data=[trace]
layout = go.Layout(width=1000, height=900)
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[19]:


sorted_corr = correl['LeagueIndex'].sort_values(ascending=False)
sorted_corr
#found the two least correlated columns to LeagueIndex i.e. GameID and TotalHours


# # Data Preprocessing and Feature Engineering:
# 
# Step 1: Split the data into features (X) and the target variable (y) for rank prediction.
# 
# Step 2: Scaled the continuous variables using standardization or normalization.

# In[20]:


# Split the dataset into features and target variable
X = data2.drop('LeagueIndex', axis=1)
y = data2['LeagueIndex']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # Model Selection, Training and Evaluation:
# 
# 1. Selected appropriate models for rank prediction, such as logistic regression, decision trees, random forests, gradient boosting, SVM, or Neural Network.
# 
# 
# 2. Split the data into training and testing sets for model evaluation.
# 
# 
# 3. Trained the chosen models on the training set.
# 
# 
# 4. Evaluated the trained models on the testing set using suitable metrics like F1 score. I used F1 score to evaluate the performance instead of accuracy because this is a class imbalance problem.

# In[21]:


# Create and train different models
models = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    SVC(),
    MLPClassifier()
]

model_names = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'Gradient Boosting',
    'SVM',
    'Neural Network'
]
scores = []
# Evaluate models and print accuracy
for model, name in zip(models, model_names):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    f1score = f1_score(y_test, y_pred, average='weighted')
    print(f"{name} f1 Score: {f1score}")
    scores.append(f1score)
  


# In[22]:


# Plotting the F1 scores
plt.figure(figsize=(8, 6))
plt.bar(model_names, scores)
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.title('Comparison of F1 Scores for Different Models')
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Set the y-axis limit
plt.show()


# # Class Imbalance Problem
# Now we will address the class imbalance problem by class weighting. Assign higher weights to the minority class samples or 
# lower weights to the majority class samples during model training. This gives more importance to the minority class during the 
# learning process. I added weights and re-evaluated the decision tree classifier. 

# In[23]:


# Calculate class weights
class_weights = dict(zip(np.unique(y_train), np.bincount(y_train)))

# Create and train the decision tree classifier with class weights
dt_classifier = DecisionTreeClassifier(class_weight = class_weights)
dt_classifier.fit(X_train_scaled, y_train)

# Make predictions on the testing data
y_pred = dt_classifier.predict(X_test_scaled)

# Compute the weighted F1 score
f1score = f1_score(y_test, y_pred, average='weighted')
print("f1 Score:",f1score)


# # Removed least correlated columns

# In[24]:


#Next, we remove the two least correlated columns to LeagueIndex. 
data3 = data2.drop(columns=['GameID','TotalHours'])


# In[25]:


# Split the dataset into features and target variable
X = data3.drop('LeagueIndex', axis=1)
y = data3['LeagueIndex']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train different models
models = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    SVC(),
    MLPClassifier()
]

model_names = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'Gradient Boosting',
    'SVM',
    'Neural Network'
]

# Evaluate models and print accuracy
for model, name in zip(models, model_names):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    f1score = f1_score(y_test, y_pred, average="weighted")
    print(f"{name} F1 Score: {f1score}")


# # K-Nearest Neighbors Classifier

# In[26]:


knn_model = KNeighborsClassifier(n_neighbors=14)  
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
f1score = f1_score(y_test, knn_pred, average="weighted")
print("KNN F1 Score:", f1score)


# # Imputation using KNN
# Now we will perform imputation. Instead of dropping all the rows with '?', we will fill the missing values through imputation.

# In[27]:


sampledata = pd.read_csv(r"C:\Users\Nimisha\OneDrive\Desktop\Assessment\starcraft_player_data.csv")


# In[28]:


sampledata[['Age','TotalHours','HoursPerWeek']] = sampledata[['Age','TotalHours','HoursPerWeek']].replace('?', None)


# In[29]:


sampledata.info()


# In[30]:


sampledata.isna().sum()


# In[31]:


#imputing the values using knn
missingdata = sampledata[['Age','TotalHours','HoursPerWeek']]


# In[32]:


k = 5
knn_imputer = KNNImputer(n_neighbors=k)
imputed_data = knn_imputer.fit_transform(missingdata)


# In[33]:


df_imputed = pd.DataFrame(imputed_data, columns=missingdata.columns)


# In[34]:


df_imputed.info()


# In[35]:


sampledata[['Age','TotalHours','HoursPerWeek']] = df_imputed[['Age','TotalHours','HoursPerWeek']]


# In[36]:


sampledata.info()


# In[37]:


sampledata.isna().sum()


# In[38]:


# Split the dataset into features and target variable
X = sampledata.drop('LeagueIndex', axis=1)
y = sampledata['LeagueIndex']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
X_train
X_test
y_train
y_test


# In[39]:


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
f1score = f1_score(y_test, rf_pred,average= "weighted")
print("Random Forest F1 Score:", f1score)


# Finally, let's address the hypothetical scenario where stakeholders want to collect more data and seek guidance. Based on the EDA and model results, I would suggest the following:
# 
# 1. Collect more samples for the minority classes: Since the dataset is imbalanced, collecting more data for the underrepresented rank levels can improve the model's performance.
# 
# 2. Gather additional features: If there are relevant features that are not present in the current dataset, collecting additional data with those features can enhance the model's predictive power.
# 
# 3. Monitor data quality: Ensure that the new data collection process maintains data quality standards, such as avoiding missing values, outliers, or inconsistencies.
# 
# 4. Perform iterative model updates: As more data becomes available, it's beneficial to periodically update and retrain the model using the augmented dataset to capture any evolving patterns or changes in player performance.
# 
# These recommendations aim to enhance the predictive capabilities of the model and provide more accurate rank predictions.

# In[ ]:




