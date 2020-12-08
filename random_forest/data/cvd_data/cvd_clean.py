# Importing the required libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# getting the data into a DataFrame.
data=pd.read_csv(r"cardio_train.csv",sep=";")
data.head()

data.info()

data.describe()

# The age is given in days, we have to convert it into years.
data["age"] = data["age"]/365
data["age"] = data["age"].astype("int")

# Dropping id column, its of no use.
data = data.drop(columns = ["id"])

sns.countplot(x = 'cardio', data = data)

# Checking the existence of outliers using boxplots
fig, ax = plt.subplots(figsize = (15,10))
sns.boxplot(data = data, width = 0.5, ax = ax, fliersize = 3)
plt.title("Visualization of outliers")

# ap_hi greater than 200 and lower than or equal to 80 will be removed.
# ap_lo greater than 180 and lower than 50 will be removed.
# height greater or equal to 100 and weight less than 28 will be removed.
outlier = ((data["ap_hi"]>200) | (data["ap_lo"]>180) | (data["ap_lo"]<50) | (data["ap_hi"]<=80) | (data["height"]<=100)
             | (data["weight"]<=28) )
print("There is {} outlier".format(data[outlier]["cardio"].count()))

# Removing  the outlier from the Dataset.
data = data[~outlier]

# BoxPlot after removing the outliers.
fig, ax = plt.subplots(figsize = (15,10))
sns.boxplot(data = data, width = 0.5, ax = ax, fliersize = 3)
plt.title("Visualization of outliers")


X = data.drop(columns = ['cardio'])
y = data['cardio']
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in X:
    if plotnumber<=16 :
        ax = plt.subplot(4,4,plotnumber)
        sns.stripplot(y,X[column])
    plotnumber+=1

plt.tight_layout()


# creating a heatmap of correlation of the data.
corr = data.corr()
f, ax = plt.subplots(figsize = (15,15))
sns.heatmap(corr, annot=True, fmt=".3f", linewidths=0.5, ax=ax)

data["bmi"] = data["weight"]/ (data["height"]/100)**2

data.head()


# Detecting Genders
a = data[data["gender"]==1]["height"].mean()
b = data[data["gender"]==2]["height"].mean()
if a > b:
    gender = "male"
    gender2 = "female"
else:
    gender = "female"
    gender2 = "male"
print("Gender:1 is "+ gender +" & Gender:2 is " + gender2)

data["gender"] = data["gender"] % 2

X = data.drop(columns = ['cardio'])
y = data['cardio']
print(X)

print(data.head())

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scalar=MinMaxScaler()
x_scaled=scalar.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.20, random_state = 9)
np.save('X_train_cvd.npy', X_train)
np.save('X_test_cvd.npy', X_test)
np.save('y_train_cvd.npy', y_train)
np.save('y_test_cvd.npy', y_test)

