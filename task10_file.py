import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#----------------------------------------------------------------------------------------
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic_data = pd.read_csv(url)
#-----------------------------------------------------------------------------------------
print("First 10 rows of the dataset:")
print(titanic_data.head(10))

#-----------------
# Dataset Overview
#-----------------

print("\nDataset summary:")
titanic_data.info()
print("\nStatistical description of the dataset:")
print(titanic_data.describe())

missing_values = titanic_data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

#----------------------
# Demographics Analysis
#----------------------

average_age = titanic_data['Age'].mean()
print(f"\nAverage age of passengers: {average_age:.2f}")

plt.figure(figsize=(8, 5))
sns.histplot(titanic_data['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

youngest = titanic_data['Age'].min()
oldest   = titanic_data['Age'].max()

print(f"youngest passenger: {youngest},Oldest passenger: {oldest}")

#----------------------
# Survival Analysis
#----------------------

survival_rate = titanic_data['Survived'].mean() * 100
print(f"\nOverall survival rate: {survival_rate:.2f}%")
gender_survival      = titanic_data.groupby('Sex')['Survived'].mean() * 100
pclass_survival      = titanic_data.groupby('Pclass')['Survived'].mean() * 100
embarkation_survival = titanic_data.groupby('Embarked')['Survived'].mean() * 100

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
gender_survival.plot(kind='bar', title='Survival Rate by Gender')
plt.ylabel('Survival Rate (%)')
plt.subplot(1, 3, 2)
pclass_survival.plot(kind='bar', title='Survival Rate by Passenger Class (Pclass)')
plt.ylabel('Survival Rate (%)')
plt.subplot(1, 3, 3)
embarkation_survival.plot(kind='bar', title='Survival Rate by Embarkation Port')
plt.ylabel('Survival Rate (%)')
plt.tight_layout()
plt.show()

#--------------------------
# Family and Fare Analysis
#--------------------------
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1
alone_survival = titanic_data[titanic_data['FamilySize'] == 1]['Survived'].mean() * 100
with_family_survival = titanic_data[titanic_data['FamilySize'] > 1]['Survived'].mean() * 100
print(f"Survival rate for passengers traveling alone: {alone_survival:.2f}%")
print(f"Survival rate for passengers traveling with family: {with_family_survival:.2f}%")

fare_survival = titanic_data.groupby('Survived')['Fare'].mean()
print("\nAverage fare of survivors vs non-survivors:")
print(fare_survival)

plt.figure(figsize=(10, 5))
sns.histplot(titanic_data['Fare'], bins=30, kde=True)
plt.title('Fare Distribution of Passengers')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

#--------------------
# Age Group Analysis
#--------------------
bins = [0, 12, 19, 59, 120]
labels = ['Children', 'Teenagers', 'Adults', 'Seniors']
titanic_data['AgeGroup'] = pd.cut(titanic_data['Age'], bins=bins, labels=labels)

age_group_survival = titanic_data.groupby('AgeGroup')['Survived'].mean() * 100
age_group_survival.plot(kind='bar', title='Survival Rate by Age Group', ylabel='Survival Rate (%)')
plt.show()
