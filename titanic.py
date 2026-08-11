# %% Imports and setups
import pandas as pd
import numpy as np
import regex as re
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from titanic.src.titanic_utils import cabin_imputation
from titanic.src.helpers import plot_grouped_overlay

np.random.seed(0)

# %% Load and clean data
df = pd.read_csv('C:\\Users\\alfie\\Documents\\VScode\\.kaggle\\titanic\\train.csv')


#Cleaning values names
df['Survived'] = df['Survived'].map({0: False, 1: True}).astype(bool)
df['Pclass'] = df['Pclass'].map({1: "Upper", 2: "Middle", 3: "Lower"})
df['Embarked'] = df['Embarked'].map({"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"})
df['Pclass'] = pd.Categorical(df['Pclass'], categories= ['Lower', 'Middle', 'Upper'], ordered = True)


# %% Plot: Survival by Pclass
ax = sns.countplot(data=df, x='Pclass', hue='Survived', order = ['Lower', 'Middle', 'Upper'])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Died','Survived'])
plt.show()

# %% Plot: Survival by Sex
ax = sns.countplot(data = df, x ='Sex', hue = 'Survived')
handles , labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Died','Survived'])
plt.show()

# %% Plot: Survival by Sex and Pclass
fig, ax = plt.subplots(layout='constrained')

#Create total and survivors groupby's
total = df.groupby(['Sex', 'Pclass']).size().unstack()
survivors = df[df['Survived'] == True].groupby(['Sex', 'Pclass']).size().unstack()
all_classes = ['Lower', 'Middle', 'Upper']

plot_grouped_overlay(ax=ax, total_df=total, sub_df=survivors, categories=all_classes)

ax.set_xlabel('Sex')
ax.set_ylabel('Count')
ax.set_title('Survivors vs Total, by Sex and Pclass')
ax.legend(handles=[Patch(color='lightgray', label='Total'), Patch(color='green', label='Survived')])

plt.show()

# %% Survivors by Gender
print(df.groupby(['Sex','Survived']).size())

# %% Cabin analysis
#Building with and without cabins to see if there is a correlation between the means
without_cabin = df[df['Cabin'].isna()]
with_cabin = df[df['Cabin'].notna()]

wo_cabin_series = pd.Series(without_cabin['Fare'])
w_cabin_series = pd.Series(with_cabin['Fare'])

#T-test to test if averages between with and without cabins are explained by each other
t_stat, p_value = stats.ttest_ind(wo_cabin_series, w_cabin_series)
print(f"T-Statistic: {t_stat}\nP_value: {p_value}")

#Grouping Cabins by class
print(df.groupby('Pclass')['Cabin'].apply(lambda x: x.str[0].value_counts()))


#Proportion of null / non-null fields by class
print(df.groupby(df['Cabin'].isna())['Pclass'].value_counts(normalize=True))

#Size of null Cabins in upper class
print(df.loc[(df['Cabin'].isna()) & (df['Pclass'] == "Upper")].shape[0])
#Size of non-null Cabins in upper class
print(df.loc[(df['Cabin'].notna()) & (df['Pclass'] == "Upper")].shape[0])

# %% Cabin imputation
#Initialise Cabin Notation
df['Cabin_Notation'] = df['Cabin']

#For non-null fields, take the cabin notation.
mask = df['Cabin'].notna()
df.loc[mask, 'Cabin_Notation'] = df.loc[mask, 'Cabin'].str[0]

cabin_imputation(df=df, pclass='Upper')
cabin_imputation(df=df, pclass='Middle')
cabin_imputation(df=df, pclass='Lower')




