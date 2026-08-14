# %% Imports and setups
import pandas as pd
import numpy as np
import regex as re
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from src.titanic_utils import cabin_imputation
from src.helpers import plot_grouped_overlay, plot_grouped_overlay_density


np.random.seed(0)

# %% Load and clean data
df = pd.read_csv('C:\\Users\\alfie\\Documents\\VScode\\.kaggle\\titanic\\data\\train.csv')


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

# %% Plot: Survival by Age and Gender

fig, ax = plt.subplots()

#Creating total and survived by age

total_mask = df['Age'].notna()
male_mask = (df['Sex'] == 'male') & (df['Survived'] == True) & (df['Age'].notna())
male_mask_inc_na = (df['Sex'] == 'male') & (df['Survived'] == True)
female_mask = (df['Sex'] == 'female') & (df['Survived'] == True) & (df['Age'].notna())
female_mask_inc_na = (df['Sex'] == 'female') & (df['Survived'] == True)
total_survivors_mask_inc_na = df['Survived'] == True

#Checking that dropped survivor is not dropping too many values
assert (male_mask_inc_na.sum()) + (female_mask_inc_na.sum()) == (df['Survived'] == True).sum(), "Male and Female survivors, including those with null ages, do not sum to total survivors."
assert (male_mask.sum()) + (female_mask.sum()) == ((total_survivors_mask_inc_na) & (df['Age'].notna())).sum(), "Non-null Male and Female survivors do not sum to non-null total survivors."


series_dict = {f'Total (includes both survived and died) n = {len(df.loc[total_mask, 'Age'])}':df.loc[total_mask, 'Age'], f'Male survivors n = {len(df.loc[male_mask, 'Age'])}': df.loc[male_mask, 'Age'], f'Female survivors n = {len(df.loc[female_mask, 'Age'])}': df.loc[female_mask, 'Age']}

plot_grouped_overlay_density(ax = ax, sub_series= series_dict, boundary= 0, n_points= 300)

ax.set_xlabel('Age')
ax.set_ylabel('Probability Density')
ax.set_title('Age distribution by Male and Female survivors, against total population')
ax.legend()
plt.figtext(0.25,0.01, "KDE estimated with boundary reflection at age 0.\nDensity of 0-18, females and 0-15 males is higher than the density of those age groups in the total population.")
fig.subplots_adjust(bottom= 0.25)
plt.show()

#Density of 0-18, females and 0-15 males is higher than the density of those age groups in the total population.

# %% Survival by Pclass and Sex heatmap version

#Create a dataframe and then create a pivot table
heatmap_df = df.loc[:, ['Pclass', 'Survived', 'Sex']]
p_table = pd.pivot_table(data = heatmap_df, values = 'Survived', index = 'Pclass', aggfunc='mean', columns= 'Sex')
p_table_count = pd.pivot_table(data = heatmap_df, values = 'Survived', index = 'Pclass', aggfunc='count', columns= 'Sex')
p_table_final = p_table.round(2).astype(str) + "\n(n=" + p_table_count.astype(str) + ")"


print(p_table_final)

fig, ax = plt.subplots()

sns.heatmap(data = p_table, ax=ax, annot = p_table_final , fmt='')

ax.set_title("Survivability rate by Class and Sex")
fig.subplots_adjust(bottom = 0.25)
plt.ylabel('Class')
plt.figtext(0.25, 0.01, "Women of all classes have a higher survivability rate than of any male irrespective of class.\nHowever, Upper class males have a much higher survivability rate than that of Middle and Lower class males.")
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




