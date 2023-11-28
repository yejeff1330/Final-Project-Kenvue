#!/usr/bin/env python
# coding: utf-8

# # MGTC28 Final Assignment
# ### Kenvue Data Cleaning & Analysis

# For this assignment, we will be using the datasets provided by Kenvue,
# The datasets are stored in .csv files, holding different information in 5 data sets (Customer DC Inventory, Factory POS $, Ecomm POS $, Total Sales, Total Trade Spend):
# 
# - POS Factory $ = POS Revenue (i.e., Retailer Revenue)
# - Need State = Grouping/categorizing of different customers needs 
# - Total Sales = All sales across customers
# - DC Amount = How much inventory at distribution centers
# - Store Amt on Hand = How much inventory at a store level
# - Total Trade Spend = the amount of money spent to promote that product
# 
# By: Jeff Ye (1007133601) & Yousuf Zaidi

# In[ ]:





# In[5]:


# Importing Libraries (PANDAS,Seaborn & Statsmodel) and CSV Files

import pandas as pd

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

import statsmodels.api as sm


dfDC = pd.read_csv('E:\Downloads\kenvue_data\Customer DC Inventory - UTSC Lecture.csv')
dfFactory = pd.read_csv('E:\Downloads\kenvue_data\Factory POS $ - UTSC Lecture.csv')
dfECOM = pd.read_csv('E:\Downloads\kenvue_data\Total Ecomm POS (Factory $) - UTSC Lecture.csv')
dfSales = pd.read_csv('E:\Downloads\kenvue_data\Total Sales - UTSC Lecture.csv')
dfTradeSpend = pd.read_csv('E:\Downloads\kenvue_data\Total Trade Spend.csv')


# In[ ]:





# In[6]:


# Compare Trade Spend Vs Sales for each product in each season

## Categorize Trade Spend for Each Need State into Seasons
def categorize_season(week):
    if 1 <= week <= 13 or 50 <= week <= 52:
        return 'Winter'
    elif 15 <= week <= 26:
        return 'Spring'
    elif 27 <= week <= 38:
        return 'Summer'
    else:
        return 'Fall'
    
dfTradeSpend["Season"] = dfTradeSpend["Fiscal Week"].apply(categorize_season)

NSTS_dfs = {}
for Need_State in range(1, 6):
    NSTS_col = f'Need State {Need_State}'
    NSTS_df = dfTradeSpend.groupby(['Fiscal Year', 'Season'])[NSTS_col].sum().reset_index()
    NSTS_dfs[NSTS_col] = NSTS_df
    
print(NSTS_dfs["Need State 1"])

#Categorize Total Sales for Each Need State into Seasons

dfSales["Season"] = dfSales["Fiscal Week"].apply(categorize_season)

NSS_dfs = {}
for Need_State in range(1, 6):
    NSS_col = f'Need State {Need_State}'
    NSS_df = dfSales.groupby(['Fiscal Year', 'Season'])[NSS_col].sum().reset_index()
    NSS_dfs[NSS_col] = NSS_df
    
print(NSS_dfs["Need State 1"])


# In[7]:


# Assuming NSTS_dfs["Need State 1"] and NSS_dfs["Need State 1"] are your DataFrames
trade_spend_df = NSTS_dfs["Need State 1"]
sales_df = NSS_dfs["Need State 1"]

# Merge the DataFrames on Fiscal Year and Season
merged_df = pd.merge(trade_spend_df, sales_df, on=['Fiscal Year', 'Season'], suffixes=('_TradeSpend', '_Sales'))


# In[9]:


plt.figure(figsize=(10, 6))

# Create the first y-axis for Trade Spend for Need State 1
ax = sns.lineplot(data=merged_df, x='Season', y='Need State 1_TradeSpend', marker='o', label='Trade Spend')
ax.set_ylabel('Trade Spend Amount')

# Create the second y-axis for Sales for Need State 1
ax2 = ax.twinx()
sns.lineplot(data=merged_df, x='Season', y='Need State 1_Sales', marker='o', label='Sales', color='r', ax=ax2)
ax2.set_ylabel('Sales Amount')

plt.title('Trade Spend Vs Sales for Need State 1 by Season')
ax.figure.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
plt.show()


# In[96]:


# Example for Need State 1
data = merged_df[['Need State 1_TradeSpend', 'Need State 1_Sales']]
# Defining the independent variable (X) and dependent variable (y)
X = data['Need State 1_TradeSpend']
y = data['Need State 1_Sales']

# Adding a constant to the model (for intercept)
X = sm.add_constant(X)

# Building the OLS model
model = sm.OLS(y, X).fit()

# Viewing the summary of the model
print(model.summary())


# In[ ]:





# In[ ]:





# In[11]:


# Merge the DataFrames for Need State 2
trade_spend_df_2 = NSTS_dfs["Need State 2"]
sales_df_2 = NSS_dfs["Need State 2"]

merged_df_2 = pd.merge(trade_spend_df_2, sales_df_2, on=['Fiscal Year', 'Season'], suffixes=('_TradeSpend', '_Sales'))


# In[12]:


print(merged_df_2[['Need State 2_TradeSpend', 'Need State 2_Sales']].describe())


# In[ ]:





# In[13]:


plt.figure(figsize=(10, 6))

# Create the first y-axis for Trade Spend
ax = sns.lineplot(data=merged_df_2, x='Season', y='Need State 2_TradeSpend', marker='o', label='Trade Spend')
ax.set_ylabel('Trade Spend Amount')

# Create the second y-axis for Sales
ax2 = ax.twinx()
sns.lineplot(data=merged_df_2, x='Season', y='Need State 2_Sales', marker='o', label='Sales', color='r', ax=ax2)
ax2.set_ylabel('Sales Amount')

plt.title('Trade Spend Vs Sales for Need State 2 by Season')
ax.figure.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
plt.show()


# In[14]:


# Example for Need State 2
data = merged_df_2[['Need State 2_TradeSpend', 'Need State 2_Sales']]


# In[15]:


# Defining the independent variable (X) and dependent variable (y)
X = data['Need State 2_TradeSpend']
y = data['Need State 2_Sales']

# Adding a constant to the model (for intercept)
X = sm.add_constant(X)

# Building the OLS model
model = sm.OLS(y, X).fit()

# Viewing the summary of the model
print(model.summary())


# In[16]:


# Merge the DataFrames for Need State 3
trade_spend_df_3 = NSTS_dfs["Need State 3"]
sales_df_3 = NSS_dfs["Need State 3"]

merged_df_3 = pd.merge(trade_spend_df_3, sales_df_3, on=['Fiscal Year', 'Season'], suffixes=('_TradeSpend', '_Sales'))


# In[17]:


plt.figure(figsize=(10, 6))

# Create the first y-axis for Trade Spend
ax = sns.lineplot(data=merged_df_3, x='Season', y='Need State 3_TradeSpend', marker='o', label='Trade Spend')
ax.set_ylabel('Trade Spend Amount')

# Create the second y-axis for Sales
ax2 = ax.twinx()
sns.lineplot(data=merged_df_3, x='Season', y='Need State 3_Sales', marker='o', label='Sales', color='r', ax=ax2)
ax2.set_ylabel('Sales Amount')

plt.title('Trade Spend Vs Sales for Need State 3 by Season')
ax.figure.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
plt.show()


# In[18]:


# Example for Need State 3
data = merged_df_3[['Need State 3_TradeSpend', 'Need State 3_Sales']]
# Defining the independent variable (X) and dependent variable (y)
X = data['Need State 3_TradeSpend']
y = data['Need State 3_Sales']

# Adding a constant to the model (for intercept)
X = sm.add_constant(X)

# Building the OLS model
model = sm.OLS(y, X).fit()

# Viewing the summary of the model
print(model.summary())
data = merged_df_3[['Need State 3_TradeSpend', 'Need State 3_Sales']]


# In[19]:


# Merge the DataFrames for Need State 4
trade_spend_df_4 = NSTS_dfs["Need State 4"]
sales_df_4 = NSS_dfs["Need State 4"]

merged_df_4 = pd.merge(trade_spend_df_4, sales_df_4, on=['Fiscal Year', 'Season'], suffixes=('_TradeSpend', '_Sales'))



# In[21]:


plt.figure(figsize=(10, 6))

# Create the first y-axis for Trade Spend
ax = sns.lineplot(data=merged_df_4, x='Season', y='Need State 4_TradeSpend', marker='o', label='Trade Spend')
ax.set_ylabel('Trade Spend Amount')

# Create the second y-axis for Sales
ax2 = ax.twinx()
sns.lineplot(data=merged_df_4, x='Season', y='Need State 4_Sales', marker='o', label='Sales', color='r', ax=ax2)
ax2.set_ylabel('Sales Amount')

plt.title('Trade Spend Vs Sales for Need State 4 by Season')
ax.figure.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
plt.show()


# In[22]:


# Example for Need State 4
data = merged_df_4[['Need State 4_TradeSpend', 'Need State 4_Sales']]
# Defining the independent variable (X) and dependent variable (y)
X = data['Need State 4_TradeSpend']
y = data['Need State 4_Sales']

# Adding a constant to the model (for intercept)
X = sm.add_constant(X)

# Building the OLS model
model = sm.OLS(y, X).fit()

# Viewing the summary of the model
print(model.summary())
data = merged_df_4[['Need State 4_TradeSpend', 'Need State 4_Sales']]


# In[23]:


# Merge the DataFrames for Need State 5
trade_spend_df_5 = NSTS_dfs["Need State 5"]
sales_df_5 = NSS_dfs["Need State 5"]

merged_df_5 = pd.merge(trade_spend_df_5, sales_df_5, on=['Fiscal Year', 'Season'], suffixes=('_TradeSpend', '_Sales'))


# In[25]:


plt.figure(figsize=(10, 6))

# Create the first y-axis for Trade Spend
ax = sns.lineplot(data=merged_df_5, x='Season', y='Need State 5_TradeSpend', marker='o', label='Trade Spend')
ax.set_ylabel('Trade Spend Amount')

# Create the second y-axis for Sales
ax2 = ax.twinx()
sns.lineplot(data=merged_df_5, x='Season', y='Need State 5_Sales', marker='o', label='Sales', color='r', ax=ax2)
ax2.set_ylabel('Sales Amount')

plt.title('Trade Spend Vs Sales for Need State 5 by Season')
ax.figure.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
plt.show()


# In[26]:


# Example for Need State 5
data = merged_df_5[['Need State 5_TradeSpend', 'Need State 5_Sales']]
# Defining the independent variable (X) and dependent variable (y)
X = data['Need State 5_TradeSpend']
y = data['Need State 5_Sales']

# Adding a constant to the model (for intercept)
X = sm.add_constant(X)

# Building the OLS model
model = sm.OLS(y, X).fit()

# Viewing the summary of the model
print(model.summary())
data = merged_df_5[['Need State 5_TradeSpend', 'Need State 5_Sales']]


# In[126]:


dfECOM["Season"] = dfECOM["Fiscal Week"].apply(categorize_season)


# In[127]:


seasonal_analysis = df_melted.groupby('Season')['Ecomm POS'].sum().reset_index()
print(seasonal_analysis)


# In[148]:


sns.barplot(x='Season', y='Ecomm POS', data=seasonal_analysis)
plt.title('E-commerce POS by Season')
plt.show()


# In[139]:


print(df_melted)


# In[133]:


print(seasonal_analysis)


# In[155]:


df_melted = pd.melt(dfECOM, id_vars=['Unnamed: 0', 'Unnamed: 1'], 
                    var_name='Fiscal Week', value_name='Ecomm POS')


df_melted['Fiscal Week'] = df_melted['Fiscal Week'].str.extract('(\d+)')
df_melted.dropna(subset=['Fiscal Week'], inplace=True)
df_melted['Fiscal Week'] = df_melted['Fiscal Week'].astype(int)

def categorize_season(week):
    if 1 <= week <= 13 or 50 <= week <= 52:
        return 'Winter'
    elif 15 <= week <= 26:
        return 'Spring'
    elif 27 <= week <= 38:
        return 'Summer'
    else:
        return 'Fall'

df_melted['Season'] = df_melted['Fiscal Week'].apply(categorize_season)

def categorize_season(week):
    if 1 <= week <= 13 or 50 <= week <= 52:
        return 'Winter'
    elif 15 <= week <= 26:
        return 'Spring'
    elif 27 <= week <= 38:
        return 'Summer'
    else:
        return 'Fall'

df_melted['Season'] = df_melted['Fiscal Week'].apply(categorize_season)


# In[156]:


# Print the column names to verify the correct identifiers
print(dfECOM.columns)


# In[ ]:





# In[180]:


# Create a pivot table for visualization
pivot_table = seasonal_analysis.pivot(index='Need State', columns='Season', values='Ecomm POS')

# Generate the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlGnBu')
plt.title('E-commerce POS by Need State and Season')
plt.ylabel('Need State')
plt.xlabel('Season')
plt.tight_layout()  # Adjust the layout for better appearance
plt.show()


# In[ ]:





# In[181]:


# Rename columns first
dfFactory.rename(columns={'Unnamed: 0': 'Need State', 'Unnamed: 1': 'Fiscal Week'}, inplace=True)

# Convert 'Fiscal Week' to numeric
dfFactory['Fiscal Week'] = pd.to_numeric(dfFactory['Fiscal Week'], errors='coerce')

# Define the function to categorize weeks into seasons
def categorize_season(week):
    # ... (season categorization logic)

# Apply the categorize_season function to 'Fiscal Week' column
    dfFactory['Season'] = dfFactory['Fiscal Week'].apply(categorize_season)




# In[182]:


# Melt the DataFrame
df_melted = dfFactory.melt(id_vars=['Need State', 'Fiscal Week'], var_name='Fiscal Year', value_name='Ecomm POS')

# Ensure 'Ecomm POS' is numeric
df_melted['Ecomm POS'] = pd.to_numeric(df_melted['Ecomm POS'], errors='coerce')

# Drop NaN values if there were any non-numeric values that got converted to NaN
df_melted.dropna(subset=['Ecomm POS'], inplace=True)


# In[183]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assume dfFactory is your initial DataFrame after loading the data
# Rename columns first to ensure they match what we expect
dfFactory.rename(columns={'Unnamed: 0': 'Need State', 'Unnamed: 1': 'Fiscal Week'}, inplace=True)

# Convert 'Fiscal Week' to numeric
dfFactory['Fiscal Week'] = pd.to_numeric(dfFactory['Fiscal Week'], errors='coerce')

# Define the function to categorize weeks into seasons
def categorize_season(week):
    if 1 <= week <= 13 or 50 <= week <= 52:
        return 'Winter'
    elif 14 <= week <= 26:
        return 'Spring'
    elif 27 <= week <= 39:
        return 'Summer'
    elif 40 <= week <= 49:
        return 'Fall'
    else:
        return 'Unknown'  # For any other case

# Apply the categorize_season function to 'Fiscal Week' column
dfFactory['Season'] = dfFactory['Fiscal Week'].apply(categorize_season)

# Check if 'Season' column exists and its unique values to ensure it's correct
print('Season column in dfFactory:', 'Season' in dfFactory.columns)
print('Unique seasons:', dfFactory['Season'].unique())

# Melt the DataFrame to long format
df_melted = dfFactory.melt(id_vars=['Need State', 'Fiscal Week'], var_name='Fiscal Year', value_name='Ecomm POS')

# Ensure 'Ecomm POS' is numeric
df_melted['Ecomm POS'] = pd.to_numeric(df_melted['Ecomm POS'], errors='coerce')

# Drop NaN values if there were any non-numeric values that got converted to NaN
df_melted.dropna(subset=['Ecomm POS'], inplace=True)

# Verify that the 'Season' column exists after melting and before grouping
print('Season column in df_melted:', 'Season' in df_melted.columns)



# In[184]:


import numpy as np
# Rename columns first
dfFactory.rename(columns={'Unnamed: 0': 'Need State', 'Unnamed: 1': 'Fiscal Week'}, inplace=True)

# Convert 'Fiscal Week' to numeric
dfFactory['Fiscal Week'] = pd.to_numeric(dfFactory['Fiscal Week'], errors='coerce')

# Define the function to categorize weeks into seasons
def categorize_season(week):
    if pd.isna(week):
        return np.nan  # Skip NaN values
    elif 1 <= week <= 13 or 50 <= week <= 52:
        return 'Winter'
    elif 14 <= week <= 26:
        return 'Spring'
    elif 27 <= week <= 39:
        return 'Summer'
    elif 40 <= week <= 49:
        return 'Fall'

# Apply the categorize_season function
dfFactory['Season'] = dfFactory['Fiscal Week'].apply(categorize_season)

# Melt the DataFrame
df_melted = dfFactory.melt(id_vars=['Need State', 'Fiscal Week', 'Season'], 
                           var_name='Fiscal Year', value_name='Sales')

# Ensure 'Sales' is numeric
df_melted['Sales'] = pd.to_numeric(df_melted['Sales'], errors='coerce')

# Drop NaN values
df_melted.dropna(subset=['Sales'], inplace=True)

# Group by 'Need State' and 'Season' and sum the 'Sales'
grouped = df_melted.groupby(['Need State', 'Season'])['Sales'].sum().reset_index()

# Create pivot table for heatmap
pivot_table = grouped.pivot(index='Need State', columns='Season', values='Sales')

# Generate heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlGnBu')
plt.title('Factory Sales by Need State and Season')
plt.xlabel('Season')
plt.ylabel('Need State')
plt.tight_layout()
plt.show()


# In[ ]:




