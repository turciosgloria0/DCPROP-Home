#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[11]:


location = "datasets/DC_Properties.csv"
df = pd.read_csv('DC_Properties.csv')


# In[12]:


df.columns


# In[13]:


df.describe()


# In[14]:


df['WARD'].unique()


# In[18]:


newdf = df.copy()
newdf.head()


# In[19]:


newdf.drop(['Unnamed: 0', 'HF_BATHRM', 'HEAT', 'AC', 'NUM_UNITS', 'ROOMS',
       'AYB', 'YR_RMDL', 'EYB', 'STORIES', 'SALEDATE',
       'QUALIFIED', 'SALE_NUM', 'BLDG_NUM', 'STYLE', 'STRUCT', 'GRADE',
       'CNDTN', 'EXTWALL', 'ROOF', 'INTWALL',
       'USECODE', 'LANDAREA', 'GIS_LAST_MOD_DTTM', 'SOURCE', 'CMPLX_NUM',
       'LIVING_GBA', 'FULLADDRESS', 'CITY', 'STATE', 'ZIPCODE', 'NATIONALGRID',
       'LATITUDE', 'LONGITUDE', 'ASSESSMENT_NBHD', 'ASSESSMENT_SUBNBHD',
       'CENSUS_TRACT', 'CENSUS_BLOCK', 'WARD', 'SQUARE', 'X', 'Y'], axis=1, inplace=True)

newdf.head()


# In[20]:


newdf.describe()


# In[21]:


newdf.info()


# In[22]:


newdf.shape


# In[23]:


newdf.isnull().sum()


# In[24]:


#clean outliers: pregnancy

q1 = newdf['PRICE'].quantile(.25)
q3 = newdf['PRICE'].quantile(.75)
iqr = q3-q1
toprange_price = q3 + iqr * 1.5
botrange_price = q1 - iqr * 1.5

print(toprange_price)
print(botrange_price)


# In[25]:


#clean outliers: pregnancy

q1 = newdf['GBA'].quantile(.25)
q3 = newdf['GBA'].quantile(.75)
iqr = q3-q1
toprange_gba = q3 + iqr * 1.5
botrange_gba = q1 - iqr * 1.5

print(toprange_gba)
print(botrange_gba)


# In[26]:


#clean outliers: pregnancy

q1 = newdf['FIREPLACES'].quantile(.25)
q3 = newdf['FIREPLACES'].quantile(.75)
iqr = q3-q1
toprange_fire = q3 + iqr * 1.5
botrange_fire = q1 - iqr * 1.5

print(toprange_fire)
print(botrange_fire)


# In[27]:


newdf['KITCHENS'].unique()


# In[28]:


newdf['KITCHENS'].value_counts()


# In[29]:


cleandf = newdf.copy()


# In[30]:


cleandf = cleandf.drop(cleandf[cleandf['PRICE'] > toprange_price].index)
cleandf = cleandf.drop(cleandf[cleandf['PRICE'] < botrange_price].index)

cleandf = cleandf.drop(cleandf[cleandf['GBA'] > toprange_gba].index)
cleandf = cleandf.drop(cleandf[cleandf['GBA'] < botrange_gba].index)

cleandf = cleandf.drop(cleandf[cleandf['FIREPLACES'] > toprange_fire].index)
cleandf = cleandf.drop(cleandf[cleandf['FIREPLACES'] < botrange_fire].index)

cleandf.head()


# In[31]:


#WHAT IS THIS??
pd.options.display.float_format='{0:,.2f}'.format
cleandf.describe()


# In[32]:


cleandf.isnull().sum()


# In[33]:


#fill with mean?
cleandf['PRICE'].fillna(cleandf['PRICE'].mean(), inplace=True)


# In[34]:


cleandf['GBA'].fillna(cleandf['GBA'].mean(), inplace=True)


# In[35]:


cleandf['KITCHENS'].fillna(cleandf['KITCHENS'].mean(), inplace=True)


# In[36]:


cleandf.isnull().sum()


# In[37]:


nomissingdf = cleandf.dropna()
nomissingdf.isnull().sum()


# In[38]:


nomissingdf.dtypes


# In[55]:


nomissingdf = pd.get_dummies(data = nomissingdf)


# In[56]:


nomissingdf.head()


# In[57]:


nomissingdf.shape


# In[58]:


nomissingdf.corr()


# In[59]:


corr = nomissingdf.corr()
sns.heatmap(corr, vmin=-1, annot=True)


# In[60]:


import statsmodels.formula.api as smf


# In[62]:


result = smf.ols('PRICE ~ BATHRM + BEDRM + GBA + KITCHENS + FIREPLACES + QUADRANT_NE + QUADRANT_NW + QUADRANT_SE + QUADRANT_SW', data=nomissingdf).fit()


# In[63]:


result.summary()


# In[64]:


result = smf.ols('PRICE ~ BATHRM + BEDRM + GBA + KITCHENS + FIREPLACES', data=nomissingdf).fit()


# In[65]:


result.summary()


# In[66]:


X = nomissingdf.drop(['PRICE'], axis=1)
Y = nomissingdf["PRICE"]


# In[67]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm


# In[68]:



X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, random_state = 5)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[69]:


lm.fit(X_train, Y_train)
lm.score(X_train, Y_train)


# In[70]:


pred_test = lm.predict(X_test)
lm.score(X_test, Y_test)


# In[71]:


print ('Fit a model X_train, and calculate MSE with Y_train:', np.mean((Y_train - lm.predict(X_train)) ** 2))

print ('Fit a model X_test, and calculate MSE with Y_test:', np.mean((Y_test - lm.predict(X_test)) ** 2))


# From the linear regression that was run on the data, and the machine learning it appears that the variables we have chosen or the way in which we cleaned those variables, does not allow for clear predictions of outcomes.

# In[72]:


df['STYLE'].unique()


# In[73]:


df['AC'].unique()


# In[116]:


newdf2 = df.copy()
newdf2.head()


# In[117]:


newdf2.drop(['Unnamed: 0', 'HF_BATHRM', 'HEAT', 'AC', 'NUM_UNITS', 'ROOMS',
       'AYB', 'YR_RMDL', 'EYB', 'STORIES', 'SALEDATE',
       'QUALIFIED', 'SALE_NUM', 'BLDG_NUM', 'STYLE', 'STRUCT', 'GRADE',
       'CNDTN', 'EXTWALL', 'ROOF', 'INTWALL',
       'USECODE', 'LANDAREA', 'GIS_LAST_MOD_DTTM', 'SOURCE', 'CMPLX_NUM',
       'LIVING_GBA', 'FULLADDRESS', 'CITY', 'STATE', 'ZIPCODE', 'NATIONALGRID',
       'LATITUDE', 'LONGITUDE', 'ASSESSMENT_NBHD', 'ASSESSMENT_SUBNBHD',
       'CENSUS_TRACT', 'CENSUS_BLOCK', 'SQUARE','QUADRANT', 'X', 'Y',], axis=1, inplace=True)

newdf2.head()


# In[118]:


newdf2.isnull().sum()


# In[119]:


q1 = newdf['PRICE'].quantile(.25)
q3 = newdf['PRICE'].quantile(.75)
iqr = q3-q1
toprange_price = q3 + iqr * 1.5
botrange_price = q1 - iqr * 1.5

print(toprange_price)
print(botrange_price)


# In[120]:


q1 = newdf['GBA'].quantile(.25)
q3 = newdf['GBA'].quantile(.75)
iqr = q3-q1
toprange_gba = q3 + iqr * 1.5
botrange_gba = q1 - iqr * 1.5

print(toprange_gba)
print(botrange_gba)


# In[121]:


q1 = newdf['FIREPLACES'].quantile(.25)
q3 = newdf['FIREPLACES'].quantile(.75)
iqr = q3-q1
toprange_fire = q3 + iqr * 1.5
botrange_fire = q1 - iqr * 1.5

print(toprange_fire)
print(botrange_fire)


# In[122]:


cleandf = newdf2.copy()


# In[123]:


cleandf = cleandf.drop(cleandf[cleandf['PRICE'] > toprange_price].index)
cleandf = cleandf.drop(cleandf[cleandf['PRICE'] < botrange_price].index)

cleandf = cleandf.drop(cleandf[cleandf['GBA'] > toprange_gba].index)
cleandf = cleandf.drop(cleandf[cleandf['GBA'] < botrange_gba].index)

cleandf = cleandf.drop(cleandf[cleandf['FIREPLACES'] > toprange_fire].index)
cleandf = cleandf.drop(cleandf[cleandf['FIREPLACES'] < botrange_fire].index)

cleandf.head()


# In[124]:


pd.options.display.float_format='{0:,.2f}'.format
cleandf.describe()


# In[125]:


cleandf.isnull().sum()


# In[126]:


cleandf['PRICE'].fillna(cleandf['PRICE'].mean(), inplace=True)


# In[127]:


cleandf['GBA'].fillna(cleandf['GBA'].mean(), inplace=True)


# In[128]:


cleandf['KITCHENS'].fillna(cleandf['KITCHENS'].mean(), inplace=True)


# In[129]:


cleandf.isnull().sum()


# In[130]:


nomissingdf = cleandf.dropna()
nomissingdf.isnull().sum()


# In[131]:


nomissingdf.dtypes


# In[132]:


nomissingdf = pd.get_dummies(data = nomissingdf)


# In[133]:


nomissingdf.head()


# In[134]:


nomissingdf.describe()


# In[135]:


nomissingdf.corr()


# In[136]:


cleanup= nomissingdf.copy()


# In[137]:


cleanup.describe()


# In[141]:


corr = nomissingdf.corr()
sns.heatmap(corr, vmin=-1,annot=True)


# In[329]:


nomissingdf.rename(columns={'WARD_Ward 1': 'WOne',"WARD_Ward 2":'WTwo','WARD_Ward 3':'WThree','WARD_Ward 4':'WFour','WARD_Ward 5':'WFive','WARD_Ward 6':'WSix','WARD_Ward 7':'WSeve','WARD_Ward 8':'WEight'}, inplace = True)


# In[330]:


result = smf.ols('PRICE ~ BATHRM + BEDRM + GBA + KITCHENS + FIREPLACES + WOne + WTwo+ WThree+ WFour+ WFive+ WSix', data=nomissingdf).fit()


# In[331]:


result.summary()


# In[332]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm


# In[333]:



X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, random_state = 5)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[334]:


lm.fit(X_train, Y_train)
lm.score(X_train, Y_train)


# In[335]:


pred_test = lm.predict(X_test)
lm.score(X_test, Y_test)


# In[336]:


print ('Fit a model X_train, and calculate MSE with Y_train:', np.mean((Y_train - lm.predict(X_train)) ** 2))

print ('Fit a model X_test, and calculate MSE with Y_test:', np.mean((Y_test - lm.predict(X_test)) ** 2))


# Still not so good...... Ill keep trying to add different variables and try and get a better fit. 

# In[156]:


df['ASSESSMENT_NBHD'].unique()


# In[ ]:


df['ASSESSMENT_NBHD'].unique()


# In[157]:


df['SALEDATE'].unique()


# In[158]:


df['CENSUS_TRACT'].unique()


# In[159]:


df['GRADE'].unique()


# In[205]:


df.describe()


# In[261]:


newdf3 = df.copy()
newdf3.head()


# In[262]:


newdf3.drop(['Unnamed: 0', 'HF_BATHRM', 'NUM_UNITS',
       'AYB', 'YR_RMDL', 'EYB', 'STORIES', 'SALEDATE',
       'QUALIFIED', 'SALE_NUM', 'BLDG_NUM', 'STYLE', 'STRUCT',
       'CNDTN', 'EXTWALL', 'ROOF', 'INTWALL',
       'USECODE', 'LANDAREA', 'GIS_LAST_MOD_DTTM', 'SOURCE', 'CMPLX_NUM',
       'LIVING_GBA', 'FULLADDRESS', 'CITY', 'STATE', 'ZIPCODE', 'NATIONALGRID',
       'LATITUDE', 'LONGITUDE', 'ASSESSMENT_NBHD', 'ASSESSMENT_SUBNBHD',
       'CENSUS_TRACT', 'CENSUS_BLOCK', 'WARD', 'SQUARE', 'X', 'Y','BATHRM','GBA', 'KITCHENS','QUADRANT', 'FIREPLACES',], axis=1, inplace=True)


# In[263]:


newdf3.head(10)


# In[264]:


newdf3.isnull().sum()


# In[265]:


q1 = newdf3['PRICE'].quantile(.25)
q3 = newdf3['PRICE'].quantile(.75)
iqr = q3-q1
toprange_price = q3 + iqr * 1.5
botrange_price = q1 - iqr * 1.5

print(toprange_price)
print(botrange_price)


# In[266]:


newdf3['PRICE'].fillna(newdf3['PRICE'].mean(), inplace=True)


# In[267]:


newdf3.head(10)


# In[ ]:


#'Very Good', 'Above Average', 'Good Quality', 'Excellent',
      # 'Average', 'Superior', 'Fair Quality', 'Exceptional-D',
     #  'Exceptional-C', 'Low Quality', 'Exceptional-A', 'Exceptional-B',
      # 'No Data', nan], dtype=object)


# In[284]:


df['AC'].unique()


# In[285]:


dfclean = newdf3.copy()
dfclean.head()


# In[286]:


df_no_missing = df.dropna()
df_no_missing


# In[316]:


dfclean = pd.get_dummies(data = newdf3)


# In[317]:


dfclean.head()


# In[287]:


df.fillna(0)


# In[ ]:


#'Very Good', 'Above Average', 'Good Quality', 'Excellent',
      # 'Average', 'Superior', 'Fair Quality', 'Exceptional-D',
     #  'Exceptional-C', 'Low Quality', 'Exceptional-A', 'Exceptional-B',
      # 'No Data', nan], dtype=object)


# In[ ]:





# In[299]:


def score_to_numeric(x):
    if x=='Y':
        return 1
    if x=='N':
        return 2
    if x=='0':
        return 3
    if x=='Very Good':
        return 1
    if x=='Above Average':
        return 2
    if x== 'Good Quality':
        return 3
    if x== 'Avereage':
        return 5
    if x== ' Superior':
        return 6
    if x=='Fair Quality':
        return 7
    if x== 'Exceptional-D':
        return 8
    if x== 'Exceptional-C':
        return 9
    if x == 'Low Quality': 
        return 10
    if x=='Exceptional-A':
        return 11
    if x== 'Exceptional-B':
        return 12
    
    if x=='Hot Water Rad':
        return 1
    if x=='Warm Cool':
        return 2
    if x=='Forced Air':
        return 3
    if x=='Ht Pump':
        return 4
    if x== 'Electric Rad':
        return 5
    if x== 'Elec Base Brd':
        return 6
    if x== ' Wall Furnace':
        return 7
    if x=='Water Base Brd':
        return 8
    if x== 'Evp Cool':
        return 9
    if x== 'Air Exchng':
        return 10
    if x == 'No Data': 
        return 11
    if x=='Ind Unit':
        return 12
    if x== 'Gravity Furnac':
        return 13
    if x== 'Air-Oil':
        return 14
    
   
    
dfclean['ACValue'] = dfclean['AC'].apply(score_to_numeric)
dfclean['GRADEValue']= dfclean['GRADE'].apply(score_to_numeric)
dfclean['HEATValue']= dfclean['HEAT'].apply(score_to_numeric)
dfclean.tail(5)


# In[300]:


dfclean.head(40)


# In[296]:


df['HEAT'].unique()


# In[ ]:


dfclean.head(40)


# In[307]:


cleanup2= dfclean.copy()


# In[308]:


cleanup2.drop(['GRADE','AC','HEAT'], axis=1, inplace=True)

cleanup2.head()


# In[319]:


corr = dfclean.corr()
sns.heatmap(corr, vmin=-1,annot=True)


# In[320]:


dfclean.describe()


# In[321]:


dfclean.columns


# In[ ]:


#I give up on try three :( sorry guys! 

