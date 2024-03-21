#!/usr/bin/env python
# coding: utf-8

# # Data Wrangling - Used Cars Pricing

# """After completing this you will be able to:
# 
#            -> Handle missing values
#            -> Correct data formatting
#            -> Standardize and normalize data
# """
# 

# ### What is the purpose of data wrangling?
# 
# You use data wrangling to convert data from an initial format to a format that may be better for analysis.

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
file_path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df = pd.read_csv(file_path, header = 0)
print(df.info())


# In[2]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df = pd.read_csv(file_path, names = headers)
print(df.head())


# In[3]:


import numpy as np
# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
print(df.head(5))


# # Evaluating for Missing Data

# In[4]:


missing_data = df.isnull()
missing_data.head(5)


# # Count missing values in each column

# In[5]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    


# """Based on the summary above, each column has 205 rows of data and seven of the columns containing missing data:
# 
# 1."normalized-losses": 41 missing data
# 2."num-of-doors": 2 missing data
# 3."bore": 4 missing data
# 4."stroke" : 4 missing data
# 5."horsepower": 2 missing data
# 6."peak-rpm": 2 missing data
# 7."price": 4 missing data
# """

# # Deal with missing data
# 
# How should you deal with missing data?
# 
# Drop data
# a. Drop the whole row
# b. Drop the whole column
# 
# Replace data............
# 
# a. Replace it by mean
# b. Replace it by frequency
# c. Replace it based on other functions

# # Replace by mean:
# 
# "normalized-losses": 41 missing data, replace them with mean
# "stroke": 4 missing data, replace them with mean
# "bore": 4 missing data, replace them with mean
# "horsepower": 2 missing data, replace them with mean
# "peak-rpm": 2 missing data, replace them with mean
# 

# # Replace by frequency:
# 
# "num-of-doors": 2 missing data, replace them with "four".
# Reason: 84% sedans are four doors. Since four doors is most frequent, it is most likely to occur

# # Drop the whole row:
# 
# "price": 4 missing data, simply delete the whole row
# Reason: You want to predict price. You cannot use any data entry without price data for prediction; 
#     therefore any row now without price data is not useful to you.

# Calculate the mean value for the "normalized-losses" column 

# In[6]:


avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)


# Replace "NaN" with mean value in "normalized-losses" column

# In[7]:


df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)


# Calculate the mean value for the "bore" column

# In[8]:


avg_bore = df["bore"].astype("float").mean(axis=0)
print("Average of bore", avg_bore)


# Replace "NaN" with mean value in "bore" column

# In[9]:


df["bore"].replace(np.nan, avg_bore, inplace=True)


# Calculate the mean vaule for "stroke" column

# In[10]:


avg_stroke = df["stroke"].astype('float').mean(axis=0)
print("Average of stroke:", avg_stroke)


# replace NaN by mean value in "stroke" column

# In[11]:


df["stroke"].replace(np.nan, avg_stroke, inplace=True)


# Calculate the mean value for the "horsepower" column

# In[12]:


avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)


# Replace "NaN" with the mean value in the "horsepower" column

# In[13]:


df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)


# Calculate the mean value for "peak-rpm" column

# In[14]:


avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)


# Replace "NaN" with the mean value in the "peak-rpm" column

# In[15]:


df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)


# To see which values are present in a particular column, we can use the ".value_counts()" method:

# In[16]:


df["num-of-doors"].value_counts()


# You can see that four doors is the most common type. We can also use the ".idxmax()" method to calculate the most common type automatically:

# In[17]:


df["num-of-doors"].value_counts().idxmax()


# Replace the missing 'num-of-doors' values by the most frequent 

# In[18]:


df["num-of-doors"].replace(np.nan, "four", inplace=True)


# In[19]:


df["num-of-doors"].value_counts()


# simply drop whole row with NaN in "price" column

# In[20]:


df.dropna(subset=["price"], axis=0, inplace=True)
# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)


# In[21]:


df.head()


# # Correct Data format

# .dtype() to check the data type
# 
# .astype() to change the data type

# In[22]:


print(df.info())
print(df)
#df.dtypes


# As you can see above, some columns are not of the correct data type. Numerical variables should have type 'float' or 'int', and variables with strings such as categories should have type 'object'. For example, the numerical values 'bore' and 'stroke' describe the engines, so you should expect them to be of the type 'float' or 'int'; however, they are shown as type 'object'. You have to convert data types into a proper format for each column using the "astype()" method.

# In[23]:


df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
print(df.info())


# # Data Standardization

# In[24]:


#L/100km = 235 / mpg
# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df["city-L/100km "]=235/df["city-mpg"]
# check your transformed data 
df.head()


# # Data Normalization

# Why normalization?
# 
# Normalization is the process of transforming values of several variables into a similar range. Typical normalizations include
# 
# scaling the variable so the variable average is 0
# scaling the variable so the variance is 1
# scaling the variable so the variable values range from 0 to 1
# Approach: replace the original value by (original value)/(maximum value)

# In[26]:


# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max() 

# show the scaled columns
df[["length","width","height"]].head()


# # Binning

# Binning is a process of transforming continuous numerical variables into discrete categorical 'bins' for grouped analysis.

# In[27]:


df["horsepower"]=df["horsepower"].astype(int, copy=True)


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[29]:


"""Find 3 bins of equal size bandwidth by using Numpy's linspace(start_value, end_value, numbers_generated function.

Since you want to include the minimum value of horsepower, set start_value = min(df["horsepower"]).

Since you want to include the maximum value of horsepower, set end_value = max(df["horsepower"]).

Since you are building 3 bins of equal length, you need 4 dividers, so numbers_generated = 4."""

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


# In[30]:


#Set group  names:
group_names = ['Low', 'Medium', 'High']


# In[31]:


#Apply the function "cut" to determine what each value of df['horsepower'] belongs to.
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)


# In[32]:


#See the number of vehicles in each bin:
df["horsepower-binned"].value_counts()


# In[33]:


#Plot the distribution of each bin:
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# # Bins Visualization

# In[72]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot


# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[51]:


df.to_csv('clean_df.csv')


# In[ ]:




