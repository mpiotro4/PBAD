#!/usr/bin/env python
# coding: utf-8

# Importing modules

# In[ ]:


import numpy as np
import pandas as pd


# Defining custom functions

# In[21]:


def display_columns():
  print("Number of columns: " + str(len(df.columns)))
  print(df.columns)


# In[22]:


def display_summary():
  columns=list(df.columns)
  columns_cnt=len(columns)
  rows_cnt=len(df.index)

  print("Number of rows: " + str(rows_cnt))
  print("Number of columns: " + str(columns_cnt))
  print("Used columns: ")
  print(columns)


# In[23]:


def rename_columns():
  new_headers = {
      # Closures & containment
      "C1M_School closing" : "C1M",
      "C2M_Workplace closing" : "C2M",
      "C3M_Cancel public events" : "C3M",
      "C4M_Restrictions on gatherings" : "C4M",
      "C5M_Close public transport" : "C5M",
      "C6M_Stay at home requirements" : "C6M",
      "C7M_Restrictions on internal movement" : "C7M",
      "C8EV_International travel controls" : "C8M",

      # Health measures
      "H1_Public information campaigns" : "H1"
  }

  df.rename(columns=new_headers, inplace=True)


# In[24]:


def filter_columns():
  # Dropping not used columns with region information & other
  region_related_cls=["CountryCode", "RegionName", "RegionCode", "Jurisdiction", "M1_Wildcard", "StringencyIndex_Average_ForDisplay"]
  df.drop(columns=region_related_cls, inplace=True)

  # Defining regexes to filter not used columns indices
  economic_filter='^E[1-4]_*'
  health_system_filter='^H[2-8][M]?_*'
  vaccination_filter='^V[1-4][A-Z]?_*'
  GRI_filter='GovernmentResponseIndex*'
  CHI_filter='ContainmentHealthIndex*'
  ESI_filter='EconomicSupportIndex*'
  filters=[economic_filter, health_system_filter, vaccination_filter, GRI_filter, CHI_filter, ESI_filter]

  for filter in filters:
    df.drop(columns=df.filter(regex=filter), inplace=True)


# In[25]:


def add_previous_stringency_index():
    df['Prev_StringencyIndex_Average']=0

    for i in range(1, len(df)):
      # First row from certain country
      if df.at[i, 'CountryName'] != df.at[i-1, 'CountryName']:
        df.loc[i, 'Prev_StringencyIndex_Average'] = 0.0
      else:
        df.loc[i, 'Prev_StringencyIndex_Average'] = df.loc[i-1, 'StringencyIndex_Average']


# In[26]:


def get_npi_difference():
  # Add column for Stringency Index diff value
  df['Daily_StringencyIndex_Change']=0

  for i in range(1, len(df)):
    # Create np arrays for calculating Stringency Index diff
    npis=np.zeros((1,0))
    npis_flag=np.zeros((1,0))

    # First row from certain country
    if df.at[i, 'CountryName'] != df.at[i-1, 'CountryName']:

      for npi in subset:
        npis = np.append(npis, [df.at[i, npi]])
      for flag in flag_subset:
        npis_flag = np.append(npis_flag, [df.at[i, flag]])
    
    else:
      # Check whether any NPI value has changed
      for npi in subset:
        if df.at[i-1, npi] != df.at[i, npi]:
          npis = np.append(npis, [df.at[i, npi]])
        else:
          npis = np.append(npis, [0])

      # Check whether any NPI flag has changed
      for flag in flag_subset:
        if df.at[i-1, flag] != df.at[i, flag]:
          npis_flag = np.append(npis_flag, [df.at[i, flag]])
        else: 
          npis_flag = np.append(npis, [0])

      daily_si=calculate_simplified_si(npis)
      df.at[i, 'Daily_StringencyIndex_Change'] = daily_si


# In[27]:


# Do not consider flags while calculating Stringency Index
def calculate_simplified_si(npis):
  # Max values of npis
  max_npi_values=[3,3,2,4,2,3,2,4,2]                                    
  si=0

  for i in range(0, len(npis)):
    si = si + (npis[i] / max_npi_values[i])
    
  si=si/len(npis) 
  return si


# In[28]:


def calculate_si():
  pass


# **[STEP 1]** Read data from CSV

# In[29]:


path="./OxCGR_nat_latest.csv"

dict=pd.read_csv(path)
df=pd.DataFrame(dict)

display_summary()


# **[STEP 2]** Deleting not used and irrelevant columns

# In[30]:


filter_columns()


# **[STEP 3]** Drop rows with NPI=Nan

# In[31]:


df = df[df['StringencyIndex_Average'].notna()]


# **[STEP 4]** Rename columns to improve readibility

# In[32]:


rename_columns()


# **[STEP 5]** Deleting rows where none of NPIs are defined

# In[33]:


subset=["C1M", "C2M", "C3M", "C4M", "C5M", "C6M", "C7M", "C8M", "H1"]
flag_subset=["C1M_Flag", "C2M_Flag", "C3M_Flag", "C4M_Flag", "C5M_Flag", "C6M_Flag", "C7M_Flag", "H1_Flag"]
df.dropna(subset=flag_subset, how='all', inplace=True)
df.fillna(0, inplace=True)


# **[STEP 6]** Delete redundant rows

# In[34]:


compared_columns=df.columns.tolist()

compared_columns.remove('Date')
compared_columns.remove('ConfirmedCases')
compared_columns.remove('ConfirmedDeaths')
compared_columns.remove('MajorityVaccinated')
compared_columns.remove('PopulationVaccinated')

df.drop_duplicates(subset=compared_columns, inplace=True)


# **[STEP 7]** Copy Stringency Index from previous day 

# In[35]:


df.reset_index(drop=True, inplace=True)
add_previous_stringency_index()


# **[STEP 8]** Calculate Stringency Index of NPI established certain day

# In[36]:


get_npi_difference()


# **[STEP 9]** Show results

# In[37]:


df.head()
print(df['Daily_StringencyIndex_Change'].max())


# **[STEP 10]** Save DataFrame with results to csv

# In[38]:


df.to_csv('OxCGRT_clean.csv', encoding = 'utf-8-sig')

