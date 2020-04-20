#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ### Goal: We’re interested in understanding our marketing effectiveness by channel and developing a spending allocation plan for next week’s advertising budget.

# # Part 1, Attribution
# Allocate conversions by channel and evaluate effectiveness

# In[2]:


data = pd.read_csv('attribution_allocation_student_data.csv')

def transfer_boolean(data):
    if data['convert_TF'] == True:
        return 1
    else: 
        return 0

data['convert_TF'] = data.apply(transfer_boolean, axis=1)
data = data[data['convert_TF']==1]
data.head()


# In[3]:


data.shape


# In[4]:


channel_spend_data = pd.read_csv('channel_spend_student_data.csv')
channel_spend_data = channel_spend_data.rename(columns={'spend by channel': 'spend_by_channel'})
channel_spend_data


# In[5]:


channel_spend_data.spend_by_channel.values


# In[6]:


spend = {'tier': ['1','2','3','total'], 'social': [50,100,150,300], 'organic_search': [0,0,0,0],
        'referral':[50,100,150,300],'email': [50,100,150,300], 'paid_search': [50,100,150,300], 
        'display': [50,100,150,300], 'direct': [0,0,0,0]}
spend_df = pd.DataFrame.from_dict(spend)
spend_df


# ## Test 3 methods for Attribution Modeling

# ### Method 1: First Interaction
# Because we are a new clothing business, we would want to see how we get initial awareness.
# The First Interaction model attributes 100% of the conversion value to the first channel with which the customer interacted.

# In[7]:


data_touch_1 = data[['convert_TF','touch_1','tier']]
data_touch_1.head()


# In[9]:


df_t1 = data_touch_1.touch_1.value_counts().to_frame(name="sum")

def pct(value):
    pct = value/2378
    return pct

df_t1['pct'] = df_t1['sum'].apply(pct)
df_t1 = df_t1.iloc[1:,]
df_t1


# Based on our results from First Interaction model, the top three channels that contributes the most conversions are Organic Search (28.3%), Direct (23.1%), and Display (18.3%).  

# In[10]:


cac_t1 = data_touch_1.groupby(['touch_1','tier']).size().to_frame(name='tier_count')
spend_list = [0,0,0,50,100,150,50,100,150,0,0,0,50,100,150,50,100,150,50,100,150]
cac_t1['spend'] = spend_list
cac_t1['CAC'] = cac_t1['spend']/cac_t1['tier_count']
cac_t1


# ### Method 2: Last Interaction
# 
# We're interested in how we got our customers to convert to decide which are the effective last touches. The Last Interaction model attributes 100% of the conversion value to the last channel with which the customer interacted before buying or converting.

# In[11]:


data.isna().sum()


# In[12]:


def last_inter(row):
    if pd.notna(row['touch_5']):
        return "touch_5_" + row['touch_5']
    elif pd.notna(row['touch_4']):
        return "touch_4_" + row['touch_4']
    elif pd.notna(row['touch_3']):
        return "touch_3_" + row['touch_3']
    elif pd.notna(row['touch_2']):
        return "touch_2_" + row['touch_2']
    else: 
        return "touch_1_" + row['touch_1']
    
data['last_int'] = data.apply(last_inter, axis=1)
data.head()


# In[13]:


data_last_int = data.last_int.value_counts().to_frame(name='sum')

def pct(value):
    pct = value/2378
    return pct

data_last_int['pct'] = data_last_int['sum'].apply(pct)
data_last_int


# Looking at both the stages of interaction ands types of channel, touch_3_direct (10,8%), touch_3_organic_search (10.3%), touch_3_display (6.7%) created the most conversions. 

# In[14]:


channel_list = []

for index, row in data_last_int.iterrows():
    if 'direct' in index:
        channel_list.append('direct')
    elif 'display' in index:
        channel_list.append('display')
    elif 'email' in index:
        channel_list.append('email')
    elif 'organic' in index:
        channel_list.append('organic_search')
    elif 'paid' in index:
        channel_list.append('paid_search')
    elif 'referral' in index:
        channel_list.append('referral')
    else:
        channel_list.append('social')

data_last_int['channel'] = channel_list
data_last_int.head()


# In[15]:


data_last_int.groupby(['channel']).sum()


# Based on the results from Last Interactions model, the top 3 channels that have most converted customers are Direct (25.8%), Display (17.1%), and Email (13.6%).

# In[16]:


cac_last_t = data[['convert_TF','tier','last_int']]

channel_list = []

for index, row in cac_last_t.iterrows():
    if 'direct' in row['last_int']:
        channel_list.append('direct')
    elif 'display' in row['last_int']:
        channel_list.append('display')
    elif 'email' in row['last_int']:
        channel_list.append('email')
    elif 'organic' in row['last_int']:
        channel_list.append('organic_search')
    elif 'paid' in row['last_int']:
        channel_list.append('paid_search')
    elif 'referral' in row['last_int']:
        channel_list.append('referral')
    else:
        channel_list.append('social')

cac_last_t['channel'] = channel_list
cac_last_t


# In[17]:


cac_last_t_gb = cac_last_t.groupby(['channel','tier']).size().to_frame(name='tier_count')
spend_list = [0,0,0,50,100,150,50,100,150,0,0,0,50,100,150,50,100,150,50,100,150]
cac_last_t_gb['spend'] = spend_list
cac_last_t_gb['CAC'] = cac_last_t_gb['spend']/cac_last_t_gb['tier_count']
cac_last_t_gb


# ### Method 3: Last Non-Direct Interaction
# We assumed few customers can directly find our website as we are a new brand and not very well-known yet. The Last Non-Direct Click model ignores direct traffic and attributes 100% of the conversion value to the last channel that the customer clicked through from before buying or converting.

# In[92]:


nondir_list = []

#def last_nondir_inter(row):
for index, row in data.iterrows():
    if (pd.notna(row['touch_5']) and 'direct' not in row['touch_5']):
        nondir_list.append("touch_5_" + row['touch_5'])
    elif (pd.notna(row['touch_4']) and 'direct' not in row['touch_4']):
        nondir_list.append("touch_4_" + row['touch_4'])
    elif (pd.notna(row['touch_3']) and 'direct' not in row['touch_3']):
        nondir_list.append("touch_3_" + row['touch_3'])
    elif (pd.notna(row['touch_2']) and 'direct' not in row['touch_2']):
        nondir_list.append("touch_2_" + row['touch_2'])
    else: 
        nondir_list.append("touch_1_" + row['touch_1'])
    
data['last_nondir_int'] = nondir_list
data.head()


# In[93]:


data_last_nondir_int = data.last_nondir_int.value_counts().to_frame(name='sum')

def pct(value):
    pct = value/2378
    return pct

data_last_nondir_int['pct'] = data_last_nondir_int['sum'].apply(pct)
data_last_nondir_int.head()


# In[71]:


channel_list = []

for index, row in data_last_nondir_int.iterrows():
    if 'direct' in index:
        channel_list.append('direct')
    elif 'display' in index:
        channel_list.append('display')
    elif 'email' in index:
        channel_list.append('email')
    elif 'organic' in index:
        channel_list.append('organic_search')
    elif 'paid' in index:
        channel_list.append('paid_search')
    elif 'referral' in index:
        channel_list.append('referral')
    else:
        channel_list.append('social')

data_last_nondir_int['channel'] = channel_list
data_last_nondir_int.head()


# In[72]:


data_last_nondir_int.groupby(['channel']).sum()


# The result of Last non-direct interaction is quite different from that of Last interaction, with Organic Search (37%), Display (22%), and Social (19.1%) being the top three channels.

# In[83]:


cac_last_nd_t = data[['convert_TF','tier','last_nondir_int']]

channel_list = []

for index, row in cac_last_nd_t.iterrows():
    if 'direct' in row['last_nondir_int']:
        channel_list.append('direct')
    elif 'display' in row['last_nondir_int']:
        channel_list.append('display')
    elif 'email' in row['last_nondir_int']:
        channel_list.append('email')
    elif 'organic' in row['last_nondir_int']:
        channel_list.append('organic_search')
    elif 'paid' in row['last_nondir_int']:
        channel_list.append('paid_search')
    elif 'referral' in row['last_nondir_int']:
        channel_list.append('referral')
    else:
        channel_list.append('social')

cac_last_nd_t['channel'] = channel_list
cac_last_nd_t


# In[87]:


cac_last_nd_t_gb = cac_last_nd_t.groupby(['channel','tier']).size().to_frame(name='tier_count')
spend_list = [0,0,0,50,100,150,50,100,150,0,0,0,50,100,150,50,100,150,50,100,150]
cac_last_nd_t_gb['spend'] = spend_list
cac_last_nd_t_gb['CAC'] = cac_last_nd_t_gb['spend']/cac_last_nd_t_gb['tier_count']
cac_last_nd_t_gb


# ### Conclusions from CAC Calculations

# As the results across all three methods are aligned, we can conclude that the most effective channels are Display, Social, and Email, with their CACs are all below ＄1 dollar, which means we spent less than 1 dollar to acquire a customer. While Paid Search and Referral are the least effective ones as their CACs are between ＄20-＄50 dollars. That is, we spent ＄20-＄50 dollars to acquire one customer. Therefore, we should consider reduce the spend on them and reallocate it to other more effective channels.

# ## Part 2, Allocation
# For one of the allocation methods, calculate the marginal CAC by spending tier by channel

# In[105]:


# Chose Last Interaction model to record the conversions by tier by channel

cac_last_nd_t.groupby(["channel","tier"]).size().to_frame(name="count")


# In[120]:


spend_df_cac = spend_df.iloc[:-1,:]

spend_df_cac['social_m_spend'] = [50,50,50]
spend_df_cac['social_m_conv'] = [83,166-83,205-166]
spend_df_cac['social_m_CAC'] = spend_df_cac['social_m_spend']/spend_df_cac['social_m_conv']

spend_df_cac['referral_m_spend'] = [50,50,50]
spend_df_cac['referral_m_conv'] = [2,2,4]
spend_df_cac['referral_m_CAC'] = spend_df_cac['referral_m_spend']/spend_df_cac['referral_m_conv']

spend_df_cac['email_m_spend'] = [50,50,50]
spend_df_cac['email_m_conv'] = [80,50,216-130]
spend_df_cac['email_m_CAC'] = spend_df_cac['email_m_spend']/spend_df_cac['email_m_conv']

spend_df_cac['paid_search_m_spend'] = [50,50,50]
spend_df_cac['paid_search_m_conv'] = [3,-1,7]
spend_df_cac['paid_search_m_CAC'] = spend_df_cac['paid_search_m_spend']/spend_df_cac['paid_search_m_conv']

spend_df_cac['display_m_spend'] = [50,50,50]
spend_df_cac['display_m_conv'] = [100,99,226-199]
spend_df_cac['display_m_CAC'] = spend_df_cac['display_m_spend']/spend_df_cac['display_m_conv']

spend_df_cac = spend_df_cac[['tier','social_m_CAC','referral_m_CAC','email_m_CAC','paid_search_m_CAC','display_m_CAC']]

spend_df_cac


# Compared to the findings in part 1, Social, Email and Display still remained effective, but the difference among the tiers for these channels became more obvious. Tier 2 has the highest marginal CAC for Email, whereas tier 3 has the highest marginal CAC for Social and Display. Referral has the biggest difference among its tiers, and Paid Search has a negative ＄50 for its second tier, suggesting that we should stop investing after the first tier.

# #### This week we spent ＄250 total on advertising across all platforms. Next week we want to allocate the same budget in ＄50 increments.

# In[121]:


spend_df_budget = spend_df_cac.copy()

spend_df_budget['social_m_spend'] = [50,50,0]
spend_df_budget['email_m_spend'] = [50,0,0]
spend_df_budget['display_m_spend'] = [50,50,0]

spend_df_budget['social_exp_conv'] = spend_df_budget['social_m_spend']/spend_df_budget['social_m_CAC']
spend_df_budget['email_exp_conv'] = spend_df_budget['email_m_spend']/spend_df_budget['email_m_CAC']
spend_df_budget['display_exp_conv'] = spend_df_budget['display_m_spend']/spend_df_budget['display_m_CAC']

spend_df_budget


# In[123]:


spend_df_budget[['social_exp_conv','email_exp_conv','display_exp_conv']].sum()


# With the constraint of ＄50 increments, I decided to invest ＄100 for both Display and Social, and ＄50 for Email. After that, I will expect to get a total of 445 conversions.
