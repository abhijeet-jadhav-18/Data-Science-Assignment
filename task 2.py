#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# In[38]:


customers_df = pd.read_csv("Customers.csv")
products_df = pd.read_csv("Products.csv")
transactions_df = pd.read_csv("Transactions.csv")


# In[39]:


customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])


# In[40]:


region_encoder = OneHotEncoder(sparse=False)
region_encoded = region_encoder.fit_transform(customers_df[['Region']])


# In[41]:


# Preprocessing Transactions Data
# Merge Transactions with Product Information


# In[42]:


transactions_merged = transactions_df.merge(products_df, on='ProductID', how='left')


# In[43]:


# Aggregate transaction data for customers
transaction_summary = transactions_merged.groupby('CustomerID').agg({
    'TotalValue': 'sum',  # Total spending
    'Quantity': 'sum',   # Total quantity purchased
    'Price_y': 'mean',   # Average price of products purchased
    'Category': lambda x: ','.join(x.unique())  # Unique categories purchased
}).reset_index().rename(columns={'Price_y': 'AveragePrice'})


# In[44]:


customer_profiles = customers_df.merge(transaction_summary, on='CustomerID', how='left')


# In[45]:


customer_profiles[['TotalValue', 'Quantity', 'AveragePrice']] = customer_profiles[['TotalValue', 'Quantity', 'AveragePrice']].fillna(0)


# In[46]:


categories_split = customer_profiles['Category'].fillna('None').str.split(',')


# In[47]:


# Use MultiLabelBinarizer to one-hot encode the multi-label data
category_encoder = MultiLabelBinarizer()
categories_encoded = category_encoder.fit_transform(categories_split)


# In[48]:


print(categories_encoded.shape)


# In[49]:


features = np.hstack([
    region_encoded,  # Encoded regions
    customer_profiles[['TotalValue', 'Quantity', 'AveragePrice']].values,  # Numeric transaction data
    categories_encoded  # Encoded categories
])


# In[50]:


# Normalize the features using Min-Max Scaling
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)


# In[51]:


similarity_matrix = cosine_similarity(features_normalized)


# In[52]:


customer_ids = customer_profiles['CustomerID']
lookalike_map = {}


# In[53]:


for i in range(20):  # For the first 20 customers
    customer_index = i
    similar_indices = np.argsort(similarity_matrix[customer_index])[::-1][1:4]  # Top 3 similar (excluding self)
    similar_customers = [
        (customer_ids[idx], round(similarity_matrix[customer_index, idx], 4))
        for idx in similar_indices
    ]
    lookalike_map[customer_ids[customer_index]] = similar_customers


# In[54]:


# Prepare Lookalike Map DataFrame for CSV export
lookalike_list = [
    {"cust_id": cust_id, "lookalikes": str(lookalikes)}
    for cust_id, lookalikes in lookalike_map.items()
]

lookalike_df = pd.DataFrame(lookalike_list)


# In[55]:


# Save the Lookalike Map to a CSV
lookalike_df.to_csv("Lookalike.csv", index=False)


# In[56]:


print("Lookalike model has been successfully built! Output saved to Lookalike.csv.")


# In[ ]:





# In[ ]:




