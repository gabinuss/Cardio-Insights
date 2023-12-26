#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#display all columns in DataFrame without truncation
pd.set_option('display.max_columns',None)


# In[3]:


# Reading data from an Excel file into a Pandas DataFrame
df = pd.read_excel('hospital admissions data.xlsx')

# Displaying the first few rows of the DataFrame to get an overview of the data
df.head()


# In[4]:


#DATA CLEANING

# Displaying information about the DataFrame including data types and non-null counts
df.info()


# In[5]:


#DATA CLEANING

# Calculating the percentage of missing values for each column in the DataFrame
missing_percentage = (df[df.columns[df.isna().sum() > 0]].isna().sum() / len(df) * 100).sort_values()

# Extracting the column names with missing values
columns_with_missing_values = missing_percentage.index

# Creating a horizontal bar plot to visualize the missing data percentages
plt.barh(columns_with_missing_values, missing_percentage)


# In[6]:


#DATA CLEANING
# Converting specific columns to numeric data type, and converting non-numeric values to NaN
df['BNP'] = pd.to_numeric(df['BNP'],errors='coerce')
df['EF'] = pd.to_numeric(df['EF'],errors='coerce')
df['GLUCOSE'] = pd.to_numeric(df['GLUCOSE'],errors='coerce')
df['TLC'] = pd.to_numeric(df['TLC'],errors='coerce')
df['PLATELETS'] = pd.to_numeric(df['PLATELETS'],errors='coerce')
df['HB'] = pd.to_numeric(df['HB'],errors='coerce')
df['CREATININE'] = pd.to_numeric(df['CREATININE'],errors='coerce')
df['UREA'] = pd.to_numeric(df['UREA'],errors='coerce')


# In[7]:


#DATA CLEANING
# Exploring the correlation of columns to determine the best strategy for data cleaning
# Drop rows with missing values and convert specific columns to integer type
df_dropped = df.dropna()
columns_to_convert = ['BNP', 'EF', 'GLUCOSE', 'TLC', 'PLATELETS', 'HB', 'CREATININE', 'UREA']
df_dropped[columns_to_convert] = df_dropped[columns_to_convert].astype(int)


# In[8]:


#DATA CLEANING
# Select only numeric columns in the cleaned DataFrame
df_dropped1 = df_dropped.select_dtypes(include='number')


# In[9]:


#DATA CLEANING
# Calculate the correlation matrix
corr_matrix = df_dropped1.corr()

# Set the correlation threshold
correlation_threshold = 0.5


# In[10]:


#DATA CLEANING
# Filter values with correlation greater than 0.5 or less than -0.5
high_correlated_values = corr_matrix[(((corr_matrix > correlation_threshold) & (corr_matrix != 1))
                                      | ((corr_matrix < -correlation_threshold) & (corr_matrix != -1)))]

# Display the first few rows of the high correlated values DataFrame
high_correlated_values.head()


# In[11]:


#DATA CLEANING

# Grab columns with at least one non-null value
high_correlated_columns = high_correlated_values.columns[high_correlated_values.notna().any()]


# In[12]:


#DATA CLEANING
# Extract the high correlated columns from the cleaned DataFrame
high_correlated = df_dropped1[high_correlated_columns]
high_correlated.head()


# In[13]:


#DATA CLEANING
# Plot a heatmap of the correlation matrix for high correlated columns
plt.figure(figsize=(10, 9))
sns.heatmap(high_correlated.corr(), annot=True, cmap='coolwarm')


# In[14]:


#The objective of the analysis lies on the premise of discharge(healing) or expiry(death).
#Therefore DAMA (Discharge Against Medical Advise) is a distortion to the data and should be removed.
df = df[df['OUTCOME'] != 'DAMA']


# In[15]:


#HANDLING MISSING VALUES

#Given that BNP does not exhibit a high correlation with any other column 
#and the percentage of missing values exceeds 50%, the decision is to drop the column.
df = df.drop('BNP', axis=1)


# In[16]:


#EF
#EJECTILE FUNCTION (The amount of blood that your heart pumps each time it beats)
df['EF'] = pd.to_numeric(df['EF'],errors='coerce')


# In[47]:


df['EF'].isna().sum()


# In[48]:


(df[['EF','PRIOR CMP']].sort_values(by = 'EF',ascending = False)).value_counts()


# In[17]:


#In the analysis, we observe a moderate correlation between EF and prior CMD. 
#We notice instances where prior CMD is 0 correspond to an EF of 60. 
#Consequently, for missing EF values, we plan to impute them using the mode, which is determined to be 60.
df['EF']=df['EF'].fillna(df['EF'].mode()[0])


# In[50]:


df['EF'].isna().sum()


# In[51]:


#GLUCOSE
df['GLUCOSE'].isna().sum()


# In[52]:


df['GLUCOSE'].head(10)


# In[53]:


#In the glucose column, most values were observed to be close to each other. 
#As a result, the decision is  to use forward fill to fill in the missing values. 
#Forward fill involves replacing each missing value with the most recent non-null value in the column,
#allowing for a smoother continuity of values in the dataset. 
df['GLUCOSE'] = df['GLUCOSE'].fillna(method='ffill')


# In[54]:


df['GLUCOSE'].isna().sum()


# In[55]:


#tlc
df['TLC'].head(10)


# In[25]:


#In the TLC column, most values were observed to be close to each other. 
#As a result, the decision is  to use forward fill to fill in the missing values. 
#allowing for a smoother continuity of values in the dataset. 
df['TLC'] = df['TLC'].fillna(method='ffill')


# In[29]:


df['TLC'].isna().sum()


# In[27]:


#PLATELETS
df['PLATELETS'].isna().sum()


# In[59]:


#In the PLATELETS column, most values were observed to be close to each other. 
#As a result, the decision is  to use backwardfill to fill in the missing values. 
#allowing for a smoother continuity of values in the dataset. 
df['PLATELETS'] = df['PLATELETS'].fillna(method='bfill')



# In[28]:


df['PLATELETS'].isna().sum()


# In[30]:


#HB
# Analyzing the correlation between 'HB' and 'ANAEMIA'
# Higher hemoglobin levels are correlated with a lower likelihood of anemia
df['HB'] = pd.to_numeric(df['HB'],errors='coerce')
sns.scatterplot(x='HB',y='ANAEMIA',data=df,hue='HB',palette='viridis',alpha=1.0)


# In[62]:


# utilizes the groupby operation to calculate the median for each group
df_blood = df[['HB','ANAEMIA']]
df_blood.groupby('ANAEMIA').median()


# In[31]:


# Define a function to replace null values in 'HB' based on 'ANAEMIA' value
def replace_hb(row):
    # Check if 'HB' is null
    if pd.isna(row['HB']):
        # If 'ANAEMIA' is 0, replace null 'HB' with 12.9
        if row['ANAEMIA'] == 0:
            return 12.9
        # If 'ANAEMIA' is 1, replace null 'HB' with 9.0
        elif row['ANAEMIA'] == 1:
            return 9.0
    # If 'HB' is not null, return the original 'HB' value
    return row['HB']

# Apply the replace_hb function to each row using the apply function
df['HB'] = df.apply(replace_hb, axis=1)


# In[32]:


df['HB'].isna().sum()


# In[33]:


#'CREATININE'
df['CREATININE'].isna().sum()


# In[34]:


#it is evident that there exists a positive correlation between creatinine levels and the severity of CKD. 
#As creatinine is a waste product normally excreted by the kidneys, 
#elevated levels of creatinine in the blood suggest reduced kidney function

df['CREATININE'] = pd.to_numeric(df['CREATININE'],errors='coerce')
sns.scatterplot(x='CREATININE',y='CKD',data=df,hue='CKD',palette='viridis',alpha=1.0)


# In[35]:


# Counting occurrences of unique pairs in the 'CREATININE' and 'CKD' columns
#'CKD' is a binary column indicating the presence (1) or absence (0) of chronic kidney disease
df[['CREATININE','CKD']].value_counts()


# In[36]:


#Grouping the DataFrame by the 'CKD' column and calculating the median of 'CREATININE' for each group
df[['CREATININE','CKD']].groupby('CKD').median()


# In[37]:


# Define a function to replace null values in CREATININE based on CKD value
def replace_CREATININE(row):
    # Check if CREATININE is null
    if pd.isna(row['CREATININE']):
        # If CKD is 0, replace null 'CREATININE' with 0.9
        if row['CKD'] == 0:
            return  0.9
        # If CKD is 1, replace null 'CREATININE' with 3.5
        elif row['CKD'] == 1:
            return 3.5
    # If CREATININE is not null, return the original CREATININE value
    return row['CREATININE']

# Apply the replace_CKD function to each row using the apply function
df['CREATININE'] = df.apply(replace_CREATININE, axis=1)


# In[38]:


df['CREATININE'].isna().sum()


# In[39]:


#UREA
df['UREA'].isna().sum()


# In[40]:


#Creatinine and UREA are among the substances that the kidneys filter out, 
#and their concentrations in the blood are correlated 

df['UREA'] = pd.to_numeric(df['UREA'],errors='coerce')
sns.scatterplot(x='UREA',y='CREATININE',data=df)


# In[41]:


# Extracting a subset DataFrame with 'CREATININE' and 'UREA'
df_urea = df[['CREATININE', 'UREA']]

# Counting occurrences of unique 'CREATININE' values for rows where 'UREA' is NaN
urea_nan_creatinine_counts = df_urea[df_urea['UREA'].isna()]['CREATININE'].value_counts()
urea_nan_creatinine_counts


# In[43]:


# Calculating the median of 'UREA' values where 'CREATININE' is equal to 0.90
urea_median = df_urea[df_urea['CREATININE'] == 0.90]['UREA'].median()

# Filling missing values in the original DataFrame's 'UREA' column with the calculated median
df['UREA'] = df['UREA'].fillna(urea_median)


# In[44]:


df['UREA'].isna().sum()


# In[77]:


#confirming there are no missing values
df.isna().sum()


# In[45]:


#noticed invalid entry in the chest infection column 
df['CHEST INFECTION'].unique()


# In[46]:


df[df['CHEST INFECTION']=='\\']


# In[47]:


# Converting 'CHEST INFECTION' column to numeric and  '\\' to NaN
df['CHEST INFECTION'] = pd.to_numeric(df['CHEST INFECTION'], errors='coerce')

# Filling missing values using forward fill method
df['CHEST INFECTION'] = df['CHEST INFECTION'].fillna(method='ffill')


# In[60]:


#Checking for duplicates


# In[49]:


#Display rows that are duplicates 
df[df.duplicated()]


# In[57]:


#FEATURE ENGINEERING


# In[50]:


# Converting 'D.O.A' (Date of Admission) column to datetime format, coercing errors to NaN
df['D.O.A'] = pd.to_datetime(df['D.O.A'], errors='coerce')

# Converting 'D.O.D' (Date of Discharge) column to datetime format, coercing errors to NaN
df['D.O.D'] = pd.to_datetime(df['D.O.D'], errors='coerce')


# In[51]:


# Extracting year, month, and day components from the Date of Admission
df['arrival_year']=df['D.O.A'].dt.year
df['arrival_month']=df['D.O.A'].dt.month
df['arrival_day']=df['D.O.A'].dt.day


# In[52]:


# Extracting year, month, and day components from the Date of Discharge
df['discharge_year']=df['D.O.D'].dt.year
df['discharge_month']=df['D.O.D'].dt.month
df['discharge_day']=df['D.O.D'].dt.day


# In[62]:


#VISUALIZATION


# In[53]:


Admission_Triage_data = df[['SMOKING ', 'ALCOHOL','DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD','RAISED CARDIAC ENZYMES',
'SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA', 'ACS', 'STEMI',
'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF', 'VALVULAR',
'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT', 'PSVT',
'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC',
'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK',
'PULMONARY EMBOLISM', 'CHEST INFECTION']]


# In[54]:


# Counting how many times each disease appears
sorted_values = Admission_Triage_data.apply(lambda x: x.value_counts()).iloc[1]
sorted_values


# In[55]:


sorted_values.nlargest(10).plot(kind='bar', color='green')
plt.xlabel('Disease')
plt.ylabel('Number of People with these symptoms and conditions')
plt.title('Commonly Admitted Diseases')


# In[56]:


# Grouping data by 'arrival_month' and 'arrival_year' and counting occurrences
# Creating a new DataFrame 'grouped_data' by grouping based on 'arrival_month' and 'arrival_year'
# The 'size()' function counts the number of occurrences in each group
# The 'reset_index()' function resets the index of the resulting DataFrame
# The 'name='count'' parameter assigns a name 'count' to the new column containing the counts
grouped_data = df.groupby(['arrival_month', 'arrival_year']).size().reset_index(name='count')
grouped_data 


# In[70]:


# Dictionary to map numerical month values to month names
months = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
          7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

# Creating a new column 'admission_month' by mapping 'arrival_month' values to month names
grouped_data['admission_month'] = grouped_data['arrival_month'].map(months)


# In[71]:


plt.figure(figsize=(12,3))
sns.lineplot(x='admission_month', y='count', hue='arrival_year', data=grouped_data,marker='o',palette='Set1')
plt.ylabel('Number of Admissions')
plt.title('Monthly Distribution of Admissions Over the Years')


# In[72]:


#Top 10 symptoms and underlying conditions of heart-related admission and the number of deaths analyzed by gender 


# In[98]:


diseases_outcome =df[['SMOKING ', 'ALCOHOL','DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD','RAISED CARDIAC ENZYMES',
'SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA', 'ACS', 'STEMI',
'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF', 'VALVULAR',
'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT', 'PSVT',
'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC',
'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK',
'PULMONARY EMBOLISM', 'CHEST INFECTION','OUTCOME','GENDER']]
#Filter rows where outcome is expiry
dead_patients = diseases_outcome[diseases_outcome['OUTCOME']=='EXPIRY']


# In[99]:


# Sum the occurrences for each disease by gender
deaths_by_disease_gender = dead_patients.groupby(['GENDER']).sum().drop(['OUTCOME'], axis=1)


# In[100]:


# Get the top 10 diseases with the most deaths overall
top_10_diseases = deaths_by_disease_gender.sum().nlargest(10)


# In[101]:


# Filter the DataFrame to include only the top 10 diseases
top_10_deadly_diseases = dead_patients[['GENDER'] + list(top_10_diseases.index)]


# In[102]:


# Melt the DataFrame to long format for better plotting
melted_data = pd.melt(top_10_deadly_diseases, id_vars=['GENDER'], var_name='Disease', value_name='Value')


# In[96]:


melted_data


# In[117]:


plt.figure(figsize=(12,4))
sns.barplot(x='Disease', y='Value', hue='GENDER', data=melted_data,estimator=sum)
plt.xlabel('Risk Factors')
plt.ylabel('Number of Deaths')
plt.title('Top 10 Deadly Risk Factors by Gender')
plt.xticks(rotation=60)
plt.show()


# In[ ]:


#Trend in deaths from heart conditions over the years 


# In[86]:


diseases_year =df[['SMOKING ', 'ALCOHOL','DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD','RAISED CARDIAC ENZYMES',
'SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA', 'ACS', 'STEMI',
'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF', 'VALVULAR',
'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT', 'PSVT',
'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC',
'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK',
'PULMONARY EMBOLISM', 'CHEST INFECTION','OUTCOME','arrival_year']]
#Filter rows where outcome is expiry
dead_patients = diseases_year[diseases_year['OUTCOME']=='EXPIRY']


# In[87]:


# Sum the occurrences for each disease per year
deaths_by_disease_year = dead_patients.groupby(['arrival_year']).sum().drop(['OUTCOME'], axis=1)


# In[88]:


# Get the top 10 diseases with the most deaths overall
top_10_diseases = deaths_by_disease_year.sum().nlargest(10)


# In[89]:


# Filter the DataFrame to include only the top 10 diseases
top_10_deadly_diseases = dead_patients[['arrival_year'] + list(top_10_diseases.index)]


# In[90]:


# Melt the DataFrame to long format for better plotting
melted_data = pd.melt(top_10_deadly_diseases, id_vars=['arrival_year'], var_name='Disease', value_name='Value')


# In[92]:


plt.figure(figsize=(10,4))
sns.lineplot(x='arrival_year', y='Value', data=melted_data,estimator=sum, color='green')
plt.xlabel('Admission_year')
plt.ylabel('Number of deaths')
plt.title('Trend in deaths over the years ')


# In[ ]:


#Disease distribution across age groups 


# In[104]:


#distribution of age
df['AGE'].describe()


# In[105]:


diseases_age =df[['SMOKING ', 'ALCOHOL','DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD','RAISED CARDIAC ENZYMES',
'SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA', 'ACS', 'STEMI',
'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF', 'VALVULAR',
'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT', 'PSVT',
'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC',
'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK',
'PULMONARY EMBOLISM', 'CHEST INFECTION','AGE']]


# In[106]:


# Create age groups
bins = [0, 40, 60, 80, float('inf')]
labels = ['0-40', '41-60', '61-80', '81+']
diseases_age['Age_Group'] = pd.cut(diseases_age['AGE'], bins=bins, labels=labels, right=False)


# In[107]:


# Drop the original 'AGE' column to avoid redundancy in the heatmap
diseases_age = diseases_age.drop('AGE', axis=1)


# In[ ]:





# In[114]:


plt.figure(figsize=(12,4))
sns.heatmap(diseases_age.groupby('Age_Group').mean(), cmap='viridis', fmt=".2f", linewidths=.5, color='green')
plt.title('Risk Factors Distribution Across Age Groups')
plt.xlabel('Risk Factor')
plt.ylabel('Age Group')


# In[ ]:


# Exploring the Variation in Admission Numbers Based on Rural or Urban Locations


# In[116]:


sns.countplot(x='RURAL', hue='arrival_year', data=df, color='green')
plt.title('Number of admissions of rural and urban by Year')
plt.xlabel('Rural and Urban ')
plt.ylabel('Number of admissions')
plt.legend(title='Year', loc='upper right')


# In[ ]:





# In[57]:


summary_stats = df[['AGE','DURATION OF STAY','EF']].describe()
print(summary_stats)


# In[58]:


#Produce summary statistics
# Exclude 'column1' and 'column2' from the DataFrame
df = df.drop(['month year',], axis=1)
summary=df.describe()
print(summary)


# In[59]:


# Visualizations for correlation between chronic kidney disease and diabetes
#df['UREA'] = pd.to_numeric(df['UREA'],errors='coerce')
#sns.scatterplot(x='UREA',y='CREATININE',data=df)


# In[60]:


plt.figure(figsize=(12, 6))
sns.scatterplot(x='duration of intensive unit stay', y='EF', data=df, color='green')
plt.xlabel('duration of intensive unit stay')
plt.ylabel('EF')
plt.title('Effect of low Ejection fraction on admission')
plt.show()


# In[61]:


# Scatter plot: Age vs HB
df['HB'] = pd.to_numeric(df['HB'], errors='coerce')
plt.figure(figsize=(10, 6))
plt.scatter(df['AGE'], df['HB'], alpha=0.6, color='green')
plt.title('Relationship between Age and HB')
plt.xlabel('Age')
plt.ylabel('HB')
plt.grid(True)
plt.show()


# In[1]:


#Visualization


# In[62]:


# Plot a histogram
age_min = 20
age_max = 100

# Create a subset of data within age range
subset_df = df[(df['AGE'] >= age_min) & (df['AGE'] <= age_max)]
plt.hist(subset_df['AGE'], bins=10, edgecolor='black',color='green')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[63]:


df['HB'] = pd.to_numeric(df['HB'], errors='coerce')
plt.figure(figsize=(12, 6))
sns.scatterplot(x='duration of intensive unit stay', y='HB', data=df, color='green')
plt.xlabel('duration of intensive unit stay')
plt.ylabel('HB')
plt.title('Effect of Low HB on admission')
plt.show()


# In[64]:


df['CREATININE'] = pd.to_numeric(df['CREATININE'], errors='coerce')
plt.figure(figsize=(12, 6))
sns.scatterplot(x='duration of intensive unit stay', y='CREATININE', data=df, color='green')
plt.xlabel('duration of intensive unit stay')
plt.ylabel('CREATININE')
plt.title('Effect of creatinine levels on duration of admission')
plt.show()


# In[65]:


df['PLATELETS'] = pd.to_numeric(df['PLATELETS'], errors='coerce')
plt.figure(figsize=(12, 6))
sns.scatterplot(x='duration of intensive unit stay', y='PLATELETS', data=df, color='green')
plt.xlabel('duration of intensive unit stay')
plt.ylabel('PLATELETS')
plt.title('Effect of platelets levels on duration of admission')
plt.show()


# In[66]:


df['GLUCOSE'] = pd.to_numeric(df['GLUCOSE'], errors='coerce')
plt.figure(figsize=(12, 6))
sns.scatterplot(x='duration of intensive unit stay', y='GLUCOSE', data=df, color='green')
plt.xlabel('duration of intensive unit stay')
plt.ylabel('GLUCOSE')
plt.title('Effect of glucose levels on duration of admission')
plt.show()


# In[67]:


# Splitting the data into subsets based on Age
cond_groups = df.groupby('AGE')


# In[68]:


plt.figure(figsize=(12, 6))
plt.bar(df['OUTCOME'], df['duration of intensive unit stay'], color='green')
plt.title('RELATIONSHIP BETWEEN DURATION OF INTENSIVE UNIT STAY AND POSSIBILITY OF DISCHARGE')
plt.ylabel('duration of intensive unit stay')
plt.show()


# In[69]:


df['duration of intensive unit stay'] = pd.to_numeric(df['duration of intensive unit stay'],errors='coerce')
sns.scatterplot(x='duration of intensive unit stay',y='GLUCOSE',data=df, color='green')


# In[118]:


df['DURATION OF STAY'] = pd.to_numeric(df['DURATION OF STAY'],errors='coerce')
plt.figure(figsize=(12, 6))
plt.bar(df['OUTCOME'], df['DURATION OF STAY'], color='green')
plt.title('RELATIONSHIP BETWEEN DURATION OF STAY AND POSSIBILITY OF DISCHARGE')
plt.ylabel('DURATION OF STAY')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




