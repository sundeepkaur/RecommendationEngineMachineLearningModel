#!/usr/bin/env python
# coding: utf-8

# # Preparing Environment

# In[1]:


from pymongo import MongoClient
import pandas as pd
from pandas import DataFrame
from math import radians
import math
import numpy as np
from sklearn.model_selection import train_test_split
#from haversine import haversine
#from haversine import haversine_vector,Unit
import warnings; warnings.simplefilter('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from datetime import datetime
from datetime import date
import calendar
distanceRange = 100
numberOfRows = 6
skillRateRange = 3


# # Data Collection :

# # Access and Import Files from the MongoDB Database

# In[2]:


from pprint import pprint
# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient("mongodb://78.47.77.101:27017/root")
db=client.admin
# Issue the serverStatus command and print the results
serverStatusResult=db.command("serverStatus")
print(serverStatusResult)


# In[3]:


#scores=5


# In[4]:


#Distance_in_km=100


# In[5]:


job_db=client.eande
print(job_db)


# In[6]:


myCursor = job_db.skills.find( {} )
df_skills = DataFrame(list(job_db.skills.find( {} )))
df_JobProposal=DataFrame(list(job_db.JobProposal.find({})))
df_PostJob=DataFrame(list(job_db.PostJob.find({})))
df_JobSkills=DataFrame(list(job_db.JobSkills.find({})))
df_UserLang=DataFrame(list(job_db.UserLang.find({})))
df_Company=DataFrame(list(job_db.Company.find({})))
df_users=DataFrame(list(job_db.users.find({})))
df_UserSkills=DataFrame(list(job_db.UserSkills.find({})))
df_UserCertification = DataFrame(list(job_db.UserCertification.find( {} )))
df_langs = DataFrame(list(job_db.langs.find( {} )))
df_roles = DataFrame(list(job_db.roles.find( {} )))


# In[7]:


#COUNTING THE USER JOB EXPERIENCE
df_UserJobExperience=DataFrame(list(job_db.UserJobExperience.find({})))
df_UserJobExperience['start_date']=pd.to_datetime(df_UserJobExperience['start_date'],format ='%d-%b-%Y %H:%M:%S')
df_UserJobExperience['end_date']=pd.to_datetime(df_UserJobExperience['end_date'],format ='%d-%b-%Y %H:%M:%S')
df_UserJobExperience['Experience in Months'] =(df_UserJobExperience.end_date.dt.year - df_UserJobExperience.start_date.dt.year) * 12 +(df_UserJobExperience.end_date.dt.month - df_UserJobExperience.start_date.dt.month)
df_UserJobExperience.head()


# # Merging dataframes

# In[8]:


df1=pd.merge(df_PostJob,df_JobSkills,left_on='_id',right_on='job_id',how='left')
df1.head().transpose()


# In[9]:


df2_jobs=pd.merge(df1,df_Company,on='company_id',how='left')
df2_jobs.head().transpose()


# In[10]:


df2_jobs=df2_jobs[['_id_x','rate_x','status_x','total_cost','job_title','company_id','company_name','lat','postal','long','skill_id','job_description']]

df2_jobs.columns=['Job_Id','Skill_rate','Job_Status','Total_salary','Job_Title','Company_id','Company_name','Company_lat','Postal','Company_long','Skill_Id','Job_Description']
df2_jobs.head()


# In[11]:


df2_jobs["Skill_Id"].fillna("Null", inplace = True)
len(df2_jobs)
df2_jobs.isna().sum()


# In[12]:


df3=pd.merge(df_users,df_UserSkills,left_on='_id',right_on='user_id',how='left')
df3.head()


# In[13]:


len(df3)
df3.isna().sum()


# In[14]:


df4=pd.merge(df_UserLang,df_UserJobExperience,on='user_id',how='left')
df4.head()


# In[15]:


df5_users=pd.merge(df3,df4,on='user_id',how='left')
#df5_user=df5_users[['address','mobile_no','dob','first_name','last_name','lat','long','skill_id','user_id','lang_id','Years of Experience','title']]
df6_users=pd.merge(df5_users,df_JobProposal,on='user_id',how='left')
df6_users.head()


# In[16]:


#Calculating age from date of birth column from df6_users
def calculate_age(born):
    born = datetime.strptime(born, "%Y-%m-%d").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
df6_users['age'] = df6_users['dob'].apply(calculate_age)


# In[17]:


df6_users=df6_users[['lat','rate_x','long','_id_x_x','Experience in Months','skill_id','job_id','age','lang_id','description','propoal_description','title']]
df6_users.columns=['User_lat','Skill_rate','User_long','User_Id','Job Experience in months','Skill_Id','Job_Id','Age','Lang_Id','Description','Proposal desc','Title']
df6_users.head()


# In[18]:


df6_users["Job Experience in months"].fillna(0, inplace = True)
df6_users["Skill_Id"].fillna("Null", inplace = True)
df6_users["Description"].fillna("Null", inplace = True)
df6_users["Title"].fillna("Null", inplace = True)
#df6_users["User_Id"].fillna("Null", inplace = True)
df6_users["Lang_Id"].fillna("Null", inplace = True)
df6_users["Job_Id"].fillna("Null", inplace = True)
df6_users["Proposal desc"].fillna("Null", inplace = True)
len(df6_users)


# # Final dataframe

# In[19]:


#i.	We have further mergerd df2_jobs and df6_users on job_id in order to use it as a dataframe to generate the final output  for a model.
df_final=pd.merge(df6_users,df2_jobs,on='Job_Id',how='left')
df_final.head().transpose()


# In[20]:


df_final["Skill_rate_y"].fillna(0, inplace = True)
df_final["Job_Status"].fillna("Null", inplace = True)
df_final["Total_salary"].fillna(0, inplace = True)
df_final["Job_Title"].fillna("Null", inplace = True)
#df6_users["User_Id"].fillna("Null", inplace = True)
df_final["Company_id"].fillna("Null", inplace = True)
df_final["Company_name"].fillna("Null", inplace = True)
#df_final["Company_lat"].fillna("Null", inplace = True)
df_final["Postal"].fillna("Null", inplace = True)
#df_final["Company_long"].fillna("Null", inplace = True)
df_final["Skill_Id_y"].fillna("Null", inplace = True)
df_final["Job_Description"].fillna("Null", inplace = True)
df_final.isna().sum()


# In[21]:


df_final["User_lat"]=pd.to_numeric(df_final["User_lat"])
df_final["User_long"]=pd.to_numeric(df_final["User_long"])
df_final["Company_lat"]=pd.to_numeric(df_final["Company_lat"])
df_final["Company_long"]=pd.to_numeric(df_final["Company_long"])


# # Calculating Distance

# In[22]:


from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

for index, row in df_final.iterrows():
    df_final.loc[index, 'Distance in km'] = haversine(row['Company_long'], row['Company_lat'], row['User_long'], row['User_lat'])


# In[23]:


#Replacing the Nan values if any from the column and then selecting final columns of the dataframe which we will be using in model
df_final['Distance in km'].fillna("Null", inplace = True)
df_final=df_final[['Skill_rate_x','User_Id','Job Experience in months','Skill_Id_x','Job_Id','Age','Lang_Id','Job_Description','Description','Title','Job_Title','Skill_rate_y','Proposal desc','Total_salary','Distance in km','Job_Status','Company_id','Company_name','Postal']]


# # Splitting data into test and train dataset

# In[24]:


#Splitting the users data in test and train
df_users_train, df_users_test=train_test_split(df6_users, test_size=0.2)


# In[25]:


#length of the users train dataset
len(df_users_train)


# In[26]:


#length of the users test dataset
len(df_users_test)


# In[27]:


#Split the jobs dataset in 80 and 20
df_jobs_train, df_jobs_test = train_test_split(df2_jobs, test_size=0.2)


# In[28]:


#length of the jobs train dataset
len(df_jobs_train)


# In[29]:


#length of the jobs test dataset
len(df_jobs_test)


# In[30]:


#taking input from title,description and proposal desc in df6_users['Description'] and filling up all the empty values

df_users_train['Job Experience in months']=df_users_train['Job Experience in months'].fillna("").astype('str')
df_users_train['Desc'] =df_users_train['Description']+" "+df_users_train['Title']+" "+df_users_train['Proposal desc']+df_users_train['Job Experience in months']
df_users_train['Desc'] = df_users_train['Desc'].fillna('')


# In[31]:


#In order to judge machin's performance qualitatively, using TfidfVectorizer function from scikit-learn,
# which transforms text to feature vectors that can be used as input to estimator ,removing the stop words
# and computing TF-IDF matrix required for calculating cosine similarity and dispalying the shape of our matrix.
tf = TfidfVectorizer(analyzer='word',ngram_range=(1,3),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(df_users_train['Desc'])
tfidf_matrix.shape
#tf.get_feature_names()


# In[32]:


print(tfidf_matrix)


# In[33]:


# computing cosine similarity matrix using linear_kernal of sklearn
cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)
cosine_sim[0]


# In[34]:


#Index will be created here for the user_id
df= df_users_train.reset_index(drop=True)
id = df_users_train['User_Id']
indices = pd.Series(df.index,index=df_users_train['User_Id'])
indices.head()


# # Getting similar users

# In[35]:


#function to get most similar users
def get_recommendations_userwise(userid):
    idx = indices[userid]
    #print (idx)
    sim_scores = list(enumerate(cosine_sim[idx]))
    #print (sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores=sim_scores[1:numberOfRows]
    job_indices = [i[0] for i in sim_scores]
    return id.iloc[job_indices]


# In[36]:


get_recommendations_userwise('5e5067b8-ce9e-437e-b643-8e9ac999a68f')


# In[37]:


print('User_Job_Title: ')
df_users_train['Title'].loc[df_users_train['User_Id'] == 'dae4b5df-752d-4f53-93c1-3a203c5a15fc']


# # Recommending jobs to users

# In[38]:


#function to recommend jobs to user by matching skill_id and filtering the output with distance
def get_job_id(usrid_list):
    jobs_userwise = df_users_train['User_Id'].isin(usrid_list)
    df1 = pd.DataFrame(data = df_users_train[jobs_userwise], columns=['Skill_Id'])
    joblist = df1['Skill_Id'].tolist()
    Job_list = df2_jobs['Skill_Id'].isin(joblist) #[1083186, 516837, 507614, 754917, 686406, 1058896, 335132])
    df_temp = pd.DataFrame(data = df_final[Job_list],columns=['User_Id','Job_Id','Distance in km','Company_name','Postal','Total_salary','Job_Description','Title'])
    return df_temp.loc[(df_temp['Distance in km'] <=distanceRange)]


# In[39]:


get_job_id(get_recommendations_userwise('6d9daf03-6e84-4919-a7c5-53da5adbc4c2'))


# In[40]:


df_users_train.tail(100)


# # Recommending users to companies

# In[41]:


#taking input from title,description and Skill_rate in df2_jobs['Description'] and filling up all the empty values
df2_jobs['Skill_rate']=df2_jobs['Skill_rate'].fillna("").astype('str')
df2_jobs['Descr'] = df2_jobs['Job_Description']+" "+df2_jobs['Job_Title']+" "+df2_jobs['Skill_rate']
df2_jobs['Descr'] = df2_jobs['Job_Description'] .fillna('')


# In[42]:


#In order to judge machin's performance qualitatively, using TfidfVectorizer function from scikit-learn,
# which transforms text to feature vectors that can be used as input to estimator ,removing the stop words
# and computing TF-IDF matrix required for calculating cosine similarity and dispalying the shape of our matrix.
tf1 = TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=0, stop_words='english')
tfidf_matrix1 = tf1.fit_transform(df2_jobs['Descr'])
tfidf_matrix1.shape


# In[43]:


# computing cosine similarity matrix using linear_kernal of sklearn
cosine_simi = linear_kernel(tfidf_matrix1,tfidf_matrix1)
cosine_simi[0]


# In[44]:


#Index will be created here for the Job_Id
df2_jobs = df2_jobs.reset_index(drop=True)
jobid = df2_jobs['Job_Id']
indices1 = pd.Series(df2_jobs.index,index=df2_jobs['Job_Id'])
indices1.head(2)


# In[45]:


#function fro similar job_ids
def get_recommendations(job_id):
    jobidx = indices1[job_id]
    #print (idx)
    sim_scores1 = list(enumerate(cosine_simi[jobidx]))
    #print (sim_scores1)
    sim_scores1= sorted(sim_scores1, key=lambda x: x[1], reverse=True)
    sim_scores1=sim_scores1[1:numberOfRows]
    user_indices = [i[0] for i in sim_scores1]
    return jobid.iloc[user_indices]


# In[46]:


get_recommendations("9ba580f8-1e27-4a92-a457-2d4f4409bbda")


# In[47]:


print('User_Job_Description: ')
df2_jobs['Job_Description'].loc[df2_jobs['Job_Id'] == 'c76be6a4-0761-4216-a5fa-52446e28e62b']


# In[48]:


#function to recommend users for the job_id 
def get_user_id(jobid_list):
    user_jobwise = df_users_train['Job_Id'].isin(jobid_list)
    df_1 = pd.DataFrame(data = df_users_train[user_jobwise], columns=['Skill_Id'])
    userlist = df_1['Skill_Id'].tolist()
    User_list = df2_jobs['Skill_Id'].isin(userlist) #[1083186, 516837, 507614, 754917, 686406, 1058896, 335132])
    df_tmp = pd.DataFrame(data = df_final[User_list],columns=['Job_Id','User_Id','Skill_Id_x','Job_Title','Age','Job_Description','Job Experience in months'])
    return df_tmp#.loc[(df_tmp['Skill_rate_x']>=killRateRange)]


# In[49]:


get_user_id(get_recommendations("c76be6a4-0761-4216-a5fa-52446e28e62b"))


# In[ ]:




