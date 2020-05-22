# RecommendationEngineMachineLearningModel
 
###### On Youtube : project video link: https://www.youtube.com/watch?v=towQBkVPLY4&t=179s 
###### On Cloud : project video link: https://wordpress-404624-1273723.cloudwaysapps.com/index.php/movies/recommendation-model-using-content-based-filtering/
######  Project Recommendation engines Overview & Scope is as following:

This project intends to create recommendation engine that can Recommend best jobs to employees based on job’s skill rating, distance of potential employee from work location and either way around as well that is to recommend best employees to a company on the basis of employee’s skills, ratings, previous experience using machine learning concepts


######  The Scope includes the aspects of the recommendation engine as below: 

- Data Collection (i.e., data will be provided by Job-O explicitly). 
- Data Storage (No Sql) 
- Data Filtering: Best filtering algorithm to be chosen after evaluating 
- Machine learning - Content Based filtering model using cosine similarity algorithm
- Tools and Libraries 
	Jupyter Lab using Python 


###### Content Based recommendation model Theory
A content-based recommender model works with data that the user provides. Based on that data, a user profile is generated, which is then used to make suggestions to the user. 

For example as we can see in the diagram that the user read some article so similar type of those articles are recommended to user in content based filtering.

So in content-based filtering mechanisms concepts of Term Frequency (TF) and Inverse Document Frequency (IDF) are used.

TF is simply the frequency of a word in the document whereas IDF is the inverse of the document frequency. These concepts are used to determine the relative importance of a document or item

Below is shown the equation which is used to calculate the TFIDF score and point to be noted is that TF-IDF weighting ignores the effect of high frequency words in determining the importance of an item or document. 

After done with calculating the TF-IDF score next step is to determine which items are close to each other using vector space model where each item is stored as a vector and angle between the vectors is used to determine the similarity between the vectors or items and also using cosine calculation which is sum product of vectors.

###### Now let’s move to the practical model part:
First of all, Data preparation is done where loading and processing of data is there

So  we Successfully loaded and processed the Job-O data from their database using robo3T interface for accessing MongoDB database of Job-O
And after this for all the required tables for our project we Loaded the data from the respective tables from database to data frames in order to get respective information 

For example Loaded the data from the skills table from database to data frame in order to get information about the skills of users.
So likewise we have loaded data from the below tables from database to data frame:
•	PostJob: has information about the jobs posted by companies
•	JobSkills:  has information about the skills required for job
•	UserJobExperience: in this table out of columns start date and end date calculated the users experience in Months
•	UserSkills: It has information about the skills of user
•	Company: It has information about the company like its name, latitude and longitude which will be required 
•	Users:  It has all the information about the user.

### After this performed joining of  the data frames: First for jobs

a.	So here we joined two data frames df_PostJob and df_JobSkills based on common column job_id so that we can get Job posted by employers and skills required for a job in one data frame i.e. named as df1 in our model.  

b.	And after that Merged df1 with df_company on company_id in order to get the all information of jobs posted ,skills and company in one data frame i.e. df2_jobs in model, which is used as final data frame for jobs information in the model, and then done filtering on the columns and kept required columns in our final jobs data frame as displayed

#### Then for Users

c.	Merged of df_users with df_Userskills on user_id in order to get the data related to user skills and users in one data frame i.e. df3 

d.	Then Merged of df_UserLang and df_UserJobExperience on user_id giving data frame df4.

e.	Then Merged df4 and df3 giving df5_users data frame which is further merged with df_JobProposal on user_id giving data frame df6_users which in total have all the information about users.

f.	Calculated age from date of birth column from df6_users

g.	Then on final users data frame done filtering on the columns and kept required columns in our final Users data frame as displayed

h.	After done creating final data frames, we checked for Nan values and than have replaced all the ‘Nan’ values from the data frame.

i.	We have even further merged final jobs and users data frames on job_id in df_final data frame in order to use it in the final output of a model.

j.	Have calculated the distance between users and job locations using haversine formula in which firstly we changed the latitude and longitude of user and company into numeric and then applied the formula.

After that Split the users and jobs data in testing and training in 20 and 80 ratios or proportions

Then created the input string to be fed to model from user title, description and job proposal desc in df6_users[descriptions] so that users can be matched based on this value.

Then In order to judge machine’s performance qualitatively, used TfidfVectorizer function from scikit-learn library, which transforms text to feature vectors that can be used as input to estimator,

Then removed the stop words and computed TF-IDF matrix required for calculating cosine similarity and displayed the shape of the matrix.

We computed cosine similarity matrix using linear_kernal of sklearn library.

After that created the Index for the user_id column which is used further in function to get most similar users.

After that We created a function “get_recommendations_userwise”  which recommend most similar users in which first it takes index of user_id and generates similar score of the indices from the cosine_sim matrix and we filtered the sim_scores up to 6 to give us first 5 matching results.

Now, lets call the get_recommendations_userwise function and pass the user_id, meaning we are saying generate job recommendations for this user_id which are job_ids as output

After this We have also created a function “get_job_id(usrid_list)” which will recommend jobs to user on basis of skill_id and filtering the output with distance saying give the results for this user_id and also make sure the distance is less than 100 kms and getting output as:


And likewise we have created model to generate recommendations of employees to  employers.
