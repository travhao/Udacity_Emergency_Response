# Disaster Response Pipeline Project

#Part of a Udacity Project in the Data Science Nanodegree

######## Introduction #########
This project takes a series of messages related to natural disasters and tries to determine
which ones are related to specific areas of emergency (death, water, food, floods, etc.) in order to 
provide more timely response in emergencies.  The files in this folder consist of the following:

The data gets processed using the process_data.py program found in the data folder.

The cleaned up data then gets scored using a Logistic Regression Model on 35 different categories
including whether the text relates to things such as death, water, food, floods, etc.).  

Finally, the saved prediction model gets used in an app that shows a couple of graphs related to the 
data.  On the application you can also enter your own sentence/comment and see how the application would
assign it out based upon the model.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


