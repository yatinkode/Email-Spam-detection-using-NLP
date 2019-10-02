# Email-Spam-detection-using- Natural Language Processing
Building a spam filter by predicting whether an email message is spam (junk email) or ham (good email).
Library used is NLTK in Python

Dataset : emails.csv

| __Column name__    | __Detail__                                                 |
|--------------------|------------------------------------------------------------|
| text               |  Contains the subject and body of email in string format   |
| spam               | If email is spam then 1 else 0                             |

## Steps Involved:

#### Step 1 : Understanding and Cleaning the data
#### Step 2 : Converting text to words
#### Step 3 : Stemming/Lemmatizing words
#### Step 4 : Vectorizing words and creating respective columns for the same
#### Step 5 : Separating data into train and test datasets
#### Step 6 : Applying machine learning algorithm (here I have choosen Random Forest) on train data
#### Step 7 : Choosing best hyperparameters
#### Step 8 : Evaluating the model on test data
