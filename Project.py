import pandas as pd
import datetime as dt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix

#%% Import data

Reviews = pd.read_json("C:/Users/ams1258.ORAD/Downloads/devided_dataset_v2/devided_dataset_v2/Toys_and_Games/train/review_training.json")
Products = pd.read_json("C:/Users/ams1258.ORAD/Downloads/devided_dataset_v2/devided_dataset_v2/Toys_and_Games/train/product_training.json")

#%% Export the data to a csv to be viewed in Excel

Reviews.to_csv('C:/Users/ams1258.ORAD/Downloads/devided_dataset_v2/devided_dataset_v2/NewReviews.csv', index=False)
Products.to_csv('C:/Users/ams1258.ORAD/Downloads/devided_dataset_v2/devided_dataset_v2/NewProducts.csv', index=False)

#%% Cleaning up the formatting of some and extracting features

## Convert Verified column to a number
Reviews['verified']=Reviews['verified'].astype(int)

## Convert unixReviewTime to datetime object
Reviews['unixReviewTime']=Reviews['unixReviewTime'].apply(lambda x:
                                        dt.datetime.fromtimestamp(x))
    
## Extract day, month, and year features
Reviews['Day'] = Reviews['unixReviewTime'].apply(lambda x: x.day)
Reviews['Month'] = Reviews['unixReviewTime'].apply(lambda x: x.month)
Reviews['Year'] = Reviews['unixReviewTime'].apply(lambda x: x.year)
## Hour = Reviews['unixReviewTime'].apply(lambda x: x.hour)
## Hour.unique() # The only hours in the data are 18 and 19, so not useful

## unixReviewTime conversion to datetime removes the necessity for ReviewTime
Reviews=Reviews.drop(columns=['reviewTime', 'unixReviewTime'])

## Remove reviews with empty texts and summaries as they provide little info
Reviews = Reviews.dropna(subset=['reviewText', 'summary'], how = 'all').reset_index()

#%% Create the Document term matrix for all of the review summaries

# Create the doc based on the list of all summaries
doc = list(Reviews['summary'])
 
# Change None's to ' ' for vectorizer
for i in range(len(doc)):
    if not isinstance(doc[i],str):
        doc[i] = ""

# Import the set of stop words and discard words we want to keep.
stop_words = set(stopwords.words('english'))
stop_words.discard('not') # "not" is crucial for sentiment analysis
stop_words.discard('never') # same for never, potentially
lemmatizer = WordNetLemmatizer()

clean_doc = []
for summary in doc:
    # Remove stop words and convert to lower case
    summary = ' '.join([word.lower() for word in summary.split() if word.lower() not in stop_words])
    # Lemmatize words
    summary = ' '.join([lemmatizer.lemmatize(word) for word in summary.split()])
    clean_doc.append(summary)
    
# Create the Vectorizer
vectorizer = CountVectorizer()
vectorizer.fit(clean_doc)

# Output the vectorized list of summaries as a vector
vector = vectorizer.transform(clean_doc)

# Create the DTM. The matrix is so large however that 
DTM = csr_matrix(vector)
# DTM = pd.DataFrame(data = vector.toarray(), columns = sorted(vectorizer.vocabulary_))
 