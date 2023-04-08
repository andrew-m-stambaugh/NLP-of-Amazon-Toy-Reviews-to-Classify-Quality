# NLP-of-Amazon-Toy-Reviews-to-Classify-Quality

This is an ongoing project for one of my Machine Learning courses at Northwestern University. The goal is to take a raw dataset of almost 1 million Amazon toy reviews, clean the data, and use NLP fundamentals to accurately classify whether a product is "awesome" or not. As of now, I have preprocessed the data and created the DTM across all of the review summaries. However, the DTM is too large for the memory to be allocated. Thus, the immediate next steps for this project are to code in a word frequency threshold for the matrix to trim down its dimensionality. Then, I will do the same for a DTM across the full reviews and begin building the predictive model from there. The type of model I will train is TBD, but there will be one ML model and one Keras deep learning model. Potential candidates for the ML model are random forest classification, Gaussian Naive Bayes, or gradient boosting.

Files in this project:

Project.py: This python file contains all the code used for the project, including data preprocessing, feature extraction, language processing, and soon to be model training.
