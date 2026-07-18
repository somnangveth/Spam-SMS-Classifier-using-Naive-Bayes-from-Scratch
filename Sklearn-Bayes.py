""" 
=========================================================================
Using Sklearn Library Naive Bayes to classify the spam / ham
=========================================================================
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('dataset/spam.csv', header=None, encoding='latin-1')
df = df.iloc[:, :2]
df.columns = ['label', 'message']
df = df.drop(index=0).reset_index(drop=True)
df['message'] = df['message'].str.lower()

#Split datasets into train / test
train_df, test_df = train_test_split(df, random_state=42, test_size=0.2, stratify=df['label'])


#Vectorize text into word-count features
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_df['message'])
X_test = vectorizer.transform(test_df['message'])

y_train = train_df['label']
y_test = test_df['label']

#Train the model
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

#Evaluate the test set
predictions = model.predict(X_test)
print(f"Accuracy score: {accuracy_score(y_test, predictions)*100}%")
print("Classification Report: \n")
print(classification_report(y_test, predictions))

new_test_email = "urgent free entry to win a prize call now to claim your cash reward"

#Vectorize
new_email_vector = vectorizer.transform([new_test_email])

#predict
prediction = model.predict(new_email_vector)
print("Predicted class:", prediction[0])

#Predict prob for each class
probabilities = model.predict_proba(new_email_vector)
print("Class order: ", model.classes_)
print("Probabilities: ", probabilities)
