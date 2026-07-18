"""
============================================================
Creating SPAM SMS Classifier using Naive Bayes
============================================================
"""
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset/spam.csv', header=None, encoding='latin-1')

#Keep only the first two cols
df = df.iloc[:, :2]

#Label the columns
df.columns = ['label', 'message']

# Drop the first row since the value is not necessary for our model
df = df.drop(index=0).reset_index(drop=True)

#Convert each message into lowercase
df['message'] = df['message'].str.lower()

train_df, test_df, train_labels, test_labels = train_test_split(df,df['label'], random_state=42, stratify=df['label'], test_size=0.2)

# Train the train_df

#make a vocabulary of unique words that occur in spam email
spam_messages = train_df.loc[train_df['label'] == 'spam', 'message']

#make a vocabulary of unique words that occur in ham email
ham_messages = train_df.loc[train_df['label'] == 'ham', 'message']

#Assign arrays for spam words
vocab_spam_words = []

for message in spam_messages:
    word_list = message.split()
    for word in word_list:
        vocab_spam_words.append(word)

#dedup the words in vocab_spam_words
vocab_unique_words_spam = list(dict.fromkeys(vocab_spam_words))

#Calculate the spamicity of each word
dict_spamicity = {}
for word in vocab_unique_words_spam:
    emails_with_word = 0
    for sentence in spam_messages:
        if word in sentence:
            emails_with_word += 1

    # print(f"Number of spam emails with with word {word}: {emails_with_word}")
    total_spam = len(spam_messages)
    spamicity = (emails_with_word + 1) / (total_spam + 2)
    # print(f"Spamicity of the word '{word}' : {spamicity}")
    dict_spamicity[word.lower()] = spamicity


#Assign arrays for ham words
vocab_ham_words = []

for message in ham_messages:
    word_list = message.split()
    for word in word_list:
        vocab_ham_words.append(word)

vocab_unique_words_ham = list(dict.fromkeys(vocab_ham_words))

#Calculate the Hamicity of each word
dict_hamicity = {}

for word in ham_messages:
    emails_with_word = 0
    for sentence in ham_messages:
        if word in sentence:
            emails_with_word += 1

    # print(f"Number of emails with word {word} : {emails_with_word}")
    total_ham = len(ham_messages)
    hamicity = (emails_with_word + 1)/(total_ham + 2)
    # print(f"Hamicity of the word '{word}' : {hamicity}")
    dict_hamicity[word.lower()] = hamicity


#Compute the Probability of Spam P(S)
prob_spam = len(spam_messages) / (len(spam_messages) + len(ham_messages))
# print(prob_spam)

#Compute the Probability of Ham P(-S)
prob_ham = len(ham_messages) / (len(ham_messages) + len(spam_messages))
# print(prob_ham)

# Test the un-labelled email
tests = []

for message in test_df['message']:
    tests.append(message)

#Declare array for distinct words
distinct_words_in_sentence_test = []

#Split email with distinct words
for sentence in tests:
    sentence_as_lists = sentence.split()
    senten = []
    for word in sentence_as_lists:
        senten.append(word)
    distinct_words_in_sentence_test.append(senten)

test_spam_tokenized = [distinct_words_in_sentence_test[0], distinct_words_in_sentence_test[1]]
test_ham_tokenized = [distinct_words_in_sentence_test[0], distinct_words_in_sentence_test[1]]

# Check if the spam word exist in the labelled/training dataset
reduced_sentences_spam_test = []
for sentence in test_spam_tokenized:
    words_ = []
    for word in sentence:
        if word in vocab_unique_words_spam:
            print(f"'{word}': ok")
            words_.append(word)
        elif word in vocab_unique_words_ham:
            print(f"'{word}': ok")
            words_.append(word)
        else:
            print(f"'{word}' not present in the labelled training data")
    reduced_sentences_spam_test.append(words_)
print(reduced_sentences_spam_test)

#Check if the ham word exist in the labelled/training dataset
reduced_sentences_ham_test = []

for sentence in test_ham_tokenized:
    words_ = []
    for word in sentence:
        if word in vocab_unique_words_spam:
            print(f"'{word}': ok")
            words_.append(word)
        elif word in vocab_unique_words_ham:
            print(f"'{word}': ok")
            words_.append(word)
        else:
            print(f"'{word} not present in the labelled dataset")
    reduced_sentences_ham_test.append(words_)
print(reduced_sentences_spam_test)

#Finding the non-key meaning the words that appear frequently in email
#but is not important in classify between ham and spam
stop_words = set(stopwords.words('english'))

vocab_unique_words = set(vocab_unique_words_ham) | set(vocab_unique_words_spam)

non_key = stop_words.intersection(vocab_unique_words)

#Remove the non-key words for spam
test_spam_stemmed = []
for email in reduced_sentences_spam_test:
    email_stemmed = []
    for word in email:
        if word in non_key:
            print(f"Remove: {word}")
        else :
            email_stemmed.append(word)
    test_spam_stemmed.append(email_stemmed)

print(test_spam_stemmed)

#Remove the non-key words in ham
test_ham_stemmed = []
for email in reduced_sentences_ham_test:
    email_stemmed = []
    for word in email:
        if word in non_key:
            print(f"Remove: {word}")
        else:
            email_stemmed.append(word)
    test_ham_stemmed.append(email_stemmed)

print(test_ham_stemmed)

#Classify spam test emails:
def mult(list_):
    total_prob = 1
    for i in list_:
        total_prob = total_prob * i
    return total_prob

def Bayes(email):
    probs = []
    for word in email:
        Pr_S = prob_spam
        print(f'prob of spam in general ', Pr_S)
        try:
            pr_WS = dict_spamicity[word]
            print(f'prob "{word}"  is a spam word : {pr_WS}')
        except KeyError:
            pr_WS = (1/(total_spam + 2))
            print(f"prob '{word}' is a spam word: {pr_WS}")

        Pr_H = prob_ham
        print(f'prob of ham in general ', Pr_H)
        try:
            pr_WH = dict_hamicity[word]
            print(f'prob "{word}"  is a ham word : {pr_WH}')
        except KeyError:
            pr_WH = (1/(total_ham + 2))
            print(f"WH for {word} is {pr_WH}")
            print(f"prob '{word}' is a ham word: {pr_WH}")

        prob_word_is_spam_BAYES = (pr_WS*Pr_S)/((pr_WS*Pr_S)+(pr_WH*Pr_H))
        print('')
        print(f"Using Bayes, prob the word '{word}' is spam: {prob_word_is_spam_BAYES}")
        print("==================================================")
        probs.append(prob_word_is_spam_BAYES)
    print(f"All word probabilities for this sentence: {probs}")
    final_classification = mult(probs)
    if final_classification > 0.5:
        print(f"Email is SPAM: with spammy confidence of {final_classification*100}%")
    else:
        print(f"Email is HAM: with spammy confidence of {final_classification*100}%")
    return final_classification

# Use the filtered test dataset to test
# for email in test_spam_stemmed:
#   print("Testing email is SPAM or HAM")
#   all_words_probs = Bayes(email)
#   print(all_words_probs)
  
new_test_email = "urgent free entry to win a prize call now to claim your cash reward"

new_email_lower = new_test_email.lower()

new_email_tokens = new_email_lower.split()

filtered = []
for word in new_email_tokens:
    if word in vocab_unique_words_spam or word in vocab_unique_words_ham:
        filtered.append(word)
    else:
        print(f"'{word}' not present in the labelled training data")

filtered_stemmed = [w for w in filtered if w not in non_key]
Bayes(filtered_stemmed)
