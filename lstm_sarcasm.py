import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Preprocessing
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Building Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, GlobalAveragePooling1D
import tensorflow.keras as k

# Download Model
import pickle
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, classification_report

data = pd.read_json("Sarcasm_Headlines_Dataset.json", lines=True)
data.head()

data.drop(columns="article_link", axis=1, inplace=True)
data.head(5)

data["headline"][0]
data.info()

punc = list(string.punctuation)
nltk.download('stopwords')
stop_words = stopwords.words("english")
lemma = WordNetLemmatizer()
nltk.download('punkt')

nltk.download('wordnet')
def Process(data):
data.lower()

data = " ".join([lemma.lemmatize(word) for word in word_tokenize(data) if ((word not in punc) and (word not in stop_words))])

data = re.sub("[^a-z]", " ", data)

return data

data["headline"] = data["headline"].apply(Process)
data.head(5)

label = to_categorical(data["is_sarcastic"], 2)
label[:5]

X = data["headline"]
Y = label
print(Y[:2])

tokenize = Tokenizer(oov_token="<oov>")
tokenize.fit_on_texts(X)
word_idx = tokenize.word_index

data_seqence = tokenize.texts_to_sequences(X)
pad_seq = pad_sequences(data_seqence, padding="pre", truncating="pre")

print("The Padding Sequance Shape is --> ", pad_seq.shape)

input_length = max(len(seq) for seq in data_seqence)

vocabulary_size = len(word_idx) + 1

input_length, vocabulary_size

x_train, x_test, y_train, y_test = train_test_split(pad_seq, label, train_size=0.7)

model = k.models.Sequential([
Embedding(vocabulary_size, 50, input_length=input_length),
GlobalAveragePooling1D(),
Dense(48, activation="relu"),
Dense(2, activation="softmax")
])

model.compile(optimizer="adam", loss=k.losses.BinaryCrossentropy(), metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), verbose=2)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# Compute predicted labels using predicted probabilities
y_pred_proba = model.predict(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Ensure the shapes of y_pred and y_true are consistent
min_len = min(len(y_pred), len(y_test))
y_true = np.argmax(y_test[:min_len], axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred[:min_len])

# Calculate precision, recall, and F1-score
precision = precision_score(y_true, y_pred[:min_len])
recall = recall_score(y_true, y_pred[:min_len])
f1 = f1_score(y_true, y_pred[:min_len])
accuracy = accuracy_score(y_true, y_pred[:min_len]) # Calculate accuracy

# Print confusion matrix and metrics
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy: {:.2f}%".format(accuracy * 100)) # Print accuracy
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Sarcastic', 'Sarcastic'], yticklabels=['Not Sarcastic', 'Sarcastic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.title("Accuracy Vs Epochs")

plt.legend()
plt.grid()

# Evaluate the model
y_pred_proba = model.predict(x_test)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test[:, 1], y_pred_proba[:, 1])
roc_auc = roc_auc_score(y_test[:, 1], y_pred_proba[:, 1])

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

text = word_tokenize(input())

new_text = ""
for word in text:
if (word not in stop_words) and (word not in punc):
new_text += lemma.lemmatize(word)
new_text += " "

print(new_text)
test_sequace = tokenize.texts_to_sequences([new_text])
test_padding = pad_sequences(test_sequace, maxlen=31, padding="pre", truncating="pre")


# test_sequace
prediction = model.predict(test_padding)

print(prediction[0])
if np.argmax(prediction) == 1: print("This Message is --> is sarcastic ")
else: print("This Message is --> is not sarcastic ")
