import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from sklearn.metrics import classification_report
import nltk
import string
import re

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# ---------------------- Preprocessing ----------------------

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# ---------------------- Load Dataset ----------------------

df = pd.read_csv("cyberbullying_tweets.csv")  # CSV should have 2 columns: text,label
df.columns = ['text', 'label']

# Clean and filter
df['cleaned_text'] = df['text'].apply(preprocess)
df = df[df['cleaned_text'].str.strip() != '']

# Binary classification
df['binary_label'] = df['label'].apply(lambda x: 'not_cyberbullying' if x == 'not_cyberbullying' else 'cyberbullying')

# Balance dataset
cyber_df = df[df['binary_label'] == 'cyberbullying']
not_cyber_df = df[df['binary_label'] == 'not_cyberbullying']
min_len = min(len(cyber_df), len(not_cyber_df))
df_balanced = pd.concat([cyber_df.sample(min_len, random_state=42), not_cyber_df.sample(min_len, random_state=42)])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------- Feature Extraction ----------------------

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_balanced['cleaned_text'])
y = df_balanced['binary_label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

#Calculate Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Evaluate
print("\nModel Evaluation:\n")
print(classification_report(y_test, model.predict(X_test)))

# ---------------------- GUI Function ----------------------

def detect_cyberbullying():
    user_input = entry.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return
    cleaned = preprocess(user_input)
    if not cleaned:
        messagebox.showwarning("Input Error", "Input too short or meaningless.")
        return
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)[0]
    
    result_label.config(
        text=f"Prediction: {prediction.capitalize()}",
        fg="green" if prediction == "not_cyberbullying" else "red"
    )

# ---------------------- GUI Interface ----------------------

root = tk.Tk()
root.title("Cyberbullying Detection System")
root.geometry("520x320")
root.configure(bg="#f0f0f0")

title_label = tk.Label(root, text="Cyberbullying Detector", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
title_label.pack(pady=10)

entry = tk.Text(root, height=6, width=55, font=("Helvetica", 12))
entry.pack(pady=10)

check_button = tk.Button(root, text="Check", font=("Helvetica", 12), command=detect_cyberbullying)
check_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#f0f0f0")
result_label.pack(pady=10)

root.mainloop()
