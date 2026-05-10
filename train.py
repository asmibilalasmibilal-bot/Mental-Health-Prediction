import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# load dataset
df = pd.read_csv(r"C:\Users\ACER\Downloads\mental_health_combined_test.csv")

# keep required columns
df = df[['text', 'status']].dropna()

X = df['text']
y = df['status']

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# vectorizer
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)

# model
model = LinearSVC()

# train
model.fit(X_train_vec, y_train)

# save
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ NEW MODEL CREATED")