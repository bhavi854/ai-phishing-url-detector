
# ============================
# phishing_detector.py
# ============================

import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Vectorizer: character-level n-grams to capture suspicious patterns
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 5))

def load_data():
    data = {
        'url': [
            # phishing examples
            'http://secure-login.com/bank',
            'https://paypal-login-verification.net',
            'http://update-account-login.com',
            'https://secure-verification-paypal.net',
            'http://bankofamerica.secure-login-update.com',
            'http://verify-user-security-check.net',
            'http://account-update.secure-service.com',
            'https://login-facebook-security.net',
            'http://secure-check-paypal.com',
            'http://appleid.verify-login.com',

            # legitimate examples
            'http://google.com',
            'https://github.com/openai',
            'https://www.microsoft.com',
            'https://en.wikipedia.org',
            'https://www.youtube.com',
            'https://www.amazon.com',
            'https://www.reddit.com',
            'https://www.linkedin.com',
            'https://stackoverflow.com',
            'https://www.nytimes.com'
        ],
        'label': [1,1,1,1,1,1,1,1,1,1,  0,0,0,0,0,0,0,0,0,0]  # 1=phishing, 0=legit
    }
    return pd.DataFrame(data)

def train_model():
    df = load_data()
    X = vectorizer.fit_transform(df['url'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    print("Training accuracy:", model.score(X_test, y_test))
    joblib.dump(model, "phishing_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

if __name__ == "__main__":
    train_model()
