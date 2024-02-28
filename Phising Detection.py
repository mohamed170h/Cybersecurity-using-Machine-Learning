import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training dataset
train_data = pd.DataFrame({
    'text': [
        "Dear customer, your account has been compromised. Please click the link to verify your information.",
        "Congratulations! You've won a free vacation. Click here to claim your prize.",
        "Urgent: Your bank account needs attention. Log in immediately to prevent suspension.",
        "Hi, this is a reminder about our upcoming meeting. Please find the attached agenda.",
        "Your Amazon order has been shipped. Click here to track your package.",
        "Dear user, we've noticed suspicious activity on your account. Update your password now.",
        "Invitation: Join us for a webinar on the latest industry trends.",
        "Your PayPal account has been locked. Click the link to unlock it.",
        "Hello, I hope this email finds you well. Attached is the report you requested.",
        "Exclusive offer: Buy one, get one free! Limited time only."
    ],
    'label': [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
})

# Test dataset
test_data = pd.DataFrame({
    'text': [
        "Important: Verify your email account now to prevent suspension.",
        "You've been selected for a job interview. Click here for details.",
        "Your credit card statement is available for download.",
        "Reminder: Tomorrow's team meeting at 10 AM. Don't forget!",
        "Claim your exclusive discount on the latest tech gadgets.",
    ],
    'label': [1, 0, 0, 0, 1]
})

# Convert the email text into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(train_data['text'])
X_test_vectorized = vectorizer.transform(test_data['text'])

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, train_data['label'])

# Make predictions on the test set
predictions = classifier.predict(X_test_vectorized)

# Print whether an email is phishing or not
for email, label in zip(test_data['text'], predictions):
    result = "Phishing" if label == 1 else "Legitimate"
    print(f"Email: '{email}' is {result}")
