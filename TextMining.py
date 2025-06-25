import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Dataset utama
data = {
    'tweet': [
        "The new album just dropped",
        "I love attending concerts",
        "He voted for the new senator",
        "Election day is coming soon",
        "She sings beautifully on stage",
        "Parliament will pass the bill",
        "Great guitar solo!",
        "Debate between presidential candidates",
        "Listening to classical music",
        "Politics today is exhausting",
        "Went to a live concert yesterday",
        "The president is planning a press release",
        "She played the violin at the orchestra",
        "They introduced a controversial new policy",
        "Rock bands are amazing",
        "The government changed the law"
    ],
    'label': [
        'musik', 'musik', 'politik', 'politik', 'musik',
        'politik', 'musik', 'politik', 'musik', 'politik',
        'musik', 'politik', 'musik', 'politik', 'musik', 'politik'
    ]
}

df = pd.DataFrame(data)

# 2. Tambahkan data tambahan spesifik untuk concert = musik
tambahan = [
    ("Can't wait for the concert tonight", 'musik'),
    ("The concert last night was amazing", 'musik'),
    ("Live concerts are the best experience", 'musik'),
    ("I am going to a music concert tonight", 'musik'),
    ("This rock concert is gonna be fire!", 'musik'),
    ("Concerts are such a joyful music event", 'musik')
]

for text, label in tambahan:
    df = pd.concat([df, pd.DataFrame({'tweet': [text], 'label': [label]})], ignore_index=True)

# 3. TF-IDF Vectorizer dengan ngram 1-2 kata
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df['tweet'])
y = df['label']

# 4. Tampilkan Vocabulary & cek kata
vocab = vectorizer.get_feature_names_out()
print("\n=== Vocabulary ===")
print(vocab)

# Cek kata penting
for kata in ['concert', 'concert tonight', 'new policy', 'president']:
    print(f'"{kata}" in vocab? {"Ya" if kata in vocab else "Tidak"}')

# 5. TF-IDF Table (opsional, biar rapi)
tfidf_df = pd.DataFrame(X.toarray(), columns=vocab)
tfidf_df['label'] = y
print("\n=== TF-IDF Table (5 contoh) ===")
print(tfidf_df.head())

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 7. Model SVM
model = LinearSVC()
model.fit(X_train, y_train)

# 8. Evaluasi
print("\n=== Evaluasi Model (SVM) ===")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 9. Prediksi tweet baru
new_tweets = [
    "I can't wait for the concert tonight",
    "New policy announced by the president"
]
X_new = vectorizer.transform(new_tweets)
predictions = model.predict(X_new)

print("\n=== Prediksi Kalimat Baru ===")
for tweet, label in zip(new_tweets, predictions):
    print(f'"{tweet}" => Prediksi: {label}')
