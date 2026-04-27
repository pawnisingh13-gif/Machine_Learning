# Movie Recommendation (Content Based)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dataset
data = {
    'movie':['Avengers','Batman','Superman','Ironman','Spiderman'],
    'genre':['action hero','dark action','action hero','tech hero','action teen']
}

df = pd.DataFrame(data)

# Convert text to vectors
cv = CountVectorizer()
matrix = cv.fit_transform(df['genre'])

# Similarity
similarity = cosine_similarity(matrix)

# Function
def recommend(movie_name):
    idx = df[df['movie']==movie_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x:x[1], reverse=True)

    for i in scores[1:3]:
        print(df.iloc[i[0]]['movie'])

# Test
recommend('Avengers')
