
"""
Day 38: Content-Based Filtering Recommendation System
365 Days AIML Engineer Challenge

Recommend movies based on their genres and descriptions.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Create movie dataset
movies = {
    "Movie_ID": [1, 2, 3, 4, 5, 6, 7, 8],
    "Title": [
        "The Dark Knight",
        "Batman Begins",
        "Iron Man",
        "Avengers",
        "Interstellar",
        "Inception",
        "The Matrix",
        "Spider Man"
    ],
    "Description": [
        "Action crime superhero batman joker",
        "Action crime batman superhero origin",
        "Action science fiction superhero technology",
        "Action adventure superhero team",
        "Science fiction space adventure physics",
        "Science fiction thriller dreams mind",
        "Science fiction action artificial intelligence",
        "Action adventure superhero spider"
    ]
}

df = pd.DataFrame(movies)

print("MOVIE DATASET")
print(df[["Title", "Description"]])

# Step 2: Convert descriptions into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(df["Description"])

print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)

# Step 3: Calculate cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=df["Title"],
    columns=df["Title"]
)

print("\nCOSINE SIMILARITY MATRIX")
print(similarity_df.round(2))


# Step 4: Recommendation function
def recommend_movies(movie_title, top_n=3):

    if movie_title not in df["Title"].values:
        print("Movie not found!")
        return

    movie_index = df.index[
        df["Title"] == movie_title
    ][0]

    scores = list(enumerate(similarity_matrix[movie_index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Exclude the selected movie
    scores = [
        item for item in scores
        if item[0] != movie_index
    ]

    print(f"\nRecommendations for: {movie_title}")

    for index, score in scores[:top_n]:
        print(
            f"{df.iloc[index]['Title']} "
            f"-> Similarity: {score:.2f}"
        )


# Step 5: Generate recommendations
recommend_movies("The Dark Knight", 3)

recommend_movies("Interstellar", 3)

recommend_movies("Iron Man", 3)

print("\nDay 38 completed successfully!")
