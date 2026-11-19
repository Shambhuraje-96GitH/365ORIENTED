
"""
Day 37: Item-Based Collaborative Filtering
365 Days AIML Engineer Challenge

Objective:
Recommend movies based on similarity between items.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-movie ratings
ratings = {
    "Movie A": [5, 4, 0, 0, 3],
    "Movie B": [4, 5, 0, 0, 2],
    "Movie C": [0, 0, 5, 4, 0],
    "Movie D": [0, 0, 4, 5, 0],
    "Movie E": [3, 2, 0, 0, 4]
}

users = ["User 1", "User 2", "User 3", "User 4", "User 5"]

df = pd.DataFrame(ratings, index=users)

print("USER-MOVIE RATING MATRIX")
print(df)

# Calculate item-item cosine similarity
item_similarity = cosine_similarity(df.T)

similarity_df = pd.DataFrame(
    item_similarity,
    index=df.columns,
    columns=df.columns
)

print("\nITEM SIMILARITY MATRIX")
print(similarity_df.round(2))


def recommend_movies(movie_name, top_n=3):

    if movie_name not in similarity_df.columns:
        print("Movie not found!")
        return

    similarities = similarity_df[movie_name]

    recommendations = similarities.drop(movie_name)

    recommendations = recommendations.sort_values(
        ascending=False
    ).head(top_n)

    print(f"\nRecommendations based on {movie_name}:")

    for movie, score in recommendations.items():
        print(f"{movie}: Similarity = {score:.2f}")


# Generate recommendations
recommend_movies("Movie A")

recommend_movies("Movie C")

print("\nDay 37 completed successfully!")
