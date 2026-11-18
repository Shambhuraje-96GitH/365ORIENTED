
"""
Day 36 - Collaborative Filtering Recommendation System
365-Day AIML Challenge

Algorithm: User-Based Collaborative Filtering
Technique: Cosine Similarity
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# 1. Create sample movie rating dataset
# --------------------------------------------------

ratings = {
    "Alice": {
        "Inception": 5,
        "Interstellar": 5,
        "Titanic": 1,
        "Avatar": 4,
        "Joker": 2
    },
    "Bob": {
        "Inception": 5,
        "Interstellar": 4,
        "Titanic": 1,
        "Avatar": 5,
        "Joker": 2
    },
    "Charlie": {
        "Inception": 1,
        "Interstellar": 2,
        "Titanic": 5,
        "Avatar": 2,
        "Joker": 5
    },
    "David": {
        "Inception": 4,
        "Interstellar": 5,
        "Titanic": 1,
        "Avatar": 4,
        "Joker": 1
    },
    "Emma": {
        "Inception": 2,
        "Interstellar": 1,
        "Titanic": 5,
        "Avatar": 3,
        "Joker": 5
    },
    "Frank": {
        "Inception": 5,
        "Interstellar": 4,
        "Titanic": 2,
        "Avatar": 5,
        "Joker": 1
    }
}

# Convert ratings into DataFrame
rating_matrix = pd.DataFrame(ratings).T

print("\nUSER-ITEM RATING MATRIX")
print(rating_matrix)

# --------------------------------------------------
# 2. Calculate User Similarity
# --------------------------------------------------

similarity_matrix = cosine_similarity(rating_matrix)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=rating_matrix.index,
    columns=rating_matrix.index
)

print("\nUSER SIMILARITY MATRIX")
print(similarity_df.round(2))

# --------------------------------------------------
# 3. Recommendation Function
# --------------------------------------------------

def recommend_movies(user, top_n=3):

    if user not in rating_matrix.index:
        print("User not found!")
        return

    # Find similar users
    similarities = similarity_df[user].drop(user)

    # Sort users by similarity
    similar_users = similarities.sort_values(
        ascending=False
    )

    # Movies not rated by target user
    unrated_movies = rating_matrix.loc[user][
        rating_matrix.loc[user] == 0
    ].index.tolist()

    # Calculate weighted scores
    recommendations = {}

    for movie in unrated_movies:

        numerator = 0
        denominator = 0

        for other_user, similarity in similar_users.items():

            user_rating = rating_matrix.loc[
                other_user, movie
            ]

            if user_rating > 0 and similarity > 0:

                numerator += similarity * user_rating
                denominator += similarity

        if denominator > 0:
            recommendations[movie] = (
                numerator / denominator
            )

    # Sort recommendations
    sorted_recommendations = sorted(
        recommendations.items(),
        key=lambda item: item[1],
        reverse=True
    )

    print(f"\nTOP {top_n} RECOMMENDATIONS FOR {user.upper()}")

    if not sorted_recommendations:
        print("No recommendations available.")
        return

    for movie, score in sorted_recommendations[:top_n]:
        print(f"{movie}: Predicted Rating = {score:.2f}")


# --------------------------------------------------
# 4. Generate Recommendations
# --------------------------------------------------

recommend_movies("Alice", top_n=3)

# --------------------------------------------------
# 5. Find Most Similar Users
# --------------------------------------------------

print("\nMOST SIMILAR USERS TO ALICE")

print(
    similarity_df["Alice"]
    .drop("Alice")
    .sort_values(ascending=False)
    .round(2)
)

print("\nRecommendation System Executed Successfully!")
