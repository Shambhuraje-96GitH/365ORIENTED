
# Day 36 - Collaborative Filtering Recommendation System

## Overview

This project implements a User-Based Collaborative
Filtering Recommendation System using Python.

The system recommends movies based on the preferences
and ratings of similar users.

## Objectives

- Understand recommendation systems.
- Implement User-Based Collaborative Filtering.
- Calculate user similarity using Cosine Similarity.
- Generate personalized movie recommendations.
- Identify similar users based on rating patterns.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

## Algorithm

User-Based Collaborative Filtering

### Working Process

1. Create a User-Item Rating Matrix.
2. Calculate similarity between users.
3. Identify users with similar preferences.
4. Find movies not rated by the target user.
5. Calculate weighted recommendation scores.
6. Display Top-N recommended movies.

## Formula

Cosine Similarity:

Similarity(A, B) =
(A . B) / (||A|| * ||B||)

Weighted Prediction:

Predicted Rating =
Sum(Similarity × Rating) / Sum(Similarity)

## Project Structure

D036/
│
├── Day 36 - Collaborative Filtering.py
└── README.md

## How to Run

Install dependencies:

pip install pandas numpy scikit-learn

Execute:

python "Day 36 - Collaborative Filtering.py"

## Expected Output

- User-Item Rating Matrix
- User Similarity Matrix
- Top Movie Recommendations
- Most Similar Users
- Successful execution message

## Real-World Applications

- Netflix movie recommendations
- Amazon product recommendations
- YouTube video recommendations
- Spotify music recommendations
- E-commerce personalization

## Limitations

- Cold Start Problem
- Sparse Rating Matrices
- Scalability Challenges
- New User Recommendation Difficulty

## Learning Outcome

Implemented a User-Based Collaborative Filtering
Recommendation System using Cosine Similarity and
personalized Top-N recommendations.

---

365-Day AIML Engineer Challenge
Day 36 - Completed
