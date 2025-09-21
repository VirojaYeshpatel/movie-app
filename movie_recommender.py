import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load Data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Step 2: Merge datasets
movie_data = pd.merge(ratings, movies, on="movieId")

# Step 3: Create user-movie matrix
user_movie_matrix = movie_data.pivot_table(index="userId", columns="title", values="rating")

# Step 4: Replace NaN with 0
movie_matrix = user_movie_matrix.fillna(0)

# Step 5: Compute similarity
similarity_matrix = cosine_similarity(movie_matrix.T)
similarity_df = pd.DataFrame(similarity_matrix, index=movie_matrix.columns, columns=movie_matrix.columns)

# Step 6: Recommendation Function
def recommend_movies(movie_name, num_recommendations=5):
    # Case-insensitive search for matches
    matches = [m for m in similarity_df.columns if movie_name.lower() in m.lower()]
    
    if not matches:
        return f"No movie found with keyword '{movie_name}'. Try another."
    
    # If multiple matches, let user choose
    print("\nFound these matches:")
    for i, m in enumerate(matches, 1):
        print(f"{i}. {m}")
    
    choice = input("\nEnter the number of the movie you mean: ")
    try:
        choice = int(choice) - 1
        selected_movie = matches[choice]
    except:
        return "Invalid choice."
    
    print(f"\nShowing results for: {selected_movie}\n")
    
    # Get similarity scores
    sim_scores = similarity_df[selected_movie].sort_values(ascending=False)
    recommended = sim_scores.iloc[1:num_recommendations+1]
    return recommended



# Step 7: Example
if __name__ == "__main__":
    movie = input("Enter a movie name: ")
    print("\nRecommended Movies:\n")
    print(recommend_movies(movie, 5))
