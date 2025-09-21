import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# --- Optimized Data Loading with Memory Management ---
@st.cache_data
def load_data():
    try:
        if os.path.exists("similarity_matrix.pkl"):
            with open("similarity_matrix.pkl", "rb") as f:
                similarity_df = pickle.load(f)
            movies = pd.read_csv("movies.csv")
            return similarity_df, movies

        if os.path.exists("movies.csv") and os.path.exists("ratings.csv"):
            movies = pd.read_csv("movies.csv")
            ratings = pd.read_csv("ratings.csv")

            if len(ratings) > 100000:  # Limit to 100k ratings
                ratings = ratings.sample(n=100000, random_state=42)

            movie_counts = ratings['movieId'].value_counts()
            popular_movies = movie_counts[movie_counts >= 50].index
            ratings = ratings[ratings['movieId'].isin(popular_movies)]
            movies = movies[movies['movieId'].isin(popular_movies)]

            movie_data = pd.merge(ratings, movies, on="movieId")
            user_movie_matrix = movie_data.pivot_table(
                index="userId",
                columns="title",
                values="rating",
                fill_value=0
            )

            if user_movie_matrix.shape[1] > 500:
                movie_rating_counts = user_movie_matrix.sum(axis=0).sort_values(ascending=False)
                top_movies = movie_rating_counts.head(500).index
                user_movie_matrix = user_movie_matrix[top_movies]

            similarity_df = compute_similarity_chunked(user_movie_matrix)

            with open("similarity_matrix.pkl", "wb") as f:
                pickle.dump(similarity_df, f)

            return similarity_df, movies

    except Exception as e:
        st.warning(f"Error loading data: {str(e)}. Using sample data.")

    sample_movies = [
        "Toy Story", "Jumanji", "Grumpier Old Men", "Waiting to Exhale",
        "Father of the Bride Part II", "Heat", "Sabrina", "Tom and Huck",
        "Sudden Death", "GoldenEye", "The American President", "Dracula: Dead and Loving It",
        "Balto", "Nixon", "Cutthroat Island", "Casino", "Sense and Sensibility",
        "Four Rooms", "Ace Ventura: When Nature Calls", "Money Train"
    ]

    np.random.seed(42)
    similarity_matrix = np.random.rand(len(sample_movies), len(sample_movies))
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    np.fill_diagonal(similarity_matrix, 1.0)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=sample_movies,
        columns=sample_movies
    )

    movies_df = pd.DataFrame({
        'movieId': range(1, len(sample_movies) + 1),
        'title': sample_movies
    })

    return similarity_df, movies_df


def compute_similarity_chunked(user_movie_matrix, chunk_size=50):
    movies = user_movie_matrix.columns
    n_movies = len(movies)
    similarity_matrix = np.zeros((n_movies, n_movies))

    for i in range(0, n_movies, chunk_size):
        end_i = min(i + chunk_size, n_movies)
        chunk_i = user_movie_matrix.iloc[:, i:end_i].T

        for j in range(0, n_movies, chunk_size):
            end_j = min(j + chunk_size, n_movies)
            chunk_j = user_movie_matrix.iloc[:, j:end_j].T

            sim_chunk = cosine_similarity(chunk_i, chunk_j)
            similarity_matrix[i:end_i, j:end_j] = sim_chunk

    return pd.DataFrame(similarity_matrix, index=movies, columns=movies)


try:
    similarity_df, movies_df = load_data()
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()


def recommend_movies(movie_name, num_recommendations=5):
    if similarity_df.empty:
        return None, pd.Series(dtype=float)

    matches = [m for m in similarity_df.columns if movie_name.lower() in m.lower()]
    if not matches:
        matches = [m for m in similarity_df.columns
                   if any(word.lower() in m.lower() for word in movie_name.split())]

    if not matches:
        return None, pd.Series(dtype=float)

    selected_movie = matches[0]
    sim_scores = similarity_df[selected_movie].sort_values(ascending=False)

    recommendations = sim_scores.iloc[1:num_recommendations + 1]
    return selected_movie, recommendations


if 'selected_genre' not in st.session_state:
    st.session_state.selected_genre = None


# --- Streamlit UI ---
st.markdown("""
    <style>
    .metric-container {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .recommendation-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommendation System")
st.markdown("*Discover movies you'll love based on your preferences*")

if not similarity_df.empty:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'''
        <div class="metric-container">
            <h3>{len(similarity_df.columns)}</h3>
            <p>Movies Available</p>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="metric-container">
            <h3>AI-Powered</h3>
            <p>Recommendations</p>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="metric-container">
            <h3>Instant</h3>
            <p>Results</p>
        </div>
        ''', unsafe_allow_html=True)

st.markdown("---")

# --- Movie Search Section ---
st.header("🔍 Find Similar Movies")
st.markdown("Enter a movie name to get personalized recommendations based on user ratings and preferences.")

if 'movie_search' not in st.session_state:
    st.session_state.movie_search = ""

movie_input = st.text_input(
    "Movie Name:",
    value=st.session_state.movie_search,
    placeholder="e.g., Toy Story, Titanic, The Matrix...",
    help="Type any movie name to find similar recommendations"
)

if movie_input:
    with st.spinner("🔄 Analyzing movie preferences and finding recommendations..."):
        selected, recommended = recommend_movies(movie_input, 8)

    if selected and not recommended.empty:
        st.markdown(f'''
        <div class="recommendation-header">
            🎯 Movies Similar to "{selected}"
        </div>
        ''', unsafe_allow_html=True)

        cols = st.columns(2)
        for i, (movie, score) in enumerate(recommended.items()):
            col_idx = i % 2
            with cols[col_idx]:
                with st.container():
                    st.markdown(f"**🎬 {movie}**")
                    st.write(f"Similarity Score: {score:.3f}")
                    st.write(f"Rank: #{i+1}")

                    # Add AI feature: YouTube trailer + Google info links
                    youtube_url = f"https://www.youtube.com/results?search_query={movie.replace(' ', '+')}+trailer"
                    google_url = f"https://www.google.com/search?q={movie.replace(' ', '+')}+movie"

                    st.markdown(f"[▶ Watch Trailer]({youtube_url})", unsafe_allow_html=True)
                    st.markdown(f"[🔗 More Info]({google_url})", unsafe_allow_html=True)

                    st.markdown("---")

    else:
        st.error("❌ No matches found. Please check the movie name spelling.")
        if not similarity_df.empty:
            st.info("💡 **Available movies in our database:**")
            sample_movies = list(similarity_df.columns)[:15]
            cols = st.columns(3)
            for i, movie in enumerate(sample_movies):
                col_idx = i % 3
                with cols[col_idx]:
                    if st.button(f"🎬 {movie}", key=f"sample_{i}"):
                        st.session_state.movie_search = movie
                        st.rerun()

# --- Genre Recommendations Section ---
st.markdown("---")
st.header("🎭 Browse by Genre")
st.markdown("Explore curated movie recommendations by genre")

genres = {
    "🏆 Top Rated": [
        "The Shawshank Redemption", "The Godfather", "The Dark Knight",
        "12 Angry Men", "Schindler's List"
    ],
    "🎯 Action": [
        "Mad Max: Fury Road", "John Wick", "The Matrix",
        "Mission: Impossible", "Die Hard"
    ],
    "💕 Romance": [
        "Titanic", "The Notebook", "Casablanca",
        "Pride and Prejudice", "When Harry Met Sally"
    ],
    "😄 Comedy": [
        "The Grand Budapest Hotel", "Superbad", "Anchorman",
        "Bridesmaids", "The Hangover"
    ],
    "👻 Horror": [
        "The Exorcist", "Halloween", "A Quiet Place",
        "Get Out", "Hereditary"
    ],
    "🚀 Sci-Fi": [
        "Blade Runner 2049", "Interstellar", "The Matrix",
        "Arrival", "Ex Machina"
    ]
}

genre_cols = st.columns(3)
selected_genre = None
for i, genre in enumerate(genres.keys()):
    col_idx = i % 3
    with genre_cols[col_idx]:
        if st.button(genre, key=f"genre_{i}", use_container_width=True):
            selected_genre = genre

if selected_genre:
    st.subheader(f"{selected_genre} Movies")
    movies = genres[selected_genre]
    cols = st.columns(2)
    for i, movie in enumerate(movies):
        col_idx = i % 2
        with cols[col_idx]:
            with st.container():
                st.markdown(f"**🎬 {movie}**")
                st.write(f"Recommended for {selected_genre.lower()} enthusiasts")

                # Add YouTube + Google links here too
                youtube_url = f"https://www.youtube.com/results?search_query={movie.replace(' ', '+')}+trailer"
                google_url = f"https://www.google.com/search?q={movie.replace(' ', '+')}+movie"

                st.markdown(f"[▶ Watch Trailer]({youtube_url})", unsafe_allow_html=True)
                st.markdown(f"[🔗 More Info]({google_url})", unsafe_allow_html=True)

                st.markdown("---")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <h3>🎬 Movie Recommendation System</h3>
    <p>Powered by Machine Learning • Built with Streamlit</p>
    <p>Discover your next favorite movie!</p>
</div>
""", unsafe_allow_html=True)
