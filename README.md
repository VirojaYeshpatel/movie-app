# 🎬 CineMatch

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20CineMatch-8b5cf6?style=for-the-badge)](https://movie-app-ypfie7rv9e8nyzj5arxgxw.streamlit.app/)

A polished hybrid movie discovery and recommendation app built with Python, scikit-learn, pandas, NumPy and Streamlit.

CineMatch blends **collaborative filtering**, **genre/content similarity**, **rating quality**, and lightweight intent parsing to help users discover movies from either a title they already love or a natural-language request such as:

- `funny 90s sci-fi`
- `dark thriller`
- `family adventure`
- `recent romance`

## ✨ Highlights

- Hybrid recommendation engine instead of a single similarity score
- Smart discovery by mood, genre and era
- Fuzzy title search for misspelled movie names
- Explainable recommendations with human-readable reasons
- Genre exploration ranked by rating quality and audience volume
- Session watchlist for saving interesting picks
- Trailer and movie-detail links from every recommendation card
- Responsive product-style Streamlit interface
- Clear separation between UI and recommendation logic
- Automated tests for the core recommendation engine

## 🧠 Recommendation Architecture

CineMatch combines three signals:

1. **Collaborative similarity** — movies liked by similar audiences
2. **Content similarity** — overlap between movie genres
3. **Popularity / rating quality** — balances confidence and viewer satisfaction

For discovery queries, the engine also extracts signals such as mood, genre and year/decade and then ranks the catalog against those preferences.

```text
MovieLens movies + ratings
        │
        ├── user × movie interaction matrix
        │        └── cosine similarity
        │
        ├── genre multi-hot vectors
        │        └── content similarity
        │
        └── rating count + average rating
                 └── popularity quality

                 ↓
          Hybrid ranker
                 ↓
    Explainable recommendations
```

## 🖥️ Product Experience

The application includes four focused experiences:

### Discover
Describe what you feel like watching and CineMatch interprets mood, genre and era hints.

### Similar Movies
Start with a movie you already know and receive blended recommendations based on audience behavior and content overlap.

### Browse Genres
Explore standout titles inside any genre using rating quality and popularity.

### Saved
Keep a lightweight watchlist while comparing titles during the current session.

## 🛠️ Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- pytest

## 📁 Project Structure

```text
movie-app/
├── movie_app.py              # Streamlit product UI
├── recommender.py            # Hybrid recommendation engine
├── movie_recommender.py      # Legacy console recommender
├── movies.csv                # Movie metadata + genres
├── ratings.csv               # User ratings dataset
├── tests/
│   └── test_recommender.py   # Core recommendation tests
├── .streamlit/
│   └── config.toml           # App theme + runtime config
├── requirements.txt
└── README.md
```

## 🚀 Run Locally

```bash
git clone https://github.com/VirojaYeshpatel/movie-app.git
cd movie-app
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies and run the app:

```bash
pip install -r requirements.txt
streamlit run movie_app.py
```

Open:

```text
http://localhost:8501
```

## 🧪 Tests

```bash
pytest -q
```

The test suite covers fuzzy movie search, source-title exclusion, mood + decade discovery, and genre filtering.

## 🌐 Live App

CineMatch is deployed on Streamlit Community Cloud:

https://movie-app-ypfie7rv9e8nyzj5arxgxw.streamlit.app/

## 📊 Dataset

The project uses MovieLens-style movie and rating data. The recommender builds its ranking signals directly from the committed CSV datasets at startup, so no generated similarity cache needs to be stored in Git.

## 🔍 Design Decisions

- Generated model/cache artifacts are not committed
- The recommendation engine is isolated from the Streamlit UI
- Ranking uses multiple signals instead of a single cosine score
- Recommendation explanations are deterministic and traceable to ranking signals
- No external paid API is required for the core experience

## 🔮 Next Improvements

- poster metadata and richer movie detail cards
- persistent user profiles and watchlists
- personalized recommendations from explicit user ratings
- matrix-factorization benchmark against the current hybrid model
- recommendation evaluation metrics such as Precision@K and Recall@K

---

Built as a production-style recommendation portfolio project with an emphasis on explainability, clean architecture and a strong interactive product experience.
