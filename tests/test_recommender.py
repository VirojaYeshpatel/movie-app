import pandas as pd

from recommender import MovieRecommender


def build_engine():
    movies = pd.DataFrame(
        [
            (1, "Galaxy Quest (1999)", "Adventure|Comedy|Sci-Fi"),
            (2, "Space Laughs (1998)", "Comedy|Sci-Fi"),
            (3, "Deep Space (2019)", "Drama|Sci-Fi"),
            (4, "Love Story (1995)", "Romance|Drama"),
            (5, "Family Fun (1997)", "Children|Comedy"),
        ],
        columns=["movieId", "title", "genres"],
    )
    ratings = pd.DataFrame(
        [
            (1, 1, 5.0), (1, 2, 4.5), (1, 3, 3.0),
            (2, 1, 4.5), (2, 2, 5.0), (2, 4, 2.0),
            (3, 1, 4.0), (3, 2, 4.5), (3, 5, 3.5),
            (4, 3, 5.0), (4, 4, 4.5), (4, 5, 3.0),
        ],
        columns=["userId", "movieId", "rating"],
    )
    return MovieRecommender(movies, ratings, minimum_ratings=1, max_movies=10)


def test_search_titles_fuzzy():
    engine = build_engine()
    assert engine.search_titles("galaxi", limit=1)[0].startswith("Galaxy Quest")


def test_recommend_similar_excludes_source():
    engine = build_engine()
    source, results = engine.recommend_similar("Galaxy Quest", limit=3)
    assert source == "Galaxy Quest (1999)"
    assert all(item.title != source for item in results)


def test_discovery_understands_decade_and_mood():
    engine = build_engine()
    results = engine.discover("funny 90s sci-fi", limit=3)
    titles = [item.title for item in results]
    assert "Galaxy Quest (1999)" in titles
    assert "Space Laughs (1998)" in titles


def test_top_by_genre_returns_only_requested_genre():
    engine = build_engine()
    results = engine.top_by_genre("Romance", limit=5)
    assert results
    assert all("Romance" in item.genres for item in results)
