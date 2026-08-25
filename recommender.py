from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


MOOD_TO_GENRES = {
    "funny": {"Comedy"},
    "comedy": {"Comedy"},
    "romantic": {"Romance"},
    "romance": {"Romance"},
    "scary": {"Horror", "Thriller"},
    "horror": {"Horror"},
    "dark": {"Thriller", "Crime", "Mystery", "Drama"},
    "thrilling": {"Thriller", "Action"},
    "action": {"Action"},
    "adventure": {"Adventure"},
    "family": {"Children", "Animation", "Adventure"},
    "kids": {"Children", "Animation"},
    "animated": {"Animation"},
    "animation": {"Animation"},
    "sci fi": {"Sci-Fi"},
    "sci-fi": {"Sci-Fi"},
    "science fiction": {"Sci-Fi"},
    "fantasy": {"Fantasy"},
    "mystery": {"Mystery"},
    "crime": {"Crime"},
    "dramatic": {"Drama"},
    "drama": {"Drama"},
    "musical": {"Musical"},
    "war": {"War"},
    "western": {"Western"},
    "documentary": {"Documentary"},
}

STOPWORDS = {
    "a", "an", "and", "the", "movie", "movies", "film", "films", "something",
    "with", "for", "from", "in", "of", "to", "me", "show", "give", "want",
    "watch", "good", "best", "like", "similar", "that", "this", "please",
}


@dataclass(frozen=True)
class Recommendation:
    title: str
    score: float
    genres: tuple[str, ...]
    year: int | None
    reason: str
    rating_count: int
    average_rating: float


class MovieRecommender:
    """Hybrid movie recommendation engine using collaborative and content signals."""

    def __init__(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        *,
        max_movies: int = 900,
        minimum_ratings: int = 10,
    ) -> None:
        self.movies = movies.copy()
        self.ratings = ratings.copy()
        self.max_movies = max_movies
        self.minimum_ratings = minimum_ratings
        self._prepare()

    @classmethod
    def from_csv(
        cls,
        movies_path: str | Path = "movies.csv",
        ratings_path: str | Path = "ratings.csv",
        **kwargs,
    ) -> "MovieRecommender":
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
        return cls(movies, ratings, **kwargs)

    def _prepare(self) -> None:
        required_movie_cols = {"movieId", "title", "genres"}
        required_rating_cols = {"userId", "movieId", "rating"}
        if not required_movie_cols.issubset(self.movies.columns):
            missing = required_movie_cols - set(self.movies.columns)
            raise ValueError(f"movies data is missing columns: {sorted(missing)}")
        if not required_rating_cols.issubset(self.ratings.columns):
            missing = required_rating_cols - set(self.ratings.columns)
            raise ValueError(f"ratings data is missing columns: {sorted(missing)}")

        self.movies["year"] = (
            self.movies["title"]
            .str.extract(r"\((\d{4})\)\s*$", expand=False)
            .pipe(pd.to_numeric, errors="coerce")
            .astype("Int64")
        )
        self.movies["clean_title"] = (
            self.movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()
        )
        self.movies["genre_list"] = self.movies["genres"].fillna("(no genres listed)").str.split("|")

        stats = (
            self.ratings.groupby("movieId")["rating"]
            .agg(rating_count="count", average_rating="mean")
            .reset_index()
        )
        self.movies = self.movies.merge(stats, on="movieId", how="left")
        self.movies["rating_count"] = self.movies["rating_count"].fillna(0).astype(int)
        self.movies["average_rating"] = self.movies["average_rating"].fillna(0.0)

        eligible = self.movies[self.movies["rating_count"] >= self.minimum_ratings].copy()
        if eligible.empty:
            eligible = self.movies.copy()

        eligible = eligible.sort_values(
            ["rating_count", "average_rating"], ascending=[False, False]
        ).head(self.max_movies)

        self.catalog = eligible.reset_index(drop=True)
        self.catalog_ids = self.catalog["movieId"].tolist()
        self.id_to_index = {movie_id: i for i, movie_id in enumerate(self.catalog_ids)}

        ratings = self.ratings[self.ratings["movieId"].isin(self.catalog_ids)]
        pivot = ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            fill_value=0.0,
        ).reindex(columns=self.catalog_ids, fill_value=0.0)

        movie_user = pivot.T.to_numpy(dtype=np.float32)
        self.collaborative_similarity = cosine_similarity(movie_user)

        mlb = MultiLabelBinarizer()
        genre_features = mlb.fit_transform(self.catalog["genre_list"])
        self.genre_classes = list(mlb.classes_)
        self.content_similarity = cosine_similarity(genre_features)

        popularity = np.log1p(self.catalog["rating_count"].to_numpy(dtype=float))
        popularity /= popularity.max() if popularity.max() else 1.0
        rating_quality = self.catalog["average_rating"].to_numpy(dtype=float) / 5.0
        self.popularity_score = (0.65 * popularity) + (0.35 * rating_quality)

    @property
    def available_genres(self) -> list[str]:
        genres = {
            genre
            for items in self.catalog["genre_list"]
            for genre in items
            if genre != "(no genres listed)"
        }
        return sorted(genres)

    def search_titles(self, query: str, limit: int = 8) -> list[str]:
        query = query.strip()
        if not query:
            return []

        titles = self.catalog["title"].tolist()
        clean = self.catalog["clean_title"].tolist()
        lowered = query.lower()

        direct = [
            title
            for title, clean_title in zip(titles, clean)
            if lowered in title.lower() or lowered in clean_title.lower()
        ]
        if direct:
            return direct[:limit]

        lookup = {clean_title.lower(): title for clean_title, title in zip(clean, titles)}
        fuzzy = difflib.get_close_matches(lowered, lookup.keys(), n=limit, cutoff=0.35)
        return [lookup[item] for item in fuzzy]

    def _row_for_title(self, title: str) -> pd.Series | None:
        exact = self.catalog[self.catalog["title"].str.lower() == title.lower()]
        if not exact.empty:
            return exact.iloc[0]

        matches = self.search_titles(title, limit=1)
        if not matches:
            return None
        return self.catalog[self.catalog["title"] == matches[0]].iloc[0]

    def recommend_similar(
        self,
        title: str,
        *,
        limit: int = 8,
        collaborative_weight: float = 0.68,
        content_weight: float = 0.24,
        popularity_weight: float = 0.08,
    ) -> tuple[str | None, list[Recommendation]]:
        row = self._row_for_title(title)
        if row is None:
            return None, []

        idx = self.id_to_index[int(row["movieId"])]
        scores = (
            collaborative_weight * self.collaborative_similarity[idx]
            + content_weight * self.content_similarity[idx]
            + popularity_weight * self.popularity_score
        )
        scores[idx] = -1.0

        ranked = np.argsort(scores)[::-1][:limit]
        source_genres = set(row["genre_list"])
        results: list[Recommendation] = []

        for candidate_idx in ranked:
            candidate = self.catalog.iloc[candidate_idx]
            shared = source_genres.intersection(candidate["genre_list"])
            reason_parts = []
            if shared:
                reason_parts.append("shared " + ", ".join(sorted(shared)[:2]) + " themes")
            collab = self.collaborative_similarity[idx, candidate_idx]
            if collab >= 0.45:
                reason_parts.append("strong audience overlap")
            if candidate["average_rating"] >= 4.0:
                reason_parts.append("high viewer rating")
            reason = " • ".join(reason_parts) or "strong overall similarity"

            results.append(self._to_recommendation(candidate, float(scores[candidate_idx]), reason))

        return str(row["title"]), results

    def discover(
        self,
        query: str,
        *,
        limit: int = 12,
        selected_genres: Iterable[str] | None = None,
    ) -> list[Recommendation]:
        query = query.strip().lower()
        requested_genres = set(selected_genres or [])
        requested_genres.update(self._genres_from_query(query))

        year_min, year_max = self._year_range_from_query(query)
        keywords = self._keywords_from_query(query)

        scores = self.popularity_score.copy() * 0.28
        reasons: list[list[str]] = [[] for _ in range(len(self.catalog))]

        if requested_genres:
            genre_scores = np.zeros(len(self.catalog), dtype=float)
            for i, genres in enumerate(self.catalog["genre_list"]):
                overlap = requested_genres.intersection(genres)
                if overlap:
                    genre_scores[i] = len(overlap) / len(requested_genres)
                    reasons[i].append("matches " + ", ".join(sorted(overlap)))
            scores += 0.54 * genre_scores

        if year_min is not None or year_max is not None:
            for i, year in enumerate(self.catalog["year"]):
                if pd.isna(year):
                    scores[i] -= 0.25
                    continue
                year_int = int(year)
                in_range = (year_min is None or year_int >= year_min) and (
                    year_max is None or year_int <= year_max
                )
                if in_range:
                    scores[i] += 0.18
                    reasons[i].append("fits the requested era")
                else:
                    scores[i] -= 0.35

        if keywords:
            for i, row in self.catalog.iterrows():
                haystack = f"{row['clean_title']} {' '.join(row['genre_list'])}".lower()
                hits = [word for word in keywords if word in haystack]
                if hits:
                    scores[i] += min(0.25, 0.08 * len(hits))
                    reasons[i].append("matches your search terms")

        if not requested_genres and year_min is None and year_max is None and not keywords:
            scores += 0.45 * (self.catalog["average_rating"].to_numpy(dtype=float) / 5.0)

        ranked = np.argsort(scores)[::-1]
        results: list[Recommendation] = []
        for idx in ranked:
            if len(results) >= limit:
                break
            if scores[idx] <= 0:
                continue
            row = self.catalog.iloc[idx]
            reason = " • ".join(reasons[idx]) or "popular with strong viewer ratings"
            results.append(self._to_recommendation(row, float(scores[idx]), reason))

        return results

    def top_by_genre(self, genre: str, *, limit: int = 10) -> list[Recommendation]:
        mask = self.catalog["genre_list"].apply(lambda items: genre in items)
        filtered = self.catalog[mask].copy()
        if filtered.empty:
            return []

        filtered["rank_score"] = (
            0.55 * (filtered["average_rating"] / 5.0)
            + 0.45
            * (
                np.log1p(filtered["rating_count"])
                / np.log1p(max(filtered["rating_count"].max(), 1))
            )
        )
        filtered = filtered.sort_values(["rank_score", "rating_count"], ascending=False).head(limit)
        return [
            self._to_recommendation(row, float(row["rank_score"]), f"standout {genre} pick")
            for _, row in filtered.iterrows()
        ]

    def catalog_summary(self) -> dict[str, int | float]:
        return {
            "movies": int(len(self.catalog)),
            "ratings": int(self.ratings["rating"].count()),
            "users": int(self.ratings["userId"].nunique()),
            "genres": int(len(self.available_genres)),
        }

    def _to_recommendation(
        self, row: pd.Series, score: float, reason: str
    ) -> Recommendation:
        year = None if pd.isna(row["year"]) else int(row["year"])
        return Recommendation(
            title=str(row["title"]),
            score=max(0.0, min(1.0, score)),
            genres=tuple(row["genre_list"]),
            year=year,
            reason=reason,
            rating_count=int(row["rating_count"]),
            average_rating=float(row["average_rating"]),
        )

    @staticmethod
    def _genres_from_query(query: str) -> set[str]:
        normalized = query.replace("_", " ")
        genres: set[str] = set()
        for phrase, mapped in MOOD_TO_GENRES.items():
            if phrase in normalized:
                genres.update(mapped)
        return genres

    @staticmethod
    def _year_range_from_query(query: str) -> tuple[int | None, int | None]:
        decade_match = re.search(r"\b(19\d0|20\d0)s\b", query)
        if decade_match:
            start = int(decade_match.group(1))
            return start, start + 9

        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", query)
        if year_match:
            year = int(year_match.group(1))
            return year, year

        if "classic" in query:
            return None, 1989
        if "recent" in query or "new" in query or "modern" in query:
            return 2015, None
        return None, None

    @staticmethod
    def _keywords_from_query(query: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        return [token for token in tokens if len(token) >= 3 and token not in STOPWORDS]
