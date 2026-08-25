from __future__ import annotations

from urllib.parse import quote_plus

import streamlit as st

from recommender import MovieRecommender, Recommendation


st.set_page_config(
    page_title="CineMatch — Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Building recommendation engine...")
def get_engine() -> MovieRecommender:
    return MovieRecommender.from_csv(max_movies=900, minimum_ratings=10)


engine = get_engine()

if "watchlist" not in st.session_state:
    st.session_state.watchlist = set()


def youtube_url(title: str) -> str:
    return f"https://www.youtube.com/results?search_query={quote_plus(title + ' official trailer')}"


def search_url(title: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(title + ' movie')}"


def score_label(score: float) -> str:
    return f"{round(score * 100)}% match"


def add_to_watchlist(title: str) -> None:
    st.session_state.watchlist.add(title)


def remove_from_watchlist(title: str) -> None:
    st.session_state.watchlist.discard(title)


def render_movie_card(rec: Recommendation, key_prefix: str) -> None:
    genres = " · ".join(rec.genres[:4])
    rating = f"{rec.average_rating:.1f}/5" if rec.average_rating else "New"
    year = rec.year or "—"

    st.markdown(
        f"""
        <div class="movie-card">
            <div class="movie-card-top">
                <span class="match-pill">{score_label(rec.score)}</span>
                <span class="rating-pill">★ {rating}</span>
            </div>
            <h3>{rec.title}</h3>
            <div class="meta">{year} &nbsp;•&nbsp; {genres}</div>
            <div class="reason">{rec.reason}</div>
            <div class="ratings-count">{rec.rating_count:,} ratings in the dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.2, 1.1, 1.2])
    with c1:
        st.link_button("▶ Trailer", youtube_url(rec.title), use_container_width=True)
    with c2:
        st.link_button("Details", search_url(rec.title), use_container_width=True)
    with c3:
        if rec.title in st.session_state.watchlist:
            if st.button("✓ Saved", key=f"{key_prefix}_remove_{rec.title}", use_container_width=True):
                remove_from_watchlist(rec.title)
                st.rerun()
        else:
            if st.button("+ Watchlist", key=f"{key_prefix}_add_{rec.title}", use_container_width=True):
                add_to_watchlist(rec.title)
                st.rerun()


st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(circle at 12% 10%, rgba(124,58,237,.15), transparent 30%),
          radial-gradient(circle at 85% 15%, rgba(14,165,233,.12), transparent 26%),
          #090b10;
        color: #f8fafc;
      }
      [data-testid="stSidebar"] {
        background: rgba(11, 15, 24, .96);
        border-right: 1px solid rgba(148,163,184,.12);
      }
      .hero {
        padding: 2.25rem 2.4rem;
        border: 1px solid rgba(148,163,184,.14);
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(30,41,59,.82), rgba(15,23,42,.54));
        box-shadow: 0 24px 80px rgba(0,0,0,.30);
        margin-bottom: 1.25rem;
      }
      .eyebrow {
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .78rem;
        font-weight: 800;
      }
      .hero h1 {
        margin: .45rem 0 .55rem;
        font-size: clamp(2.35rem, 5vw, 4.8rem);
        line-height: .98;
        letter-spacing: -.045em;
      }
      .hero p {
        color: #cbd5e1;
        font-size: 1.08rem;
        max-width: 760px;
        margin: 0;
      }
      .stat-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border: 1px solid rgba(148,163,184,.12);
        background: rgba(15,23,42,.58);
        min-height: 95px;
      }
      .stat-number {
        font-size: 1.7rem;
        font-weight: 800;
      }
      .stat-label {
        color: #94a3b8;
        font-size: .85rem;
      }
      .movie-card {
        border: 1px solid rgba(148,163,184,.14);
        border-radius: 20px;
        padding: 1.1rem 1.15rem 1rem;
        background: linear-gradient(145deg, rgba(30,41,59,.78), rgba(15,23,42,.76));
        min-height: 205px;
        box-shadow: 0 10px 35px rgba(0,0,0,.18);
      }
      .movie-card-top {
        display: flex;
        justify-content: space-between;
        gap: .5rem;
      }
      .match-pill, .rating-pill {
        font-size: .75rem;
        font-weight: 800;
        border-radius: 999px;
        padding: .3rem .55rem;
      }
      .match-pill {
        color: #ddd6fe;
        background: rgba(124,58,237,.25);
        border: 1px solid rgba(167,139,250,.28);
      }
      .rating-pill {
        color: #fde68a;
        background: rgba(245,158,11,.12);
        border: 1px solid rgba(245,158,11,.18);
      }
      .movie-card h3 {
        font-size: 1.15rem;
        margin: .9rem 0 .35rem;
      }
      .meta, .ratings-count {
        color: #94a3b8;
        font-size: .82rem;
      }
      .reason {
        margin: .9rem 0 .6rem;
        color: #dbeafe;
        font-size: .92rem;
        line-height: 1.42;
      }
      .section-title {
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: .35rem;
        margin-bottom: .2rem;
      }
      .section-copy {
        color: #94a3b8;
        margin-bottom: 1rem;
      }
      div[data-testid="stMetric"] {
        background: rgba(15,23,42,.48);
        border: 1px solid rgba(148,163,184,.12);
        padding: .7rem 1rem;
        border-radius: 16px;
      }
      .stTabs [data-baseweb="tab-list"] {
        gap: .35rem;
      }
      .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: .45rem .8rem;
      }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

summary = engine.catalog_summary()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Hybrid recommendation engine</div>
      <h1>CineMatch</h1>
      <p>Discover movies by mood, era and genre, or start from a title you already love.
      Recommendations blend audience behavior, genre similarity and rating quality.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

stat_cols = st.columns(4)
stats = [
    (f"{summary['movies']:,}", "Curated catalog"),
    (f"{summary['ratings']:,}", "Viewer ratings"),
    (f"{summary['users']:,}", "Audience profiles"),
    (f"{summary['genres']}", "Genres"),
]
for column, (value, label) in zip(stat_cols, stats):
    with column:
        st.markdown(
            f'<div class="stat-card"><div class="stat-number">{value}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

st.sidebar.markdown("## 🎬 CineMatch")
st.sidebar.caption("Recommendation intelligence built from MovieLens ratings and movie metadata.")
st.sidebar.divider()
st.sidebar.markdown("### Your watchlist")
if st.session_state.watchlist:
    for movie in sorted(st.session_state.watchlist):
        st.sidebar.write(f"• {movie}")
    if st.sidebar.button("Clear watchlist", use_container_width=True):
        st.session_state.watchlist.clear()
        st.rerun()
else:
    st.sidebar.caption("Save interesting picks and they’ll appear here.")

st.sidebar.divider()
st.sidebar.markdown("### How ranking works")
st.sidebar.caption(
    "Similar-title results combine collaborative similarity, shared genres and popularity. "
    "Discovery results interpret mood, genre and era signals before ranking titles."
)

discover_tab, similar_tab, explore_tab, watchlist_tab = st.tabs(
    ["✨ Smart Discovery", "🎯 Similar Movies", "🧭 Explore Genres", "🔖 Watchlist"]
)

with discover_tab:
    st.markdown('<div class="section-title">Describe what you feel like watching</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Try: “funny 90s movie”, “dark sci-fi”, “family adventure”, or “recent thriller”.</div>',
        unsafe_allow_html=True,
    )
    query = st.text_input(
        "Discovery prompt",
        placeholder="e.g. a funny 90s movie with adventure",
        label_visibility="collapsed",
        key="discovery_query",
    )
    selected_genres = st.multiselect(
        "Optional genre filters",
        engine.available_genres,
        default=[],
        placeholder="Add genres if you want tighter results",
    )

    if query or selected_genres:
        results = engine.discover(query, selected_genres=selected_genres, limit=12)
    else:
        results = engine.discover("", limit=12)

    if not results:
        st.warning("No titles matched those filters. Try a broader mood, genre or era.")
    else:
        for row_start in range(0, len(results), 3):
            cols = st.columns(3)
            for col, rec in zip(cols, results[row_start : row_start + 3]):
                with col:
                    render_movie_card(rec, f"discover_{row_start}")

with similar_tab:
    st.markdown('<div class="section-title">Start with a movie you already love</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Search the catalog, select a title, and get blended recommendations with explanations.</div>',
        unsafe_allow_html=True,
    )
    search = st.text_input(
        "Movie title",
        placeholder="Type a movie title...",
        label_visibility="collapsed",
        key="similar_search",
    )
    suggestions = engine.search_titles(search, limit=8) if search else []
    if search and not suggestions:
        st.info("No close title found. Try fewer words or a different spelling.")
    elif suggestions:
        selected = st.selectbox("Best matches", suggestions, index=0)
        canonical, results = engine.recommend_similar(selected, limit=12)
        if canonical:
            st.caption(f"Recommendations based on **{canonical}**")
        for row_start in range(0, len(results), 3):
            cols = st.columns(3)
            for col, rec in zip(cols, results[row_start : row_start + 3]):
                with col:
                    render_movie_card(rec, f"similar_{row_start}")

with explore_tab:
    st.markdown('<div class="section-title">Explore standout titles by genre</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Browse top-ranked picks using rating quality and audience volume.</div>',
        unsafe_allow_html=True,
    )
    genre_index = engine.available_genres.index("Drama") if "Drama" in engine.available_genres else 0
    genre = st.selectbox("Choose a genre", engine.available_genres, index=genre_index)
    results = engine.top_by_genre(genre, limit=12)
    for row_start in range(0, len(results), 3):
        cols = st.columns(3)
        for col, rec in zip(cols, results[row_start : row_start + 3]):
            with col:
                render_movie_card(rec, f"genre_{genre}_{row_start}")

with watchlist_tab:
    st.markdown('<div class="section-title">Saved for later</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A lightweight session watchlist for comparing titles before you choose.</div>',
        unsafe_allow_html=True,
    )
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Save a movie from any recommendation card.")
    else:
        for title in sorted(st.session_state.watchlist):
            st.markdown(f"### {title}")
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.link_button("▶ Trailer", youtube_url(title), use_container_width=True)
            with c2:
                st.link_button("Details", search_url(title), use_container_width=True)
            with c3:
                if st.button("Remove", key=f"watchlist_remove_{title}", use_container_width=True):
                    remove_from_watchlist(title)
                    st.rerun()
            st.divider()

st.divider()
st.caption(
    "CineMatch • Hybrid collaborative + content-based movie recommendation • "
    "Built with Python, scikit-learn and Streamlit"
)
