from __future__ import annotations

from urllib.parse import quote_plus

import streamlit as st

from recommender import MovieRecommender, Recommendation


st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Preparing recommendations...")
def get_engine() -> MovieRecommender:
    return MovieRecommender.from_csv(max_movies=900, minimum_ratings=10)


engine = get_engine()

if "watchlist" not in st.session_state:
    st.session_state.watchlist = set()


def youtube_url(title: str) -> str:
    return f"https://www.youtube.com/results?search_query={quote_plus(title + ' official trailer')}"


def search_url(title: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(title + ' movie')}"


def save_movie(title: str) -> None:
    st.session_state.watchlist.add(title)


def remove_movie(title: str) -> None:
    st.session_state.watchlist.discard(title)


def match_percentage(score: float) -> int:
    return max(0, min(100, round(score * 100)))


def render_movie(rec: Recommendation, key: str) -> None:
    genres = " • ".join(rec.genres[:3]) or "Movie"
    year = str(rec.year) if rec.year else "Year unavailable"
    rating = f"{rec.average_rating:.1f} / 5" if rec.average_rating else "Not rated"

    with st.container(border=True):
        top_left, top_right = st.columns([4, 1.5], vertical_alignment="center")
        with top_left:
            st.markdown(f"#### {rec.title}")
            st.caption(f"{year}  •  {genres}")
        with top_right:
            st.metric("Match", f"{match_percentage(rec.score)}%")

        st.write(rec.reason)

        m1, m2 = st.columns(2)
        with m1:
            st.caption(f"★ {rating}")
        with m2:
            st.caption(f"{rec.rating_count:,} viewer ratings")

        b1, b2, b3 = st.columns(3)
        with b1:
            st.link_button("Watch trailer", youtube_url(rec.title), use_container_width=True)
        with b2:
            st.link_button("Movie details", search_url(rec.title), use_container_width=True)
        with b3:
            if rec.title in st.session_state.watchlist:
                if st.button("Saved ✓", key=f"remove_{key}", use_container_width=True):
                    remove_movie(rec.title)
                    st.rerun()
            else:
                if st.button("Save", key=f"save_{key}", use_container_width=True):
                    save_movie(rec.title)
                    st.rerun()


def render_grid(results: list[Recommendation], prefix: str) -> None:
    for index in range(0, len(results), 2):
        left, right = st.columns(2, gap="large")
        pair = results[index : index + 2]
        with left:
            render_movie(pair[0], f"{prefix}_{index}")
        if len(pair) > 1:
            with right:
                render_movie(pair[1], f"{prefix}_{index + 1}")


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3, h4 {
        letter-spacing: -0.02em;
    }
    .brand-kicker {
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: .4rem;
    }
    .hero-title {
        font-size: clamp(3rem, 7vw, 5.3rem);
        line-height: .95;
        font-weight: 800;
        letter-spacing: -.055em;
        margin: 0;
    }
    .hero-copy {
        max-width: 720px;
        color: #aeb8c7;
        font-size: 1.05rem;
        line-height: 1.65;
        margin-top: 1rem;
    }
    .section-copy {
        color: #93a0b3;
        margin-top: -.35rem;
        margin-bottom: 1.2rem;
    }
    [data-testid="stMetric"] {
        background: transparent;
        border: none;
        padding: 0;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #10141d;
        border-color: #242b38;
        border-radius: 18px;
    }
    div.stButton > button,
    div.stLinkButton > a {
        border-radius: 10px;
        min-height: 2.55rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: .25rem;
        border-bottom: 1px solid #242b38;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    [data-testid="stSidebar"] {
        background: #0b0e14;
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

summary = engine.catalog_summary()

st.markdown('<div class="brand-kicker">Movie discovery, made personal</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">CineMatch.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Find your next movie from a simple mood, genre or era — or start with a title you already love. '
    'Recommendations combine audience similarity, genre overlap and rating quality, with a clear reason behind every pick.</div>',
    unsafe_allow_html=True,
)

st.write("")
stat1, stat2, stat3, stat4 = st.columns(4)
with stat1:
    st.metric("Movies", f"{summary['movies']:,}")
with stat2:
    st.metric("Viewer ratings", f"{summary['ratings']:,}")
with stat3:
    st.metric("Audience profiles", f"{summary['users']:,}")
with stat4:
    st.metric("Genres", summary["genres"])

st.write("")

with st.sidebar:
    st.markdown("## CineMatch")
    st.caption("Hybrid movie recommendation using MovieLens audience behavior and movie metadata.")
    st.divider()
    st.markdown("### Saved movies")
    if st.session_state.watchlist:
        for title in sorted(st.session_state.watchlist):
            st.write(f"• {title}")
        if st.button("Clear saved movies", use_container_width=True):
            st.session_state.watchlist.clear()
            st.rerun()
    else:
        st.caption("Movies you save will appear here during this session.")
    st.divider()
    st.caption("No external paid service is required for recommendations.")

smart_tab, similar_tab, genre_tab, saved_tab = st.tabs(
    ["Discover", "Similar movies", "Browse genres", "Saved"]
)

with smart_tab:
    st.markdown("## What are you in the mood for?")
    st.markdown(
        '<div class="section-copy">Describe it naturally. Examples: “funny 90s movie”, “dark sci-fi”, “family adventure”, “recent romance”.</div>',
        unsafe_allow_html=True,
    )

    q1, q2 = st.columns([2, 1], gap="large")
    with q1:
        query = st.text_input(
            "Describe your movie",
            placeholder="Try: a clever sci-fi movie from the 90s",
            label_visibility="collapsed",
            key="discovery_query",
        )
    with q2:
        selected_genres = st.multiselect(
            "Genres",
            engine.available_genres,
            placeholder="Optional genres",
            label_visibility="collapsed",
        )

    results = engine.discover(query, selected_genres=selected_genres, limit=8)
    if not results:
        st.info("No close matches found. Try a broader description or remove a genre filter.")
    else:
        st.caption("Recommended for you")
        render_grid(results, "discover")

with similar_tab:
    st.markdown("## Find movies like one you love")
    st.markdown(
        '<div class="section-copy">Search for a title, choose the closest match, and CineMatch will blend audience behavior with content similarity.</div>',
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "Search title",
        placeholder="Type a movie title",
        label_visibility="collapsed",
        key="similar_search",
    )

    if search:
        suggestions = engine.search_titles(search, limit=8)
        if not suggestions:
            st.info("No close title found. Try fewer words or another spelling.")
        else:
            selected = st.selectbox("Choose a title", suggestions, index=0)
            canonical, results = engine.recommend_similar(selected, limit=8)
            if canonical:
                st.caption(f"Because you chose {canonical}")
            render_grid(results, "similar")
    else:
        st.info("Start typing a movie title to see similar recommendations.")

with genre_tab:
    st.markdown("## Browse the catalog")
    st.markdown(
        '<div class="section-copy">Explore strong picks inside a genre, ranked with rating quality and audience confidence.</div>',
        unsafe_allow_html=True,
    )

    default_index = engine.available_genres.index("Drama") if "Drama" in engine.available_genres else 0
    genre = st.selectbox("Genre", engine.available_genres, index=default_index)
    results = engine.top_by_genre(genre, limit=8)
    render_grid(results, f"genre_{genre}")

with saved_tab:
    st.markdown("## Saved movies")
    st.markdown(
        '<div class="section-copy">Keep a short list while you decide what to watch.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.watchlist:
        st.info("Nothing saved yet. Use the Save button on any movie card.")
    else:
        for index, title in enumerate(sorted(st.session_state.watchlist)):
            with st.container(border=True):
                left, right = st.columns([3, 2], vertical_alignment="center")
                with left:
                    st.markdown(f"#### {title}")
                with right:
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        st.link_button("Trailer", youtube_url(title), use_container_width=True)
                    with b2:
                        st.link_button("Details", search_url(title), use_container_width=True)
                    with b3:
                        if st.button("Remove", key=f"saved_remove_{index}", use_container_width=True):
                            remove_movie(title)
                            st.rerun()

st.write("")
st.divider()
st.caption("CineMatch • Python • scikit-learn • Streamlit • Hybrid recommendation engine")
