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


def match_percentage(score: float) -> int:
    return max(0, min(100, round(score * 100)))


def render_movie(rec: Recommendation, key: str) -> None:
    genres = " • ".join(rec.genres[:3]) or "Movie"
    year = str(rec.year) if rec.year else "—"
    rating = f"{rec.average_rating:.1f}" if rec.average_rating else "—"

    st.markdown(
        f"""
        <div class="cinema-card">
            <div class="card-glow"></div>
            <div class="card-topline">
                <span class="match-chip">{match_percentage(rec.score)}% MATCH</span>
                <span class="rating-chip">★ {rating}</span>
            </div>
            <div class="movie-title">{rec.title}</div>
            <div class="movie-meta">{year} &nbsp;•&nbsp; {genres}</div>
            <div class="movie-reason">{rec.reason}</div>
            <div class="movie-foot">{rec.rating_count:,} viewer ratings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.link_button("▶ Trailer", youtube_url(rec.title), use_container_width=True)
    with c2:
        st.link_button("Details", search_url(rec.title), use_container_width=True)
    with c3:
        if rec.title in st.session_state.watchlist:
            if st.button("Saved ✓", key=f"remove_{key}", use_container_width=True):
                st.session_state.watchlist.discard(rec.title)
                st.rerun()
        else:
            if st.button("+ Save", key=f"save_{key}", use_container_width=True):
                st.session_state.watchlist.add(rec.title)
                st.rerun()


def render_grid(results: list[Recommendation], prefix: str) -> None:
    for index in range(0, len(results), 2):
        left, right = st.columns(2, gap="large")
        with left:
            render_movie(results[index], f"{prefix}_{index}")
        if index + 1 < len(results):
            with right:
                render_movie(results[index + 1], f"{prefix}_{index + 1}")


st.markdown(
    """
    <style>
    :root {
        --bg: #05070b;
        --surface: rgba(16, 20, 29, .72);
        --surface-strong: rgba(20, 25, 36, .92);
        --border: rgba(255,255,255,.10);
        --muted: #9aa6b8;
        --text: #f7f8fb;
        --violet: #8b5cf6;
        --cyan: #22d3ee;
        --pink: #ec4899;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background:
            radial-gradient(circle at 12% 18%, rgba(139,92,246,.13), transparent 28%),
            radial-gradient(circle at 90% 8%, rgba(34,211,238,.10), transparent 24%),
            linear-gradient(180deg, #05070b 0%, #080b12 56%, #05070b 100%) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(5,7,11,.72);
        backdrop-filter: blur(18px);
        border-bottom: 1px solid rgba(255,255,255,.05);
    }

    .block-container {
        max-width: 1220px;
        padding-top: 5.4rem !important;
        padding-bottom: 5rem !important;
    }

    [data-testid="stSidebar"] {
        background: #080b12;
        border-right: 1px solid rgba(255,255,255,.07);
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        min-height: 480px;
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 34px;
        padding: 58px 58px 46px;
        background:
            linear-gradient(135deg, rgba(18,23,34,.92), rgba(9,12,19,.80)),
            radial-gradient(circle at 75% 20%, rgba(139,92,246,.18), transparent 34%);
        box-shadow:
            0 40px 120px rgba(0,0,0,.48),
            inset 0 1px 0 rgba(255,255,255,.08);
        transform-style: preserve-3d;
        perspective: 1100px;
    }

    .hero-grid {
        position: absolute;
        inset: 0;
        opacity: .20;
        background-image:
            linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
        background-size: 38px 38px;
        mask-image: linear-gradient(to bottom, black, transparent 85%);
    }

    .orb-wrap {
        position: absolute;
        right: 42px;
        top: 42px;
        width: 360px;
        height: 360px;
        transform: rotateX(12deg) rotateY(-18deg);
        filter: drop-shadow(0 44px 70px rgba(76,29,149,.34));
    }

    .orb {
        position: absolute;
        inset: 35px;
        border-radius: 50%;
        background:
            radial-gradient(circle at 35% 28%, rgba(255,255,255,.92) 0 4%, rgba(255,255,255,.22) 8%, transparent 22%),
            radial-gradient(circle at 30% 30%, #c4b5fd 0%, #7c3aed 28%, #312e81 58%, #090b12 76%);
        box-shadow:
            inset -28px -34px 64px rgba(0,0,0,.55),
            inset 24px 20px 55px rgba(255,255,255,.10),
            0 0 70px rgba(139,92,246,.35);
        animation: floatOrb 6s ease-in-out infinite;
    }

    .ring {
        position: absolute;
        inset: 0;
        border-radius: 50%;
        border: 2px solid rgba(34,211,238,.24);
        transform: rotateX(68deg) rotateZ(-16deg);
        box-shadow: 0 0 45px rgba(34,211,238,.15);
        animation: spinRing 12s linear infinite;
    }

    .ring.r2 {
        inset: 42px -18px;
        border-color: rgba(236,72,153,.20);
        transform: rotateX(72deg) rotateZ(48deg);
        animation-duration: 16s;
    }

    @keyframes floatOrb {
        0%,100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-14px) scale(1.025); }
    }

    @keyframes spinRing {
        from { transform: rotateX(68deg) rotateZ(-16deg) rotate(0deg); }
        to   { transform: rotateX(68deg) rotateZ(-16deg) rotate(360deg); }
    }

    .hero-copy-wrap {
        position: relative;
        z-index: 2;
        width: min(690px, 66%);
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 18px;
        padding: 8px 12px;
        border: 1px solid rgba(167,139,250,.22);
        border-radius: 999px;
        background: rgba(139,92,246,.08);
        color: #c4b5fd;
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 0;
        color: var(--text);
        font-size: clamp(4rem, 8vw, 7.1rem);
        line-height: .86;
        letter-spacing: -.065em;
        font-weight: 900;
        text-shadow: 0 12px 36px rgba(0,0,0,.45);
    }

    .hero-title span {
        background: linear-gradient(90deg, #ffffff, #c4b5fd 48%, #67e8f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-desc {
        max-width: 640px;
        margin-top: 26px;
        color: #b7c0ce;
        font-size: 1.05rem;
        line-height: 1.72;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 28px;
    }

    .hero-badge {
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 12px;
        padding: 10px 12px;
        background: rgba(255,255,255,.035);
        color: #d8dee8;
        font-size: .82rem;
        backdrop-filter: blur(12px);
    }

    .stats-row {
        margin: 28px 0 36px;
    }

    [data-testid="stMetric"] {
        padding: 16px 18px;
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.018));
        box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
    }

    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 800;
    }

    .section-heading {
        margin-top: 10px;
        margin-bottom: 4px;
        font-size: 2rem;
        line-height: 1.05;
        font-weight: 850;
        letter-spacing: -.035em;
    }

    .section-sub {
        color: var(--muted);
        margin-bottom: 22px;
    }

    .cinema-card {
        position: relative;
        overflow: hidden;
        min-height: 260px;
        margin-top: 12px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 24px;
        background:
            linear-gradient(145deg, rgba(21,26,38,.94), rgba(11,14,21,.88));
        box-shadow:
            0 18px 45px rgba(0,0,0,.28),
            inset 0 1px 0 rgba(255,255,255,.055);
        transform: perspective(900px) rotateX(0deg) rotateY(0deg) translateZ(0);
        transition: transform .28s ease, border-color .28s ease, box-shadow .28s ease;
    }

    .cinema-card:hover {
        transform: perspective(900px) rotateX(2.2deg) rotateY(-2.2deg) translateY(-5px) translateZ(12px);
        border-color: rgba(167,139,250,.30);
        box-shadow:
            0 28px 70px rgba(0,0,0,.42),
            0 0 42px rgba(139,92,246,.08),
            inset 0 1px 0 rgba(255,255,255,.08);
    }

    .card-glow {
        position: absolute;
        width: 180px;
        height: 180px;
        right: -65px;
        top: -70px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(139,92,246,.22), transparent 68%);
        pointer-events: none;
    }

    .card-topline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
    }

    .match-chip, .rating-chip {
        border-radius: 999px;
        padding: 7px 10px;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .04em;
    }

    .match-chip {
        color: #ddd6fe;
        background: rgba(139,92,246,.13);
        border: 1px solid rgba(167,139,250,.20);
    }

    .rating-chip {
        color: #fde68a;
        background: rgba(245,158,11,.08);
        border: 1px solid rgba(245,158,11,.15);
    }

    .movie-title {
        margin-top: 26px;
        color: #fff;
        font-size: 1.34rem;
        line-height: 1.18;
        font-weight: 800;
        letter-spacing: -.025em;
    }

    .movie-meta, .movie-foot {
        color: #8e9bae;
        font-size: .82rem;
    }

    .movie-meta { margin-top: 7px; }
    .movie-foot { margin-top: 16px; }

    .movie-reason {
        margin-top: 20px;
        color: #cdd5df;
        font-size: .94rem;
        line-height: 1.55;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        padding: 6px;
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 16px;
        background: rgba(255,255,255,.028);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 11px;
        padding: 0 16px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(139,92,246,.20), rgba(34,211,238,.09));
    }

    div.stButton > button,
    div.stLinkButton > a {
        min-height: 42px;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,.10) !important;
        background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.025)) !important;
        color: #eef2f7 !important;
        font-weight: 700 !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 8px 18px rgba(0,0,0,.16);
        transition: all .2s ease;
    }

    div.stButton > button:hover,
    div.stLinkButton > a:hover {
        transform: translateY(-2px);
        border-color: rgba(167,139,250,.34) !important;
        background: linear-gradient(135deg, rgba(139,92,246,.15), rgba(34,211,238,.07)) !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div {
        min-height: 48px;
        border-radius: 13px !important;
        background: rgba(15,19,28,.92) !important;
        border-color: rgba(255,255,255,.09) !important;
    }

    footer { visibility: hidden; }

    @media (max-width: 900px) {
        .block-container { padding-top: 4.8rem !important; }
        .hero-shell { min-height: 620px; padding: 38px 28px; }
        .hero-copy-wrap { width: 100%; }
        .hero-title { font-size: clamp(3.5rem, 18vw, 5.7rem); }
        .orb-wrap { width: 260px; height: 260px; right: -30px; top: 330px; opacity: .72; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

summary = engine.catalog_summary()

st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-grid"></div>
        <div class="orb-wrap">
            <div class="ring"></div>
            <div class="ring r2"></div>
            <div class="orb"></div>
        </div>
        <div class="hero-copy-wrap">
            <div class="eyebrow">Cinematic recommendation engine</div>
            <div class="hero-title"><span>CineMatch</span>.</div>
            <div class="hero-desc">
                Discover the right movie for your mood, era or genre — or start from a title you already love.
                CineMatch combines audience behavior, content similarity and rating quality to surface explainable recommendations.
            </div>
            <div class="hero-badges">
                <div class="hero-badge">Hybrid ranking</div>
                <div class="hero-badge">Natural-language discovery</div>
                <div class="hero-badge">Explainable picks</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="stats-row"></div>', unsafe_allow_html=True)
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric("Movies", f"{summary['movies']:,}")
with s2:
    st.metric("Viewer ratings", f"{summary['ratings']:,}")
with s3:
    st.metric("Audience profiles", f"{summary['users']:,}")
with s4:
    st.metric("Genres", summary["genres"])

with st.sidebar:
    st.markdown("## CineMatch")
    st.caption("Saved movies")
    if st.session_state.watchlist:
        for title in sorted(st.session_state.watchlist):
            st.write(f"• {title}")
        if st.button("Clear saved", use_container_width=True):
            st.session_state.watchlist.clear()
            st.rerun()
    else:
        st.caption("Nothing saved yet.")

st.write("")
discover_tab, similar_tab, genre_tab, saved_tab = st.tabs(
    ["Discover", "Similar movies", "Browse genres", "Saved"]
)

with discover_tab:
    st.markdown('<div class="section-heading">What are you in the mood for?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Try “funny 90s movie”, “dark sci-fi”, “family adventure”, or “recent romance”.</div>', unsafe_allow_html=True)

    q1, q2 = st.columns([2.1, 1], gap="large")
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
        render_grid(results, "discover")

with similar_tab:
    st.markdown('<div class="section-heading">Start with a movie you already love.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Search a title and get recommendations based on audience behavior and content similarity.</div>', unsafe_allow_html=True)

    search = st.text_input(
        "Search title",
        placeholder="Type a movie title",
        label_visibility="collapsed",
        key="similar_search",
    )

    if not search:
        st.info("Start typing a movie title.")
    else:
        suggestions = engine.search_titles(search, limit=8)
        if not suggestions:
            st.info("No close title found. Try fewer words or another spelling.")
        else:
            selected = st.selectbox("Choose a title", suggestions, index=0)
            canonical, results = engine.recommend_similar(selected, limit=8)
            if canonical:
                st.caption(f"Because you chose {canonical}")
            render_grid(results, "similar")

with genre_tab:
    st.markdown('<div class="section-heading">Explore the catalog.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Browse strong picks inside a genre, ranked by quality and audience confidence.</div>', unsafe_allow_html=True)

    default_index = engine.available_genres.index("Drama") if "Drama" in engine.available_genres else 0
    genre = st.selectbox("Genre", engine.available_genres, index=default_index)
    render_grid(engine.top_by_genre(genre, limit=8), f"genre_{genre}")

with saved_tab:
    st.markdown('<div class="section-heading">Your shortlist.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Keep a few options while you decide what to watch.</div>', unsafe_allow_html=True)

    if not st.session_state.watchlist:
        st.info("Nothing saved yet. Use + Save on any recommendation.")
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
                            st.session_state.watchlist.discard(title)
                            st.rerun()

st.write("")
st.divider()
st.caption("CineMatch • Hybrid recommendation engine • Python • scikit-learn • Streamlit")
