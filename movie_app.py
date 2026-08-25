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
            <div class="card-aurora"></div>
            <div class="card-scan"></div>
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
        --violet: #8b5cf6;
        --cyan: #22d3ee;
        --pink: #ec4899;
        --text: #f8fafc;
        --muted: #98a4b6;
    }

    html { scroll-behavior: smooth; }

    [data-testid="stAppViewContainer"] {
        background: #04060a;
        overflow: hidden;
    }

    [data-testid="stAppViewContainer"]::before,
    [data-testid="stAppViewContainer"]::after {
        content: "";
        position: fixed;
        inset: -30%;
        pointer-events: none;
        z-index: 0;
    }

    [data-testid="stAppViewContainer"]::before {
        background:
            radial-gradient(circle at 20% 28%, rgba(139,92,246,.25), transparent 22%),
            radial-gradient(circle at 76% 18%, rgba(34,211,238,.18), transparent 20%),
            radial-gradient(circle at 68% 76%, rgba(236,72,153,.14), transparent 21%),
            radial-gradient(circle at 30% 80%, rgba(59,130,246,.12), transparent 20%);
        filter: blur(50px) saturate(130%);
        animation: auroraDrift 18s ease-in-out infinite alternate;
    }

    [data-testid="stAppViewContainer"]::after {
        opacity: .32;
        background-image:
            radial-gradient(circle, rgba(255,255,255,.78) 0 1px, transparent 1.4px),
            radial-gradient(circle, rgba(167,139,250,.72) 0 1px, transparent 1.4px),
            radial-gradient(circle, rgba(103,232,249,.65) 0 1px, transparent 1.4px);
        background-size: 98px 98px, 143px 143px, 181px 181px;
        background-position: 0 0, 24px 60px, 71px 23px;
        animation: starsMove 42s linear infinite;
    }

    @keyframes auroraDrift {
        0% { transform: translate3d(-2%, -1%, 0) rotate(0deg) scale(1); }
        50% { transform: translate3d(4%, 2%, 0) rotate(5deg) scale(1.05); }
        100% { transform: translate3d(-1%, 4%, 0) rotate(-4deg) scale(1.09); }
    }

    @keyframes starsMove {
        from { transform: translate3d(0, 0, 0); }
        to { transform: translate3d(-140px, 110px, 0); }
    }

    [data-testid="stHeader"] {
        background: rgba(4,6,10,.62);
        backdrop-filter: blur(22px) saturate(140%);
        border-bottom: 1px solid rgba(255,255,255,.055);
    }

    .block-container {
        position: relative;
        z-index: 1;
        max-width: 1230px;
        padding-top: 5.6rem !important;
        padding-bottom: 5rem !important;
    }

    .hero-shell {
        position: relative;
        isolation: isolate;
        overflow: hidden;
        min-height: 520px;
        padding: 64px 60px 52px;
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 36px;
        background:
            linear-gradient(135deg, rgba(16,21,32,.86), rgba(7,10,16,.76));
        backdrop-filter: blur(26px) saturate(135%);
        box-shadow:
            0 50px 140px rgba(0,0,0,.52),
            0 0 0 1px rgba(139,92,246,.035),
            inset 0 1px 0 rgba(255,255,255,.10);
        perspective: 1200px;
        transform-style: preserve-3d;
        animation: heroEnter 1s cubic-bezier(.2,.8,.2,1) both;
    }

    .hero-shell::before {
        content: "";
        position: absolute;
        inset: -3px;
        z-index: -2;
        border-radius: inherit;
        background: conic-gradient(from var(--angle), #22d3ee22, #8b5cf655, #ec489933, #22d3ee22);
        filter: blur(16px);
        animation: borderSpin 9s linear infinite;
    }

    .hero-shell::after {
        content: "";
        position: absolute;
        inset: 1px;
        z-index: -1;
        border-radius: 34px;
        background: linear-gradient(145deg, rgba(11,15,23,.96), rgba(7,10,16,.88));
    }

    @property --angle { syntax: '<angle>'; initial-value: 0deg; inherits: false; }
    @keyframes borderSpin { to { --angle: 360deg; } }
    @keyframes heroEnter {
        from { opacity: 0; transform: translateY(26px) scale(.985); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    .hero-grid {
        position: absolute;
        inset: 0;
        opacity: .18;
        background-image:
            linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px);
        background-size: 42px 42px;
        mask-image: linear-gradient(to bottom, #000, transparent 88%);
        animation: gridSlide 20s linear infinite;
    }

    @keyframes gridSlide { to { background-position: 42px 42px; } }

    .holo-beam {
        position: absolute;
        left: -30%;
        top: -20%;
        width: 34%;
        height: 150%;
        background: linear-gradient(100deg, transparent, rgba(255,255,255,.08), rgba(103,232,249,.10), transparent);
        transform: rotate(14deg);
        filter: blur(8px);
        animation: beamSweep 7s ease-in-out infinite;
    }

    @keyframes beamSweep {
        0%, 18% { transform: translateX(0) rotate(14deg); opacity: 0; }
        34% { opacity: 1; }
        65%, 100% { transform: translateX(440%) rotate(14deg); opacity: 0; }
    }

    .orb-wrap {
        position: absolute;
        right: 32px;
        top: 36px;
        width: 405px;
        height: 405px;
        transform-style: preserve-3d;
        animation: orbitStage 8s ease-in-out infinite;
        filter: drop-shadow(0 46px 80px rgba(76,29,149,.38));
    }

    @keyframes orbitStage {
        0%,100% { transform: rotateX(9deg) rotateY(-16deg) translateY(0); }
        50% { transform: rotateX(15deg) rotateY(-6deg) translateY(-18px); }
    }

    .orb {
        position: absolute;
        inset: 58px;
        border-radius: 50%;
        background:
            radial-gradient(circle at 33% 27%, rgba(255,255,255,.98) 0 3%, rgba(255,255,255,.30) 7%, transparent 17%),
            radial-gradient(circle at 30% 28%, #ddd6fe 0%, #8b5cf6 25%, #4c1d95 50%, #111827 76%, #02040a 100%);
        box-shadow:
            inset -36px -42px 74px rgba(0,0,0,.65),
            inset 28px 23px 58px rgba(255,255,255,.11),
            0 0 72px rgba(139,92,246,.42),
            0 0 150px rgba(34,211,238,.08);
        animation: spherePulse 5.5s ease-in-out infinite, sphereHue 14s linear infinite;
    }

    .orb::after {
        content: "";
        position: absolute;
        inset: 12%;
        border-radius: 50%;
        border-top: 1px solid rgba(255,255,255,.17);
        transform: rotate(-18deg);
        filter: blur(.2px);
    }

    @keyframes spherePulse {
        0%,100% { transform: scale(1) rotate(-2deg); }
        50% { transform: scale(1.045) rotate(3deg); }
    }

    @keyframes sphereHue {
        0%,100% { filter: hue-rotate(0deg) saturate(1); }
        50% { filter: hue-rotate(28deg) saturate(1.18); }
    }

    .ring {
        position: absolute;
        inset: 4px;
        border-radius: 50%;
        border: 2px solid rgba(34,211,238,.26);
        box-shadow: 0 0 45px rgba(34,211,238,.17), inset 0 0 30px rgba(34,211,238,.05);
        animation: ringOne 10s linear infinite;
    }

    .ring.r2 {
        inset: 46px -24px;
        border-color: rgba(236,72,153,.23);
        box-shadow: 0 0 46px rgba(236,72,153,.12);
        animation: ringTwo 14s linear infinite reverse;
    }

    .ring.r3 {
        inset: 74px -48px;
        border-color: rgba(167,139,250,.15);
        animation: ringThree 18s linear infinite;
    }

    @keyframes ringOne { to { transform: rotateX(68deg) rotateZ(360deg); } }
    @keyframes ringTwo { to { transform: rotateX(74deg) rotateY(28deg) rotateZ(360deg); } }
    @keyframes ringThree { to { transform: rotateX(62deg) rotateY(-22deg) rotateZ(-360deg); } }

    .satellite {
        position: absolute;
        width: 12px;
        height: 12px;
        left: 45px;
        top: 195px;
        border-radius: 50%;
        background: #67e8f9;
        box-shadow: 0 0 18px #22d3ee, 0 0 34px #22d3ee88;
        animation: satelliteMove 5s ease-in-out infinite alternate;
    }

    @keyframes satelliteMove {
        from { transform: translate3d(0, -35px, 40px) scale(.8); }
        to { transform: translate3d(290px, 74px, -30px) scale(1.2); }
    }

    .hero-copy-wrap {
        position: relative;
        z-index: 3;
        width: min(690px, 65%);
        transform: translateZ(42px);
    }

    .eyebrow {
        display: inline-flex;
        padding: 9px 13px;
        border: 1px solid rgba(167,139,250,.25);
        border-radius: 999px;
        background: rgba(139,92,246,.10);
        backdrop-filter: blur(12px);
        color: #d8b4fe;
        font-size: .74rem;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
        box-shadow: inset 0 1px rgba(255,255,255,.06), 0 0 24px rgba(139,92,246,.08);
        animation: badgeGlow 3.8s ease-in-out infinite;
    }

    @keyframes badgeGlow {
        50% { box-shadow: inset 0 1px rgba(255,255,255,.09), 0 0 32px rgba(139,92,246,.18); }
    }

    .hero-title {
        margin: 24px 0 0;
        font-size: clamp(4rem, 8vw, 7.25rem);
        line-height: .86;
        font-weight: 950;
        letter-spacing: -.068em;
    }

    .hero-title span {
        background: linear-gradient(90deg, #ffffff 0%, #d8b4fe 25%, #67e8f9 54%, #f0abfc 78%, #ffffff 100%);
        background-size: 260% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleShimmer 7s linear infinite;
        filter: drop-shadow(0 14px 30px rgba(0,0,0,.38));
    }

    @keyframes titleShimmer { to { background-position: 260% center; } }

    .hero-desc {
        max-width: 635px;
        margin-top: 26px;
        color: #bbc5d3;
        font-size: 1.06rem;
        line-height: 1.72;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 30px;
    }

    .hero-badge {
        position: relative;
        overflow: hidden;
        padding: 10px 13px;
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 13px;
        background: linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.025));
        color: #d8dee9;
        font-size: .82rem;
        backdrop-filter: blur(15px);
        animation: badgeFloat 5s ease-in-out infinite;
    }

    .hero-badge:nth-child(2) { animation-delay: -1.4s; }
    .hero-badge:nth-child(3) { animation-delay: -2.8s; }
    @keyframes badgeFloat { 50% { transform: translateY(-5px); } }

    [data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        padding: 18px 20px;
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 20px;
        background: linear-gradient(145deg, rgba(20,25,36,.76), rgba(8,11,17,.65));
        backdrop-filter: blur(18px);
        box-shadow: 0 16px 42px rgba(0,0,0,.20), inset 0 1px rgba(255,255,255,.06);
        animation: metricFloat 7s ease-in-out infinite;
    }

    div[data-testid="column"]:nth-child(2) [data-testid="stMetric"] { animation-delay: -1.2s; }
    div[data-testid="column"]:nth-child(3) [data-testid="stMetric"] { animation-delay: -2.4s; }
    div[data-testid="column"]:nth-child(4) [data-testid="stMetric"] { animation-delay: -3.6s; }

    @keyframes metricFloat { 50% { transform: translateY(-6px); box-shadow: 0 24px 52px rgba(0,0,0,.27), 0 0 28px rgba(139,92,246,.05); } }

    [data-testid="stMetric"]::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(110deg, transparent 28%, rgba(255,255,255,.055) 45%, transparent 62%);
        transform: translateX(-120%);
        animation: metricShine 8s ease-in-out infinite;
        pointer-events: none;
    }

    @keyframes metricShine { 0%,60% { transform: translateX(-120%); } 80%,100% { transform: translateX(120%); } }

    [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 850; }

    .section-heading {
        margin-top: 14px;
        font-size: 2.15rem;
        line-height: 1;
        font-weight: 900;
        letter-spacing: -.04em;
    }

    .section-sub { color: var(--muted); margin: 8px 0 22px; }

    .cinema-card {
        position: relative;
        isolation: isolate;
        overflow: hidden;
        min-height: 270px;
        margin-top: 14px;
        padding: 26px;
        border: 1px solid rgba(255,255,255,.105);
        border-radius: 25px;
        background: linear-gradient(145deg, rgba(18,23,34,.91), rgba(8,11,17,.88));
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 48px rgba(0,0,0,.30), inset 0 1px rgba(255,255,255,.06);
        transform: perspective(1000px) rotateX(0) rotateY(0) translateZ(0);
        transition: transform .4s cubic-bezier(.2,.8,.2,1), border-color .35s ease, box-shadow .35s ease;
        animation: cardReveal .7s cubic-bezier(.2,.8,.2,1) both;
    }

    @keyframes cardReveal {
        from { opacity: 0; transform: perspective(1000px) translateY(24px) scale(.98); }
        to { opacity: 1; transform: perspective(1000px) translateY(0) scale(1); }
    }

    .cinema-card:hover {
        transform: perspective(1000px) rotateX(3deg) rotateY(-3.5deg) translateY(-8px) translateZ(24px) scale(1.012);
        border-color: rgba(167,139,250,.40);
        box-shadow: 0 34px 80px rgba(0,0,0,.46), 0 0 50px rgba(139,92,246,.10), inset 0 1px rgba(255,255,255,.10);
    }

    .card-aurora {
        position: absolute;
        width: 260px;
        height: 260px;
        right: -105px;
        top: -120px;
        z-index: -1;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(139,92,246,.35), rgba(34,211,238,.08) 45%, transparent 70%);
        filter: blur(10px);
        animation: cardOrb 6s ease-in-out infinite alternate;
    }

    @keyframes cardOrb { to { transform: translate(-35px, 38px) scale(1.18); filter: blur(17px) hue-rotate(35deg); } }

    .card-scan {
        position: absolute;
        top: -30%;
        left: -65%;
        width: 38%;
        height: 170%;
        background: linear-gradient(105deg, transparent, rgba(255,255,255,.09), rgba(103,232,249,.08), transparent);
        transform: rotate(12deg);
        animation: cardScan 7s ease-in-out infinite;
        pointer-events: none;
    }

    @keyframes cardScan {
        0%,55% { transform: translateX(0) rotate(12deg); opacity: 0; }
        65% { opacity: .8; }
        90%,100% { transform: translateX(470%) rotate(12deg); opacity: 0; }
    }

    .card-topline { display:flex; justify-content:space-between; align-items:center; gap:12px; }
    .match-chip,.rating-chip { border-radius:999px; padding:7px 10px; font-size:.72rem; font-weight:850; letter-spacing:.04em; }
    .match-chip { color:#ddd6fe; background:rgba(139,92,246,.14); border:1px solid rgba(167,139,250,.25); box-shadow:0 0 22px rgba(139,92,246,.08); }
    .rating-chip { color:#fde68a; background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.16); }
    .movie-title { margin-top:26px; color:#fff; font-size:1.38rem; line-height:1.16; font-weight:850; letter-spacing:-.028em; }
    .movie-meta,.movie-foot { color:#8e9bae; font-size:.82rem; }
    .movie-meta { margin-top:7px; }
    .movie-foot { margin-top:17px; }
    .movie-reason { margin-top:20px; color:#ccd5e1; font-size:.95rem; line-height:1.56; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 7px;
        padding: 7px;
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 17px;
        background: rgba(255,255,255,.026);
        backdrop-filter: blur(15px);
        box-shadow: inset 0 1px rgba(255,255,255,.04);
    }

    .stTabs [data-baseweb="tab"] { height:43px; border-radius:11px; padding:0 17px; transition:.25s ease; }
    .stTabs [data-baseweb="tab"]:hover { background:rgba(255,255,255,.045); transform:translateY(-1px); }
    .stTabs [aria-selected="true"] { background:linear-gradient(135deg, rgba(139,92,246,.22), rgba(34,211,238,.09)); box-shadow:0 8px 22px rgba(139,92,246,.08); }

    div.stButton > button, div.stLinkButton > a {
        min-height:43px;
        border-radius:13px !important;
        border:1px solid rgba(255,255,255,.10) !important;
        background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.025)) !important;
        color:#eef2f7 !important;
        font-weight:750 !important;
        box-shadow:inset 0 1px rgba(255,255,255,.06), 0 8px 20px rgba(0,0,0,.17);
        transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease !important;
    }

    div.stButton > button:hover, div.stLinkButton > a:hover {
        transform:translateY(-3px) scale(1.012);
        border-color:rgba(167,139,250,.38) !important;
        background:linear-gradient(135deg, rgba(139,92,246,.16), rgba(34,211,238,.075)) !important;
        box-shadow:0 14px 28px rgba(0,0,0,.28), 0 0 24px rgba(139,92,246,.07) !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div {
        min-height:49px;
        border-radius:14px !important;
        background:rgba(13,17,25,.88) !important;
        border-color:rgba(255,255,255,.095) !important;
        backdrop-filter:blur(18px);
        transition:.25s ease;
    }

    [data-testid="stTextInput"] input:focus {
        border-color:rgba(139,92,246,.55) !important;
        box-shadow:0 0 0 3px rgba(139,92,246,.09), 0 0 34px rgba(139,92,246,.07) !important;
    }

    footer { visibility:hidden; }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration:.001ms !important; animation-iteration-count:1 !important; transition-duration:.001ms !important; }
    }

    @media (max-width: 900px) {
        .block-container { padding-top:4.9rem !important; }
        .hero-shell { min-height:680px; padding:40px 28px; }
        .hero-copy-wrap { width:100%; }
        .hero-title { font-size:clamp(3.6rem, 18vw, 5.7rem); }
        .orb-wrap { width:300px; height:300px; right:-25px; top:390px; opacity:.78; }
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
        <div class="holo-beam"></div>
        <div class="orb-wrap">
            <div class="ring"></div>
            <div class="ring r2"></div>
            <div class="ring r3"></div>
            <div class="satellite"></div>
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

st.write("")
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
discover_tab, similar_tab, genre_tab, saved_tab = st.tabs(["Discover", "Similar movies", "Browse genres", "Saved"])

with discover_tab:
    st.markdown('<div class="section-heading">What are you in the mood for?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Try “funny 90s movie”, “dark sci-fi”, “family adventure”, or “recent romance”.</div>', unsafe_allow_html=True)
    q1, q2 = st.columns([2.1, 1], gap="large")
    with q1:
        query = st.text_input("Describe your movie", placeholder="Try: a clever sci-fi movie from the 90s", label_visibility="collapsed", key="discovery_query")
    with q2:
        selected_genres = st.multiselect("Genres", engine.available_genres, placeholder="Optional genres", label_visibility="collapsed")
    results = engine.discover(query, selected_genres=selected_genres, limit=8)
    if not results:
        st.info("No close matches found. Try a broader description or remove a genre filter.")
    else:
        render_grid(results, "discover")

with similar_tab:
    st.markdown('<div class="section-heading">Start with a movie you already love.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Search a title and get recommendations based on audience behavior and content similarity.</div>', unsafe_allow_html=True)
    search = st.text_input("Search title", placeholder="Type a movie title", label_visibility="collapsed", key="similar_search")
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
