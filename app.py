import streamlit as st
import pandas as pd

from movie_recommender import (
    load_data, data_quality_report, build_user_item_matrix,
    build_item_similarity, recommend_for_user, similar_movies,
    popular_movies
)

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")
st.title("Movie Recommendation System")
st.caption("GYEST305 — Collaborative Filtering | Batch 3")

@st.cache_data
def load_project():
    ratings_raw, movies_raw = load_data()
    ratings, movies, quality = data_quality_report(ratings_raw, movies_raw)
    matrix = build_user_item_matrix(ratings)
    sim = build_item_similarity(matrix)
    return ratings, movies, quality, matrix, sim

ratings, movies, quality, matrix, sim = load_project()

st.sidebar.header("Recommendation Options")
mode = st.sidebar.selectbox(
    "Mode",
    ["Recommend for User", "Find Similar Movies", "Popular / Cold Start"]
)

if mode == "Recommend for User":
    user_id = st.sidebar.number_input(
        "User ID", min_value=1, max_value=int(ratings.userId.max()), value=1
    )
    n = st.sidebar.slider("Number of recommendations", 5, 15, 10)

    if st.button("Generate Recommendations"):
        result = recommend_for_user(
            int(user_id), ratings, movies, matrix, sim, n=n
        )
        st.subheader(f"Recommendations for User {int(user_id)}")
        st.dataframe(result, use_container_width=True)

elif mode == "Find Similar Movies":
    title = st.sidebar.text_input("Movie title", "Toy Story")
    n = st.sidebar.slider("Number of similar movies", 5, 15, 10)

    if st.button("Find Similar Movies"):
        result = similar_movies(title, movies, sim, n=n)
        if result.empty:
            st.warning("Movie not found. Try a title such as Toy Story.")
        else:
            st.subheader(f"Movies Similar to {title}")
            st.dataframe(result, use_container_width=True)

else:
    n = st.sidebar.slider("Number of movies", 5, 15, 10)
    st.subheader("Popular Recommendations for a New User")
    st.dataframe(
        popular_movies(ratings, movies, n=n),
        use_container_width=True
    )

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Users", ratings.userId.nunique())
c2.metric("Movies", movies.movieId.nunique())
c3.metric("Ratings", f"{len(ratings):,}")

st.info(
    "Cold-start users have no personal rating history. "
    "The fallback uses movies with sufficient ratings and strong average ratings."
)
