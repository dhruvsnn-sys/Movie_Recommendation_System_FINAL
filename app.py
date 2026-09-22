import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from movie_recommender import (
    load_data,
    data_quality_report,
    build_user_item_matrix,
    popular_movies,
)


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)

st.title("Movie Recommendation System")
st.caption("GYEST305 - Collaborative Filtering | Batch 3")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_project():
    ratings_raw, movies_raw = load_data()

    ratings, movies, quality = data_quality_report(
        ratings_raw, movies_raw
    )

    matrix = build_user_item_matrix(ratings)

    return ratings, movies, quality, matrix


ratings, movies, quality, matrix = load_project()


# ---------------------------------------------------------
# LIGHTWEIGHT ITEM NEAREST-NEIGHBOUR MODEL
# ---------------------------------------------------------
@st.cache_resource
def build_neighbour_model(matrix):
    """
    Build a lightweight nearest-neighbour model.

    Instead of calculating a complete movie x movie
    similarity matrix, this searches only for the
    movies we actually need.
    """

    item_matrix = np.nan_to_num(
        matrix.T.to_numpy(dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    model.fit(item_matrix)

    return model


@st.cache_data
def get_movie_lookup(movies):
    lookup = movies.copy()
    lookup["title_lower"] = lookup["title"].str.lower()
    return lookup


neighbour_model = build_neighbour_model(matrix)
movie_lookup = get_movie_lookup(movies)

movie_ids = list(matrix.columns)
movie_id_to_index = {
    movie_id: i for i, movie_id in enumerate(movie_ids)
}


# ---------------------------------------------------------
# SIMILAR MOVIES
# ---------------------------------------------------------
def find_similar_movies(title, n=10):

    title_clean = title.strip().lower()

    matches = movie_lookup[
        movie_lookup["title_lower"].str.contains(
            title_clean,
            na=False,
            regex=False,
        )
    ]

    if matches.empty:
        return pd.DataFrame()

    selected_movie_id = matches.iloc[0]["movieId"]

    if selected_movie_id not in movie_id_to_index:
        return pd.DataFrame()

    movie_index = movie_id_to_index[selected_movie_id]

    count = min(n + 1, len(movie_ids))

    distances, indices = neighbour_model.kneighbors(
        np.nan_to_num(
            matrix.T.iloc[[movie_index]].to_numpy(dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ),
        n_neighbors=count,
    )

    rows = []

    for distance, index in zip(
        distances[0][1:],
        indices[0][1:],
    ):
        similar_movie_id = movie_ids[index]

        movie_row = movies[
            movies["movieId"] == similar_movie_id
        ]

        if movie_row.empty:
            continue

        similarity = 1.0 - float(distance)

        rows.append(
            {
                "title": movie_row.iloc[0]["title"],
                "similarity": round(similarity, 4),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# USER RECOMMENDATIONS
# ---------------------------------------------------------
def recommend_for_user_light(user_id, n=10):

    user_rows = ratings[
        ratings["userId"] == user_id
    ]

    if user_rows.empty:
        return pd.DataFrame()

    rated_movie_ids = set(
        user_rows["movieId"].tolist()
    )

    scores = {}
    weights = {}

    # Use only the user's highest-rated movies as seeds.
    # This keeps the calculation lightweight.
    seed_movies = (
        user_rows
        .sort_values("rating", ascending=False)
        .head(20)
    )

    for _, row in seed_movies.iterrows():

        movie_id = row["movieId"]
        rating = float(row["rating"])

        if movie_id not in movie_id_to_index:
            continue

        movie_index = movie_id_to_index[movie_id]

        count = min(
            21,
            len(movie_ids)
        )

        distances, indices = neighbour_model.kneighbors(
            np.nan_to_num(
                matrix.T.iloc[[movie_index]].to_numpy(dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            ),
            n_neighbors=count,
        )

        for distance, index in zip(
            distances[0][1:],
            indices[0][1:],
        ):

            candidate_id = movie_ids[index]

            # Do not recommend something the user
            # has already rated.
            if candidate_id in rated_movie_ids:
                continue

            similarity = 1.0 - float(distance)

            if similarity <= 0:
                continue

            scores[candidate_id] = (
                scores.get(candidate_id, 0)
                + similarity * rating
            )

            weights[candidate_id] = (
                weights.get(candidate_id, 0)
                + similarity
            )

    if not scores:
        return pd.DataFrame()

    results = []

    for movie_id, score in scores.items():

        predicted_rating = (
            score / weights[movie_id]
        )

        movie_row = movies[
            movies["movieId"] == movie_id
        ]

        if movie_row.empty:
            continue

        results.append(
            {
                "title": movie_row.iloc[0]["title"],
                "predicted_rating": round(
                    min(5.0, predicted_rating),
                    2,
                ),
            }
        )

    result = pd.DataFrame(results)

    if result.empty:
        return result

    return (
        result
        .sort_values(
            "predicted_rating",
            ascending=False,
        )
        .head(n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Recommendation Options")

mode = st.sidebar.selectbox(
    "Mode",
    [
        "Recommend for User",
        "Find Similar Movies",
        "Popular / Cold Start",
    ],
)


# ---------------------------------------------------------
# MODE 1: USER RECOMMENDATIONS
# ---------------------------------------------------------
if mode == "Recommend for User":

    user_id = st.sidebar.number_input(
        "User ID",
        min_value=1,
        max_value=int(ratings["userId"].max()),
        value=1,
    )

    n = st.sidebar.slider(
        "Number of recommendations",
        5,
        15,
        10,
    )

    if st.button("Generate Recommendations"):

        with st.spinner("Generating recommendations..."):

            result = recommend_for_user_light(
                int(user_id),
                n=n,
            )

        if result.empty:

            st.warning(
                "No recommendations could be generated "
                "for this user."
            )

        else:

            st.subheader(
                f"Recommendations for User {int(user_id)}"
            )

            st.dataframe(
                result,
                use_container_width=True,
            )


# ---------------------------------------------------------
# MODE 2: SIMILAR MOVIES
# ---------------------------------------------------------
elif mode == "Find Similar Movies":

    title = st.sidebar.text_input(
        "Movie title",
        "Toy Story",
    )

    n = st.sidebar.slider(
        "Number of similar movies",
        5,
        15,
        10,
    )

    if st.button("Find Similar Movies"):

        with st.spinner("Finding similar movies..."):

            result = find_similar_movies(
                title,
                n=n,
            )

        if result.empty:

            st.warning(
                "Movie not found or not enough rating data. "
                "Try a movie such as Toy Story."
            )

        else:

            st.subheader(
                f"Movies Similar to {title}"
            )

            st.dataframe(
                result,
                use_container_width=True,
            )


# ---------------------------------------------------------
# MODE 3: POPULAR / COLD START
# ---------------------------------------------------------
else:

    n = st.sidebar.slider(
        "Number of movies",
        5,
        15,
        10,
    )

    st.subheader(
        "Popular Recommendations for a New User"
    )

    st.dataframe(
        popular_movies(
            ratings,
            movies,
            n=n,
        ),
        use_container_width=True,
    )


# ---------------------------------------------------------
# PROJECT INFORMATION
# ---------------------------------------------------------
st.divider()

c1, c2, c3 = st.columns(3)

c1.metric(
    "Users",
    ratings["userId"].nunique(),
)

c2.metric(
    "Movies",
    movies["movieId"].nunique(),
)

c3.metric(
    "Ratings",
    f"{len(ratings):,}",
)

st.info(
    "Cold-start users have no personal rating history. "
    "The fallback uses movies with sufficient ratings "
    "and strong average ratings."
)
