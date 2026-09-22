"""
GYEST305 Mini Project — Movie Recommendation System
Batch 3: Collaborative Filtering

This program:
1. Downloads MovieLens latest-small automatically.
2. Checks data quality: shape, dtypes, missing values, duplicates, ranges.
3. Performs EDA and creates 5 required/recommended visualizations.
4. Builds a user-item matrix.
5. Uses item-item cosine similarity for collaborative filtering.
6. Evaluates rating prediction with MAE and RMSE using a per-user temporal holdout.
7. Evaluates top-10 recommendation quality with Hit Rate@10 and Precision@10.
8. Demonstrates recommendations and similar movies.
9. Writes a machine-readable results summary for the report/presentation.

Run:
    python movie_recommender.py
"""

from pathlib import Path
import io
import json
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "ml-latest-small"
OUTPUT_DIR = BASE_DIR / "outputs"


def download_dataset():
    ratings_file = DATASET_DIR / "ratings.csv"
    movies_file = DATASET_DIR / "movies.csv"

    if ratings_file.exists() and movies_file.exists():
        return

    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "ml-latest-small.zip"
    print("Downloading MovieLens latest-small dataset...")
    if not zip_path.exists():
        urllib.request.urlretrieve(DATA_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)


def load_data():
    download_dataset()
    ratings = pd.read_csv(DATASET_DIR / "ratings.csv")
    movies = pd.read_csv(DATASET_DIR / "movies.csv")
    return ratings, movies


def data_quality_report(ratings, movies):
    OUTPUT_DIR.mkdir(exist_ok=True)

    report = {
        "ratings_rows": int(len(ratings)),
        "ratings_columns": int(len(ratings.columns)),
        "movies_rows": int(len(movies)),
        "movies_columns": int(len(movies.columns)),
        "unique_users": int(ratings["userId"].nunique()),
        "unique_movies_rated": int(ratings["movieId"].nunique()),
        "movie_catalog_size": int(movies["movieId"].nunique()),
        "ratings_missing_cells": int(ratings.isna().sum().sum()),
        "movies_missing_cells": int(movies.isna().sum().sum()),
        "ratings_duplicate_rows": int(ratings.duplicated().sum()),
        "movies_duplicate_rows": int(movies.duplicated().sum()),
        "rating_min": float(ratings["rating"].min()),
        "rating_max": float(ratings["rating"].max()),
        "rating_mean": float(ratings["rating"].mean()),
    }

    # Type and uniqueness details
    report["ratings_dtypes"] = {k: str(v) for k, v in ratings.dtypes.items()}
    report["movies_dtypes"] = {k: str(v) for k, v in movies.dtypes.items()}

    # Basic validation / cleaning
    valid_rating = ratings["rating"].between(0.5, 5.0)
    report["invalid_rating_rows"] = int((~valid_rating).sum())

    # The supplied MovieLens dataset is expected to be clean for these fields.
    # We still perform explicit checks as required by the project guide.
    ratings_clean = ratings.dropna(
        subset=["userId", "movieId", "rating"]).copy()
    movies_clean = movies.dropna(subset=["movieId", "title"]).copy()
    ratings_clean = ratings_clean.drop_duplicates()
    movies_clean = movies_clean.drop_duplicates()
    ratings_clean = ratings_clean[ratings_clean["rating"].between(0.5, 5.0)]

    report["rows_after_cleaning"] = int(len(ratings_clean))
    report["rows_removed_by_cleaning"] = int(len(ratings) - len(ratings_clean))

    with open(OUTPUT_DIR / "data_quality.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return ratings_clean, movies_clean, report


def build_user_item_matrix(ratings):
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean",
    )


def build_item_similarity(train_matrix):
    # Columns are movies. Missing ratings are represented as zero only for
    # similarity calculation; actual ratings remain unchanged in the matrix.
    item_user = train_matrix.T.fillna(0.0)
    sim = cosine_similarity(item_user)
    return pd.DataFrame(sim, index=item_user.index, columns=item_user.index)


def weighted_predict(user_ratings, candidate_movie_id, sim_df, default_rating):
    """Predict one rating using similarity-weighted item-item CF."""
    if candidate_movie_id not in sim_df.index:
        return float(default_rating)

    numerator = 0.0
    denominator = 0.0

    for rated_movie_id, rating in user_ratings.items():
        if rated_movie_id not in sim_df.index:
            continue
        sim = float(sim_df.at[candidate_movie_id, rated_movie_id])
        if sim > 0:
            numerator += sim * float(rating)
            denominator += sim

    if denominator == 0:
        return float(default_rating)

    return float(np.clip(numerator / denominator, 0.5, 5.0))


def temporal_train_test_split(ratings):
    """
    Hold out each user's latest rating when the user has at least two ratings.
    This avoids using the held-out interaction when constructing similarity.
    """
    r = ratings.sort_values(["userId", "timestamp"]).copy()
    test_idx = r.groupby("userId", sort=False).tail(1).index
    test = r.loc[test_idx].copy()
    train = r.drop(test_idx).copy()
    return train, test


def evaluate_model(ratings):
    train, test = temporal_train_test_split(ratings)
    train_matrix = build_user_item_matrix(train)
    sim_df = build_item_similarity(train_matrix)

    global_mean = float(train["rating"].mean())
    predictions = []

    for row in test.itertuples(index=False):
        if row.userId not in train_matrix.index:
            pred = global_mean
        else:
            user_ratings = train_matrix.loc[row.userId].dropna()
            # User mean is used only as fallback when there is no similarity.
            fallback = float(user_ratings.mean()) if len(
                user_ratings) else global_mean
            pred = weighted_predict(
                user_ratings, row.movieId, sim_df, fallback)

        predictions.append(pred)

    y_true = test["rating"].to_numpy(dtype=float)
    y_pred = np.array(predictions, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Top-10 recommendation evaluation.
    # A held-out item is considered relevant when its held-out rating >= 4.
    hits = 0
    relevant_users = 0
    precision_values = []

    train_matrix_full = train_matrix
    for row in test.itertuples(index=False):
        if row.rating < 4.0:
            continue
        relevant_users += 1

        if row.userId not in train_matrix_full.index:
            precision_values.append(0.0)
            continue

        user_ratings = train_matrix_full.loc[row.userId].dropna()
        seen = set(user_ratings.index)

        candidate_ids = [m for m in sim_df.index if m not in seen]
        if not candidate_ids:
            precision_values.append(0.0)
            continue

        scored = []
        for movie_id in candidate_ids:
            score = weighted_predict(
                user_ratings, movie_id, sim_df,
                float(user_ratings.mean())
            )
            scored.append((movie_id, score))

        top10 = [m for m, _ in sorted(
            scored, key=lambda x: x[1], reverse=True)[:10]]
        hit = int(row.movieId in top10)
        hits += hit
        precision_values.append(hit / 10.0)

    hit_rate = float(hits / relevant_users) if relevant_users else 0.0
    precision_at_10 = float(np.mean(precision_values)
                            ) if precision_values else 0.0

    result = {
        "train_ratings": int(len(train)),
        "test_ratings": int(len(test)),
        "mae": mae,
        "rmse": rmse,
        "relevant_test_users_rating_ge_4": int(relevant_users),
        "hit_rate_at_10": hit_rate,
        "precision_at_10": precision_at_10,
        "evaluation_method": "Per-user latest-rating temporal holdout; item-item cosine similarity on training ratings",
    }

    with open(OUTPUT_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result, train_matrix, sim_df


def recommend_for_user(user_id, ratings, movies, matrix, sim_df, n=10):
    if user_id not in matrix.index:
        return popular_movies(ratings, movies, n)

    user_ratings = matrix.loc[user_id].dropna()
    if user_ratings.empty:
        return popular_movies(ratings, movies, n)

    seen = set(user_ratings.index)
    fallback = float(user_ratings.mean())
    scores = []

    for movie_id in sim_df.index:
        if movie_id in seen:
            continue
        score = weighted_predict(user_ratings, movie_id, sim_df, fallback)
        scores.append((movie_id, score))

    result = pd.DataFrame(scores, columns=["movieId", "predicted_rating"])
    result = result.sort_values("predicted_rating", ascending=False).head(n)
    result = result.merge(movies[["movieId", "title", "genres"]], on="movieId")
    return result[["movieId", "title", "genres", "predicted_rating"]]


def popular_movies(ratings, movies, n=10, min_ratings=50):
    stats = ratings.groupby("movieId").agg(
        rating_count=("rating", "count"),
        mean_rating=("rating", "mean")
    ).reset_index()

    stats = stats[stats["rating_count"] >= min_ratings].copy()
    # Simple shrinkage score prevents movies with very few ratings dominating.
    stats["score"] = (
        stats["mean_rating"] * stats["rating_count"]
        / (stats["rating_count"] + 50)
    )

    return (
        stats.sort_values(["score", "rating_count"], ascending=False)
        .head(n)
        .merge(movies[["movieId", "title", "genres"]], on="movieId")
        [["movieId", "title", "genres", "mean_rating", "rating_count"]]
    )


def similar_movies(movie_title, movies, sim_df, n=10):
    matches = movies[movies["title"].str.contains(
        movie_title, case=False, na=False, regex=False
    )]
    if matches.empty:
        return pd.DataFrame()

    movie_id = int(matches.iloc[0]["movieId"])
    if movie_id not in sim_df.index:
        return pd.DataFrame()

    sims = (
        sim_df[movie_id]
        .drop(index=movie_id)
        .sort_values(ascending=False)
        .head(n)
        .rename("similarity")
        .reset_index()
        .rename(columns={"index": "movieId"})
    )

    return sims.merge(
        movies[["movieId", "title", "genres"]], on="movieId"
    )[["movieId", "title", "genres", "similarity"]]


def make_visualizations(ratings, movies, sim_df):
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Rating distribution
    plt.figure(figsize=(8, 5))
    ratings["rating"].value_counts().sort_index().plot(kind="bar")
    plt.title("Distribution of Movie Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Number of Ratings")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_rating_distribution.png", dpi=200)
    plt.close()

    # 2. Ratings per user
    user_counts = ratings.groupby("userId").size()
    plt.figure(figsize=(8, 5))
    plt.hist(user_counts, bins=30)
    plt.title("Distribution of Number of Ratings per User")
    plt.xlabel("Number of Ratings by User")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_ratings_per_user.png", dpi=200)
    plt.close()

    # 3. Ratings per movie
    movie_counts = ratings.groupby(
        "movieId").size().sort_values(ascending=False).head(15)
    plot_data = movie_counts.reset_index(name="rating_count").merge(
        movies[["movieId", "title"]], on="movieId"
    )
    plt.figure(figsize=(10, 6))
    plt.barh(plot_data["title"].str.slice(0, 35)[::-1],
             plot_data["rating_count"][::-1])
    plt.title("Top 15 Movies by Number of Ratings")
    plt.xlabel("Number of Ratings")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_ratings_per_movie.png", dpi=200)
    plt.close()

    # 4. User-item matrix sample
    matrix = build_user_item_matrix(ratings)
    sample = matrix.iloc[:25, :30]
    plt.figure(figsize=(12, 7))
    plt.imshow(sample.fillna(0), aspect="auto")
    plt.title("Sample User-Item Rating Matrix")
    plt.xlabel("Movies")
    plt.ylabel("Users")
    plt.colorbar(label="Rating (0 = not rated)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_user_item_heatmap.png", dpi=200)
    plt.close()

    # 5. Similarity visualization for Toy Story
    sim = similar_movies("Toy Story", movies, sim_df, n=10)
    if not sim.empty:
        plt.figure(figsize=(10, 6))
        plt.barh(sim["title"].str.slice(0, 35)[::-1], sim["similarity"][::-1])
        plt.title("Movies Most Similar to Toy Story")
        plt.xlabel("Cosine Similarity")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "05_similarity_toy_story.png", dpi=200)
        plt.close()


def write_results_summary(ratings, movies, quality, evaluation, recommendations, similar):
    summary = {
        "dataset": {
            "name": "MovieLens latest-small",
            "source": DATA_URL,
            "ratings": int(len(ratings)),
            "users": int(ratings.userId.nunique()),
            "movies_in_catalog": int(movies.movieId.nunique()),
            "movies_with_ratings": int(ratings.movieId.nunique()),
            "rating_scale": "0.5 to 5.0 in 0.5-star increments",
        },
        "quality": quality,
        "evaluation": evaluation,
        "demo_user": int(ratings.userId.value_counts().index[0]),
        "recommendations": recommendations.to_dict(orient="records"),
        "toy_story_similar": similar.to_dict(orient="records"),
    }

    with open(OUTPUT_DIR / "project_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def main():
    print("=" * 70)
    print("GYEST305 — MOVIE RECOMMENDATION SYSTEM")
    print("=" * 70)

    ratings_raw, movies_raw = load_data()
    ratings, movies, quality = data_quality_report(ratings_raw, movies_raw)

    print("\nDATASET")
    print(f"Ratings: {len(ratings):,}")
    print(f"Users: {ratings.userId.nunique():,}")
    print(f"Movies in catalog: {movies.movieId.nunique():,}")
    print(f"Movies with ratings: {ratings.movieId.nunique():,}")

    print("\nDATA QUALITY")
    print(f"Missing rating cells: {quality['ratings_missing_cells']}")
    print(f"Duplicate rating rows: {quality['ratings_duplicate_rows']}")
    print(f"Invalid rating rows: {quality['invalid_rating_rows']}")
    print(
        f"Rows removed during cleaning: {quality['rows_removed_by_cleaning']}")

    print("\nBUILDING MODEL")
    matrix = build_user_item_matrix(ratings)
    sim_df = build_item_similarity(matrix)
    print(f"User-item matrix shape: {matrix.shape}")

    evaluation = {}
    train_matrix = matrix
    train_sim_df = sim_df

    print("\nEVALUATION")
    print(f"MAE: {evaluation['mae']:.4f}")
    print(f"RMSE: {evaluation['rmse']:.4f}")
    print(f"Hit Rate@10: {evaluation['hit_rate_at_10']:.4f}")
    print(f"Precision@10: {evaluation['precision_at_10']:.4f}")

    demo_user = int(ratings.userId.value_counts().index[0])
    recommendations = recommend_for_user(
        demo_user, ratings, movies, matrix, sim_df, n=10
    )
    similar = similar_movies("Toy Story", movies, sim_df, n=10)

    print(f"\nEXAMPLE RECOMMENDATIONS — USER {demo_user}")
    print(recommendations[["title", "predicted_rating"]
                          ].to_string(index=False))

    print("\nEXAMPLE SIMILARITY — TOY STORY")
    print(similar[["title", "similarity"]].to_string(index=False))

    make_visualizations(ratings, movies, sim_df)
    write_results_summary(ratings, movies, quality,
                          evaluation, recommendations, similar)

    print("\nOutputs written to:", OUTPUT_DIR)
    print("Run build_submission.py after this program to create the report and PPT.")


if __name__ == "__main__":
    main()
