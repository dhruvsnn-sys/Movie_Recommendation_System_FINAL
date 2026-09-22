# FINAL GYEST305 MINI PROJECT
## Movie Recommendation System — Batch 3

This package is structured to follow the GYEST305 Mini Project Guide.

### Team
Anush M  
Dhruv S Nambiar  
Joel C M  
Joseph Christy VS  
Madhav K Bhattathiri

## 1. Quick start

Install:

```bash
pip install -r requirements.txt
```

Run the complete analysis:

```bash
python movie_recommender.py
```

Then build the final report and presentation:

```bash
python build_submission.py
```

The first command downloads the MovieLens latest-small dataset automatically.

## 2. Interactive application

```bash
streamlit run app.py
```

The app provides:
- user recommendations
- similar-movie search
- cold-start/popular recommendations

## 3. Project workflow

The implementation follows:

Data → Clean → Explore → Prepare → Model → Evaluate → Interpret → Present

### Data quality
The program explicitly checks:
- missing values
- duplicates
- data types
- invalid rating values
- row counts before/after cleaning

### EDA
It generates:
1. Rating distribution
2. Number of ratings per user
3. Number of ratings per movie
4. User-item matrix heatmap
5. Similarity visualization for Toy Story

### Model
Item-item collaborative filtering with cosine similarity.

### Evaluation
The latest rating of each user is held out when possible. The similarity model is built using only the training ratings.

Metrics:
- MAE
- RMSE
- Hit Rate@10
- Precision@10

## 4. Generated files

After running the commands:

```text
outputs/
├── data_quality.json
├── evaluation_results.json
├── project_results.json
├── 01_rating_distribution.png
├── 02_ratings_per_user.png
├── 03_ratings_per_movie.png
├── 04_user_item_heatmap.png
└── 05_similarity_toy_story.png

Final_Project_Report.docx
Final_Project_Presentation.pptx
Final_Project_Results.md
```

## 5. Important academic point

Do not present the recommender as a black box. Be able to explain:
- what the user-item matrix means
- why it is sparse
- how cosine similarity is calculated
- how a predicted rating is obtained
- how recommendations are ranked
- why the train/test split avoids direct leakage
- what MAE and RMSE mean
- what Hit Rate@10 and Precision@10 mean
- what cold start means

## 6. Dataset reference

GroupLens Research — MovieLens latest-small:
https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html

Dataset citation:
Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), 19:1–19:19.
