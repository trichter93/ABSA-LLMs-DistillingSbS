import pandas as pd
import requests
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
def create_review_dataframe(review_path, apps_info_path, skiprows=0, nrows=100000):
    if nrows == None:
        reviews = pd.read_csv(review_path, skiprows=range(1, skiprows + 1))
    else:
        reviews = pd.read_csv(review_path, skiprows=range(1, skiprows + 1), nrows=nrows)
    reviews['titleBody'] = reviews['title'].fillna('') + ' ' + reviews['body'].fillna('')
    apps_info_df = pd.read_csv(apps_info_path)
    def get_app_info(app_id, apps_info_df=apps_info_df):
        url = f"https://itunes.apple.com/lookup?id={app_id}"
        response = requests.get(url)
        data = response.json()
        data2 = apps_info_df[apps_info_df["id"] == app_id]
        # Check if the API response is successful
        if data["resultCount"] > 0:
            app_info = {
                "app_name": data["results"][0]["trackName"],
                "app_category": data["results"][0]["primaryGenreName"]
            }
            return app_info
        elif not data2.empty:
            app_info = {
                "app_name": data2["track_name"].values[0],
                "app_category": data2["prime_genre"].values[0]
            }
            return app_info
        return None

    def get_app_info_cached(app_id, app_info_cache={}):
        if str(app_id) in app_info_cache:
            return app_info_cache[str(app_id)]
        else:
            app_info = get_app_info(app_id)
            app_info_cache[str(app_id)] = app_info
            return app_info
        
    app_info = reviews['appId'].apply(get_app_info_cached)
    reviews['appName'] = app_info.apply(lambda info: info["app_name"] if info else None)
    reviews['appCategory'] = app_info.apply(lambda info: info["app_category"] if info else None)
    reviews['tokenCount'] = reviews['titleBody'].apply(lambda x: len(word_tokenize(x)))
    return reviews


def sorted_reviews_token(reviews, ascending=False):
    # Sort the DataFrame based on token counts in descending order
    sorted_reviews = reviews.sort_values(by='tokenCount', ascending=ascending)
    return sorted_reviews


def create_top_N_entries(sorted_reviews, N1=0, N2=100):
    top_N_entries = sorted_reviews.iloc[N1:N2+1].copy()
    top_N_entries.reset_index(drop=True, inplace=True)
    
def extract_reviews(df, min_token_requirements={"100" : 40, "1000" : 50, "10000" : 60, "100000" : 60, "500000" : 70}, max_reviews_per_app=200, max_reviews_per_category=2000, max_tokens_per_review = 300):
    sorted_df = df.sort_values(by=['numberOfReviews','appName', 'tokenCount', 'voteSum'], ascending=[True, False, False, False])
    sorted_df = sorted_df.dropna(subset=['numberOfReviews'])
    selected_reviews_per_category = {}
    for index, row in sorted_df.iterrows():
        nr_reviews = int(row['numberOfReviews'])
        app = row['appName']
        category = row['appCategory']
        token_count = row['tokenCount']
        vote_sum = row['voteSum']
        if category not in selected_reviews_per_category:
            selected_reviews_per_category[category] = {}
        if app not in selected_reviews_per_category[category]:
            selected_reviews_per_category[category][app] = []
        if len(selected_reviews_per_category[category]) < max_reviews_per_category and len(selected_reviews_per_category[category][app]) < max_reviews_per_app:
            for upper_bound, min_tokens in min_token_requirements.items() :
                if ((int(nr_reviews) <= int(upper_bound) and token_count >= min_tokens) or int(vote_sum) > 50) and token_count <= max_tokens_per_review:
                    selected_reviews_per_category[category][app].append(row)
                    break
    final_selected_reviews = pd.concat([pd.DataFrame(selected_reviews) for category_reviews in selected_reviews_per_category.values()
                                        for selected_reviews in category_reviews.values()],
                                       ignore_index=True)

    return final_selected_reviews