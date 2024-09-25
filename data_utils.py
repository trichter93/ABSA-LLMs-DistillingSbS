import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
import json
import re
from itertools import chain
import sys
from google.colab import drive
drive.mount('/content/drive')
base = '/content/drive/MyDrive/ABSA-LLMs-DistillingSbS'
sys.path.append(base)
import Review

def load_from_source():
    with open(base +'/data/absa_datasets/absa_train.json', 'r') as f:
        review_list_train = json.load(f)
    with open(base +'/data/absa_datasets/absa_valid.json', 'r') as f:
        review_list_valid = json.load(f)
    with open(base +'/data/absa_datasets/absa_test.json', 'r') as f:
        review_list_test = json.load(f)
    for review in list(chain(review_list_train, review_list_valid, review_list_test)):
        review['ABSALabels'] = json.dumps(review['ABSALabels'])
        review['ABSARationales'] = json.dumps(review['ABSARationales'])
    train_dataset = Dataset.from_list(review_list_train)
    valid_dataset = Dataset.from_list(review_list_valid)
    test_dataset = Dataset.from_list(review_list_test)
    return train_dataset, valid_dataset, test_dataset


def show_hist(reviews):
    plt.figure(figsize=(10,6))

    bins = [10 * i for i in range(0, 10)]

    n, bins, patches = plt.hist(reviews["tokenCount"].values, bins=bins, color='blue', edgecolor='black', rwidth=0.8)

    plt.xlim([0, 100])
    plt.xlabel("Token Count")
    plt.ylabel("Review Count")

    for i in range(len(bins) - 1):
        lower_bound, upper_bound = bins[i], bins[i + 1]
        bin_reviews = reviews[(reviews["tokenCount"] >= lower_bound) & (reviews["tokenCount"] < upper_bound)]
        sample_reviews = bin_reviews.sample(min(10, len(bin_reviews)))

    plt.show()


def show_category_breakdown(reviews):
    category_breakdown = reviews["appCategory"].value_counts()
    print(category_breakdown)
    print(category_breakdown.sum())


def show_breakdown_per_category_and_app(reviews):
    def show_breakdown_per_category(reviews, category : str):
        category_reviews = reviews[reviews['appCategory'] == category]

        # Count the number of reviews per app in the "Reference" category
        app_counts = category_reviews['appName'].value_counts()

        # Display the top 10 apps with the most reviews
        top_10_apps = app_counts.head(10)
        print(top_10_apps)
    unique_categories = reviews['appCategory'].unique().tolist()
    for category in unique_categories:
      print(f"Category: {category}")
      show_breakdown_per_category(reviews, category)
      print("-----------")
      
redundant_characters = ['!', '?', '.', ',', "'", '"', '-', '_', '(', ')', ':', ';', '*', '+', '=', '~', '^', '{', '}', '[', ']', '&', '%', '@']
def reduce_redundant_chars(text, redundant_chars=redundant_characters):
    # Replace consecutive occurrences of redundant characters with a single occurrence
    for char in redundant_chars:
        text = re.sub(f'{re.escape(char)}+', char, text)
    text = re.sub(r'\?!\?+', '?!', text)
    text = re.sub(r'!\?!+', '?!', text)
    # Reduce certain combinations (e.g., '?!?')
    text = re.sub(r'(\?!)+', '?!', text)
    text = re.sub(r'(!\?)+', '?!', text)

    return text
def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def insert_newlines(input_str, max_line_length):
    lines = []
    words = input_str.split()

    current_line = ""
    for word in words:
        if len(current_line) + len(word) <= max_line_length:
            current_line += word + " "
        else:
            lines.append(current_line.strip())
            current_line = word + " "

    # Add the last line
    lines.append(current_line.strip())

    return '\n'.join(lines)

def print_review_nicely(review : Review):
    result= f"""App Name : {review.app_name}
Title: {review.title}
Body: {insert_newlines(review.body, 100)}
App Category: {review.app_category}
User Id: {review.user_review_id}
          """
    if review.aspects_and_scores:
        result += f"""Aspects and Scores: {review.aspects_and_scores}"""
    print(result)

def print_review_row_nicely(review_row : pd.Series, labels_column = "ABSALabels", rationales_column = "ABSARationales", labels_column2 = "best_24_05_07_t5", labels_column3 = "distilled_24_18_06"):
    result = f"""App Name : {review_row['appName']}
Title: {review_row['title']}
Body: {insert_newlines(review_row['body'], 100)}
App Category: {review_row['appCategory']}
User Id: {review_row['userReviewId']}

          """
    print(result)
    if labels_column in review_row.index:
        print(f"""Aspects and Sentiments for {labels_column}: {json.dumps(json.loads(review_row[labels_column]), indent=2)}""")
    if rationales_column is not None and rationales_column in review_row.index:
        print(f"""Aspects and Rationales: {json.dumps(json.loads(review_row[rationales_column]), indent=2)}""")
    if labels_column2 in review_row.index:
        print(f"""Aspects and Sentiments for {labels_column2}: {json.dumps(json.loads(review_row[labels_column2]), indent=2)}""")
    if labels_column3 in review_row.index:
        print(f"""Aspects and Sentiments for {labels_column3}: {json.dumps(json.loads(review_row[labels_column3]), indent=2)}""")