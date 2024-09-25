import pandas as pd
import json

def jaccard_score_and_sentiment_agreement(df : pd.DataFrame, ground_truth : str, generated : str):
  """
    Compute Jaccard score and sentiment agreement metrics between ground truth and generated aspect-based sentiment labels.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing aspect-based sentiment labels.
    - ground_truth : str
        Name of the column in `df` containing ground truth aspect-based sentiment labels (JSON format).
    - generated : str
        Name of the column in `df` containing generated aspect-based sentiment labels (JSON format).

    Returns:
    - dict
        Dictionary containing the following metrics averaged across all rows in `df`:
        {
            'jaccard score': float,
                Average Jaccard score computed as the ratio of intersection size to union size of aspect sets.
            'average intersection size': float,
                Average size of intersection (common aspects) between ground truth and generated labels.
            'agreed sentiments ratio': float,
                Ratio of agreed sentiments (same sentiment for common aspects) to total common aspects.
            'sum': float,
                Sum of Jaccard score and agreed sentiments ratio, providing an overall agreement measure.
        }
    """
  jaccard_score = 0
  agreed_aspects = 0
  agreed_sentiments = 0
  intersection_size = 0
  for index, row in df.iterrows():
    aspects_and_scores = json.loads(row[ground_truth])
    generated_aspects_and_scores = json.loads(row[generated])
    union_keys = set(aspects_and_scores.keys()).union(generated_aspects_and_scores.keys())
    intersection_keys = set(aspects_and_scores.keys()).intersection(generated_aspects_and_scores.keys())
    union_keys_list = list(union_keys)
    intersection_keys_list = list(intersection_keys)
    strings_to_delete = ["satisfaction with features"]
    agreed_aspects += len(intersection_keys_list)
    for key in intersection_keys_list:
      if aspects_and_scores[key] == generated_aspects_and_scores[key]:
        agreed_sentiments += 1
    if len(union_keys_list) != 0:
      jaccard_score += len(intersection_keys_list) / len(union_keys_list)
      intersection_size += len(intersection_keys_list)
  return {'jaccard score' : jaccard_score / len(df),'average intersection size' : intersection_size/len(df) ,'agreed sentiments ratio' : agreed_sentiments / agreed_aspects, 'sum' : jaccard_score / len(df)+agreed_sentiments / agreed_aspects}

def correctly_analyzed_user_review_ids(completed_reviews, aspect, model):
    correct = []
    incorrect = []
    for review in completed_reviews:
        if aspect in review.generated_aspects_and_scores[model] and aspect in review.aspects_and_scores:
            correct.append(review)
        elif aspect in review.generated_aspects_and_scores[model] and aspect not in review.aspects_and_scores:
            incorrect.append(review)

    return [correct, incorrect]
    
def precision_aspect(df : pd.DataFrame, column_true : str, column_pred : str, aspect : str) -> float:
  """
    Compute precision for a specific aspect in aspect-based sentiment analysis.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing the true and predicted aspect labels.
    - column_true : str
        Name of the column in `df` containing true aspect labels (JSON format).
    - column_pred : str
        Name of the column in `df` containing predicted aspect labels (JSON format).
    - aspect : str
        Aspect label for which precision is computed.

    Returns:
    - float
        Precision score for the specified aspect, computed as TP / (TP + FP),
        where TP is the number of true positives and FP is the number of false positives.
    """
  TP = 0
  FP = 0
  for index, row in df.iterrows():
    temp_true = json.loads(row[column_true])
    temp_pred = json.loads(row[column_pred])
    if aspect in temp_true and aspect in temp_pred:
      TP += 1
    elif aspect not in temp_true and aspect in temp_pred:
      FP += 1
  return TP / (TP + FP)

def recall_aspect(df : pd.DataFrame, column_true : str, column_pred : str, aspect : str) -> float:
  """
    Compute recall for a specific aspect in aspect-based sentiment analysis.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing the true and predicted aspect labels.
    - column_true : str
        Name of the column in `df` containing true aspect labels (JSON format).
    - column_pred : str
        Name of the column in `df` containing predicted aspect labels (JSON format).
    - aspect : str
        Aspect label for which recall is computed.

    Returns:
    - float
        Recall score for the specified aspect, computed as TP / (TP + FN),
        where TP is the number of true positives and FN is the number of false negatives.
    """
  TP = 0
  FN = 0
  for index, row in df.iterrows():
    temp_true = json.loads(row[column_true])
    temp_pred = json.loads(row[column_pred])
    if aspect in temp_true and aspect in temp_pred:
      TP += 1
    elif aspect in temp_true and aspect not in temp_pred:
      FN += 1
  return TP / (TP + FN)

def f1_aspect(df : pd.DataFrame, column_true : str, column_pred : str, aspect : str) -> float:
  """
    Compute F1 score for a specific aspect in aspect-based sentiment analysis.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing the true and predicted aspect labels.
    - column_true : str
        Name of the column in `df` containing true aspect labels (JSON format).
    - column_pred : str
        Name of the column in `df` containing predicted aspect labels (JSON format).
    - aspect : str
        Aspect label for which F1 score is computed.

    Returns:
    - float
        F1 score for the specified aspect, computed as 2 * (precision * recall) / (precision + recall).
        Precision and recall are computed based on TP (true positives), FP (false positives), and FN (false negatives).
    """
  precision = precision_aspect(df, column_true, column_pred, aspect)
  recall = recall_aspect(df, column_true, column_pred, aspect)
  return 2 * (precision * recall) / (precision + recall)

def accuracy_sentiment(df : pd.DataFrame, column_true : str, column_pred : str) -> float:
  """
    Compute accuracy for aspect-based sentiment analysis.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing the true and predicted aspect labels.
    - column_true : str
        Name of the column in `df` containing true aspect labels (JSON format).
    - column_pred : str
        Name of the column in `df` containing predicted aspect labels (JSON format).

    Returns:
    - float
        Accuracy score for aspect-based sentiment analysis, computed as the ratio of correct predictions
        to the total number of predictions evaluated across all aspects.
    """
  correct = 0
  total = 0
  for index, row in df.iterrows():
    temp_true = json.loads(row[column_true])
    temp_pred = json.loads(row[column_pred])
    intersect = list(set(temp_true.keys()) & set(temp_pred.keys()))
    total += len(intersect)
    for aspect in intersect:
      if temp_true[aspect] == temp_pred[aspect]:
        correct += 1
  return correct / total

def all_occuring_aspects(df : pd.DataFrame, column_true : str, column_pred = None, category = None):
  unique_aspects = set()
  if category is not None:
    df = df[df["appCategory"] == category]
  if column_pred is not None:
    for index, row in df.iterrows():
      temp_true = json.loads(row[column_true])
      temp_pred = json.loads(row[column_pred])
      unique_aspects.update(temp_pred.keys())
      unique_aspects.update(temp_true.keys())
  else:
    for index, row in df.iterrows():
      temp_true = json.loads(row[column_true])
      unique_aspects.update(temp_true.keys())
  return list(unique_aspects)

def macro_f1_aspect(df : pd.DataFrame, column_true : str, column_pred : str, weighted=True, category = None):
    """
    Compute the macro F1 score for aspect-based sentiment analysis.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing the true and predicted aspect labels.
    - column_true : str
        Name of the column in `df` containing true aspect labels (JSON format).
    - column_pred : str
        Name of the column in `df` containing predicted aspect labels (JSON format).
    - path_aspects : str
        Path to a JSON file containing all possible aspect labels.
    - weighted : bool, optional (default=True)
        If True, compute weighted macro F1 score based on support (TP + FP + FN) of each aspect.
        If False, compute unweighted macro F1 score.

    Returns:
    - tuple
        (macro_f1, metrics_per_aspect)
        - macro_f1 : float
            Macro F1 score averaged across all aspects.
        - metrics_per_aspect : dict
            Dictionary containing detailed metrics per aspect:
            {
                aspect: {
                    'TP': int,
                    'FP': int,
                    'FN': int,
                    'precision': float,
                    'recall': float,
                    'f1': float,
                    'support': int
                },
                ...
            }
            'support' is the total occurrences (TP + FP + FN) of each aspect.
            'precision', 'recall', and 'f1' are calculated based on TP, FP, and FN.
    """
    #with open(path_aspects, "r") as f:
      #all_aspects = json.load(f)
    if category is not None:
      df = df[df["appCategory"] == category]
      all_aspects = all_occuring_aspects(df, column_true=column_true, column_pred = column_pred, category = category)
    else:
      all_aspects = all_occuring_aspects(df, column_true=column_true, column_pred = column_pred)
    metrics_per_aspect = {aspect : {'TP' : 0, 'FP' : 0, 'FN' : 0, 'precision' : 0, 'recall' : 0, 'f1' : 0, 'support' : 0} for aspect in all_aspects}
    for index, row in df.iterrows():
      temp_true = json.loads(row[column_true])
      temp_pred = json.loads(row[column_pred])
      for aspect in list(set(temp_true.keys())|set(temp_pred.keys())):
        if aspect in temp_true and aspect in temp_pred:
          metrics_per_aspect[aspect]['TP']+=1
        elif aspect not in temp_true and aspect in temp_pred:
          metrics_per_aspect[aspect]['FP']+=1
        elif aspect in temp_true and aspect not in temp_pred:
          metrics_per_aspect[aspect]['FN']+=1
    total_f1 = 0
    total_recall = 0
    total_precision = 0
    if weighted is True:
      total_weight=0
      for aspect in metrics_per_aspect:
        metrics_per_aspect[aspect]['support'] = metrics_per_aspect[aspect]['TP'] + metrics_per_aspect[aspect]['FP'] + metrics_per_aspect[aspect]['FN']
        try:
          metrics_per_aspect[aspect]['precision'] = metrics_per_aspect[aspect]['TP'] / (metrics_per_aspect[aspect]['TP'] + metrics_per_aspect[aspect]['FP'])
        except ZeroDivisionError as e:
          metrics_per_aspect[aspect]['precision'] = 0
        try:
          metrics_per_aspect[aspect]['recall'] = metrics_per_aspect[aspect]['TP'] / (metrics_per_aspect[aspect]['TP'] + metrics_per_aspect[aspect]['FN'])
        except ZeroDivisionError as e:
          metrics_per_aspect[aspect]['recall'] = 0
        try:
          metrics_per_aspect[aspect]['f1'] = 2 * metrics_per_aspect[aspect]['precision'] * metrics_per_aspect[aspect]['recall'] / (metrics_per_aspect[aspect]['precision'] + metrics_per_aspect[aspect]['recall'])
        except ZeroDivisionError as e:
          metrics_per_aspect[aspect]['f1'] = 0
        total_f1 += metrics_per_aspect[aspect]['f1']* metrics_per_aspect[aspect]['support']
        total_recall += metrics_per_aspect[aspect]['recall']* metrics_per_aspect[aspect]['support']
        total_precision += metrics_per_aspect[aspect]['precision']* metrics_per_aspect[aspect]['support']
        total_weight += metrics_per_aspect[aspect]['support']
      macro_f1 = total_f1 / total_weight
      macro_recall = total_recall / total_weight
      macro_precision = total_precision / total_weight
    else:
      total_weight=0
      for aspect in metrics_per_aspect:
        metrics_per_aspect[aspect]['support'] = metrics_per_aspect[aspect]['TP'] + metrics_per_aspect[aspect]['FP'] + metrics_per_aspect[aspect]['FN']
        try:
          metrics_per_aspect[aspect]['precision'] = metrics_per_aspect[aspect]['TP'] / (metrics_per_aspect[aspect]['TP'] + metrics_per_aspect[aspect]['FP'])
        except ZeroDivisionError as e:
          metrics_per_aspect[aspect]['precision'] = 0
        try:
          metrics_per_aspect[aspect]['recall'] = metrics_per_aspect[aspect]['TP'] / (metrics_per_aspect[aspect]['TP'] + metrics_per_aspect[aspect]['FN'])
        except ZeroDivisionError as e:
          metrics_per_aspect[aspect]['recall'] = 0
        try:
          metrics_per_aspect[aspect]['f1'] = 2 * metrics_per_aspect[aspect]['precision'] * metrics_per_aspect[aspect]['recall'] / (metrics_per_aspect[aspect]['precision'] + metrics_per_aspect[aspect]['recall'])
        except ZeroDivisionError as e:
          metrics_per_aspect[aspect]['f1'] = 0
        total_f1 += metrics_per_aspect[aspect]['f1']
        total_recall += metrics_per_aspect[aspect]['recall']
        total_precision += metrics_per_aspect[aspect]['precision']
      macro_f1 = total_f1 / len(metrics_per_aspect)
      macro_recall = total_recall / len(metrics_per_aspect)
      macro_precision = total_precision / len(metrics_per_aspect)
    return (round(macro_f1, 3), metrics_per_aspect, round(macro_recall, 3), round(macro_precision,3))

def histogram_aspect_sentiments(reviews : pd.Dataframe, N : int, category=None, app = None, column="ABSALabels"):
  """

  Analyzes aspect terms for a given category in a reviews dataframe.

  Args:
    reviews: A pandas dataframe containing customer reviews.
    category: The category to analyze (e.g., "games").

  Returns:
    None. Prints insights and creates visualizations.
  """

  # Filter reviews for the chosen category
  if category is not None:
    filtered_reviews = reviews[reviews["appCategory"] == category]
  if app is not None:
    filtered_reviews = reviews[reviews["appName"] == app]

  # Function to extract aspects and sentiment from a json string
  def sentiments_per_aspect(df):
      dict_sentiments_per_aspect = {}
      default = {'positive' : 0, 'negative' : 0, 'neutral' : 0, 'total' : 0}
      for i in range(len(df)):
        temp_dict = json.loads(df.iloc[i][column])
        for aspect, sentiment in temp_dict.items():
          dict_sentiments_per_aspect[aspect] = dict_sentiments_per_aspect.get(aspect, default.copy())
          dict_sentiments_per_aspect[aspect][sentiment] += 1
          dict_sentiments_per_aspect[aspect]['total'] += 1
      return dict_sentiments_per_aspect
  dict_sentiments_per_aspect = sentiments_per_aspect(filtered_reviews)
  sorted_dict_sentiments_per_aspect = sorted(dict_sentiments_per_aspect.items(), key= lambda x: x[1]['total'], reverse=True)
  # Apply the extraction function to the ABSALabels column
  top_N_aspects = sorted_dict_sentiments_per_aspect[:N]

  pos_list = [t[1]['positive'] for t in top_N_aspects]
  neg_list = [t[1]['negative'] for t in top_N_aspects]
  neu_list = [t[1]['neutral'] for t in top_N_aspects]


  X_axis = np.arange(N)

  plt.bar(X_axis - 0.2, pos_list, 0.2, label = 'positive', color='green')
  plt.bar(X_axis, neg_list, 0.2, label = 'negative', color='red')
  plt.bar(X_axis + 0.2, neu_list, 0.2, label = 'neutral', color='blue')
  # Calculate sentiment proportions
  plt.xticks(X_axis, [t[0] for t in top_N_aspects])
  plt.xlabel("Aspect Term")
  plt.ylabel("Count")
  if app is not None:
    plt.title(f"Sentiment Distribution of Top {N} Most Frequent Aspect Terms for App: {app}")
  if category is not None:
    plt.title(f"Sentiment Distribution of Top {N} Most Frequent Aspect Terms for Category: {category}")
  plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
  plt.legend(title="Sentiment")
  #if app is not None:
    #plt.savefig(f"{images_dir}/{app}_sentiments_35k.png")
  #if category is not None:
    #plt.savefig(f"{images_dir}/{category}_sentiments_35k.png")
  plt.show()

  # Analyze the results:
  # - Analyze sentiment proportions to understand user feedback on each aspect.
 

def subplot_histogram_aspects_sentiments(df : pd.DataFrame, N : int, category=None, app = None, column="ABSALabels", ax = None):
  """

  Analyzes aspect terms for a given category in a reviews dataframe.

  Args:
    df: A pandas dataframe containing customer reviews.
    category: The category to analyze (e.g., "games").

  Returns:
    None. Prints insights and creates visualizations.
  """

  # Filter reviews for the chosen category
  if category is not None:
    filtered_df = df[df["appCategory"] == category]
  if app is not None:
    filtered_df = df[df["appName"] == app]

  # Function to extract aspects and sentiment from a json string
  def sentiments_per_aspect(df):
      dict_sentiments_per_aspect = {}
      default = {'positive' : 0, 'negative' : 0, 'neutral' : 0, 'total' : 0}
      for i in range(len(df)):
        temp_dict = json.loads(df.iloc[i][column])
        for aspect, sentiment in temp_dict.items():
          dict_sentiments_per_aspect[aspect] = dict_sentiments_per_aspect.get(aspect, default.copy())
          dict_sentiments_per_aspect[aspect][sentiment] += 1
          dict_sentiments_per_aspect[aspect]['total'] += 1
      return dict_sentiments_per_aspect
  dict_sentiments_per_aspect = sentiments_per_aspect(filtered_df)
  sorted_dict_sentiments_per_aspect = sorted(dict_sentiments_per_aspect.items(), key= lambda x: x[1]['total'], reverse=True)
  # Apply the extraction function to the ABSALabels column
  top_N_aspects = sorted_dict_sentiments_per_aspect[:N]

  pos_list = [t[1]['positive'] for t in top_N_aspects]
  neg_list = [t[1]['negative'] for t in top_N_aspects]
  neu_list = [t[1]['neutral'] for t in top_N_aspects]


  X_axis = np.arange(N)
  if ax is None:
      fig, ax = plt.subplots()
  else:
      fig = ax.figure
  ax.bar(X_axis - 0.2, pos_list, 0.2, label = 'positive', color='green')
  ax.bar(X_axis, neg_list, 0.2, label = 'negative', color='red')
  ax.bar(X_axis + 0.2, neu_list, 0.2, label = 'neutral', color='blue')
  # Calculate sentiment proportions
  ax.set_xticks(ticks=X_axis)
  ax.set_xticklabels([t[0] for t in top_N_aspects], rotation=45, ha='right')
  #ax.set_xlabel("Aspect Term")
  #ax.set_ylabel("Count")
  if app is not None:
    ax.set_title(f"App: {app}")
  if category is not None:
    ax.set_title(f"Category: {category}")
   # Rotate x-axis labels for readability

  #if app is not None:
    #ax.savefig(f"{images_dir}/{app}_sentiments_35k.png")
  #if category is not None:
    #ax.savefig(f"{images_dir}/{category}_sentiments_35k.png")
  return fig
  
def calculate_good_response_percentages(df):
    # Initialize a dictionary to hold the counts of good responses per question
    good_response_counts = {
        "Aspect Coverage": 0,
        "Sentiment Accuracy": 0,
        "Aspect-Opinion Pairing": 0,
        "Label Granularity": 0,
        "Consistency with Similar Reviews": 0,
        "Clarity of Sentiment": 0,
        "Redundancy Check": 0,
        "False Positives/Negatives": 0
    }

    # Total number of reviews
    total_reviews = len(df)

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        eval_json = json.loads(row['eval_questions_absa'])

        # Check each question and see if it has a good response
        for question, details in eval_json.items():
            answer = details["A"].lower()
            if (question in good_response_counts) and (
                (question in ["Aspect Coverage", "Sentiment Accuracy", "Aspect-Opinion Pairing", "Label Granularity", "Consistency with Similar Reviews", "Clarity of Sentiment"] and answer == "yes") or
                (question in ["Redundancy Check", "False Positives/Negatives"] and answer == "no")
            ):
                good_response_counts[question] += 1

    # Calculate the percentage of good responses for each question
    good_response_percentages = {question: (count / total_reviews) * 100 for question, count in good_response_counts.items()}

    return good_response_percentages

def precision_aspect(df : pd.DataFrame, column_true : str, column_pred : str, aspect : str) -> float:
  """
    Compute precision for a specific aspect in aspect-based sentiment analysis.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing the true and predicted aspect labels.
    - column_true : str
        Name of the column in `df` containing true aspect labels (JSON format).
    - column_pred : str
        Name of the column in `df` containing predicted aspect labels (JSON format).
    - aspect : str
        Aspect label for which precision is computed.

    Returns:
    - float
        Precision score for the specified aspect, computed as TP / (TP + FP),
        where TP is the number of true positives and FP is the number of false positives.
    """
  TP = 0
  FP = 0
  for index, row in df.iterrows():
    temp_true = json.loads(row[column_true])
    temp_pred = json.loads(row[column_pred])
    if aspect in temp_true and aspect in temp_pred:
      TP += 1
    elif aspect not in temp_true and aspect in temp_pred:
      FP += 1
  return TP / (TP + FP)

def recall_aspect(df : pd.DataFrame, column_true : str, column_pred : str, aspect : str) -> float:
  """
    Compute recall for a specific aspect in aspect-based sentiment analysis.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing the true and predicted aspect labels.
    - column_true : str
        Name of the column in `df` containing true aspect labels (JSON format).
    - column_pred : str
        Name of the column in `df` containing predicted aspect labels (JSON format).
    - aspect : str
        Aspect label for which recall is computed.

    Returns:
    - float
        Recall score for the specified aspect, computed as TP / (TP + FN),
        where TP is the number of true positives and FN is the number of false negatives.
    """
  TP = 0
  FN = 0
  for index, row in df.iterrows():
    temp_true = json.loads(row[column_true])
    temp_pred = json.loads(row[column_pred])
    if aspect in temp_true and aspect in temp_pred:
      TP += 1
    elif aspect in temp_true and aspect not in temp_pred:
      FN += 1
  return TP / (TP + FN)

def f1_aspect(df : pd.DataFrame, column_true : str, column_pred : str, aspect : str) -> float:
  """
    Compute F1 score for a specific aspect in aspect-based sentiment analysis.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing the true and predicted aspect labels.
    - column_true : str
        Name of the column in `df` containing true aspect labels (JSON format).
    - column_pred : str
        Name of the column in `df` containing predicted aspect labels (JSON format).
    - aspect : str
        Aspect label for which F1 score is computed.

    Returns:
    - float
        F1 score for the specified aspect, computed as 2 * (precision * recall) / (precision + recall).
        Precision and recall are computed based on TP (true positives), FP (false positives), and FN (false negatives).
    """
  precision = precision_aspect(df, column_true, column_pred, aspect)
  recall = recall_aspect(df, column_true, column_pred, aspect)
  return 2 * (precision * recall) / (precision + recall)

def accuracy_sentiment(df : pd.DataFrame, column_true : str, column_pred : str) -> float:
  """
    Compute accuracy for aspect-based sentiment analysis.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing the true and predicted aspect labels.
    - column_true : str
        Name of the column in `df` containing true aspect labels (JSON format).
    - column_pred : str
        Name of the column in `df` containing predicted aspect labels (JSON format).

    Returns:
    - float
        Accuracy score for aspect-based sentiment analysis, computed as the ratio of correct predictions
        to the total number of predictions evaluated across all aspects.
    """
  correct = 0
  total = 0
  for index, row in df.iterrows():
    temp_true = json.loads(row[column_true])
    temp_pred = json.loads(row[column_pred])
    intersect = list(set(temp_true.keys()) & set(temp_pred.keys()))
    total += len(intersect)
    for aspect in intersect:
      if temp_true[aspect] == temp_pred[aspect]:
        correct += 1
  return correct / total

def all_occuring_aspects(df : pd.DataFrame, column_true : str, column_pred = None, category = None):
  unique_aspects = set()
  if category is not None:
    df = df[df["appCategory"] == category]
  if column_pred is not None:
    for index, row in df.iterrows():
      temp_true = json.loads(row[column_true])
      temp_pred = json.loads(row[column_pred])
      unique_aspects.update(temp_pred.keys())
      unique_aspects.update(temp_true.keys())
  else:
    for index, row in df.iterrows():
      temp_true = json.loads(row[column_true])
      unique_aspects.update(temp_true.keys())
  return list(unique_aspects)