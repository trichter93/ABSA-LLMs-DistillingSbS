from google.colab import userdata
import openai
from Review import Review

OPENAI_API_KEY = userdata.get("OpenAI_API_KEY")
OPENAI_MODEL = 'gpt-3.5-turbo'
client = openai.Client(api_key=OPENAI_API_KEY)


PROMPT_ABSA_WITH_RATIONALES = """Consider the following review of a mobile application:

App Name: {review.app_name}
Review Title: \"{review.title}\"
Review Body: \"{review.body}\"
App Category : {review.app_category}

Perform aspect-based sentiment analysis on the review, and consider only the aspects explicitly or implicitly mentioned in the review.
Generate a JSON-string with aspects as keys and JSON-arrays as values with the sentiment polarities (positive, negative, neutral) as the first element and a short
rationale explaining what prompted the inclusion of each aspect as the second element.

#### Example 1:
- Review Title : \"Disappointing\"
- Review Body : \"This app has potential, but the lack of essential features like offline mode, dark mode, and collaboration options is disappointing. The features need improvement.\"
- Aspects, Sentiments and Rationales : {{
  "offline mode": ["negative", "The absence of offline mode is cited as a disappointment."],
  "dark mode": ["negative", "The lack of dark mode is missing and cited as a disappointment."],
  "collaboration options": ["negative", "The absence of collaboration options is cited as disappointing, suggesting dissatisfaction with the app's collaborative features."],
  "features": ["negative", "Overall dissatisfaction with the app's features, indicating that they need improvement to meet user expectations."]
}}

#### Example 2:
- Review Title: \"Its OK, but competitors are better\"
- Review Body: \"While this app is decent, competitors like AppX and AppY offer a more seamless and efficient experience. They have better user interfaces and faster response times.\"
- Aspects and Sentiments: {{
  "user interface": ["negative", "The comparison suggests that the app's user interface is inferior to competitors, indicating dissatisfaction with its design or usability."],
  "response time": ["negative", "The mention of competitors having faster response times implies dissatisfaction with the app's performance in this aspect."]
}}

### Example 3:
- Review Title: \"This app is awesome!\"
- Review Body: \"This app is awesome! I love it! I can't wait to use it!\"
- Aspects and Sentiments: {{
}}

Be creative and include aspects even if they are only mentioned implicitly. Keep aspects and sentiments in lower case. If there are no aspects, return an empty JSON-Object."""

def generate_aspects_and_scores_json_format(review : Review, column_name, model_name=OPENAI_MODEL, temperature=None, top_p = None, prompt_raw = PROMPT_ABSA_WITH_RATIONALES, predefined_aspects=True, rationales = False, failed_ids = []):
  """Use this if model is 'gpt-3.5-turbo' or 'gpt-4'"""
  if temperature is not None:
    response = client.chat.completions.create(
      model=model_name,  # Change this to "gpt-3.5-turbo"
      response_format={ "type": "json_object" },
      messages=[{"role": "user", "content": prompt}],
      max_tokens=800,  # Adjust as needed
      temperature=temperature,  # Adjust as needed
      n=1,  # Number of completions to generate
    )
  if top_p is not None:
    response = client.chat.completions.create(
      model=model_name,  # Change this to "gpt-3.5-turbo"
      response_format={ "type": "json_object" },
      messages=[{"role": "user", "content": prompt}],
      max_tokens=800,  # Adjust as needed
      top_p=top_p,  # Adjust as needed
      n=1,  # Number of completions to generate
    )
  new_aspects_and_scores = response.choices[0].message.content.lower()
  if predefined_aspects:
    try:
      review.update_aspects_and_scores(column_name=column_name,  new_aspects_and_scores=new_aspects_and_scores, temp=temperature, top_p=top_p, predefined_aspects = predefined_aspects, rationales = rationales)
    except ValueError as v:
      print(v)
      print(f"review : {review.body}")
      print(f"new_aspects_and_scores : {new_aspects_and_scores}")
      print(f"self.aspects : {review.aspects}")
      new_aspects_and_scores_dict = json.loads(new_aspects_and_scores)
      invalid_aspects = set(new_aspects_and_scores_dict.keys()) - set(review.aspects)
      print(f"The aspect(s) that was not found in self.aspects : {invalid_aspects}")
      cleaned_aspects_and_scores_dict = {key: value for key, value in new_aspects_and_scores_dict.items() if key not in invalid_aspects}
      review.update_aspects_and_scores(column_name=column_name,  new_aspects_and_scores=cleaned_aspects_and_scores_dict, temp=temperature, top_p=top_p, predefined_aspects=predefined_aspects, rationales=rationales)
  else:
      try: 
        review.update_aspects_and_scores(column_name=column_name,  new_aspects_and_scores=new_aspects_and_scores, temp=temperature, top_p=top_p, predefined_aspects=predefined_aspects, rationales = rationales)
        print('success')
      except ValueError as v:
          print(v)
          failed_ids.append(review.user_review_id)
  