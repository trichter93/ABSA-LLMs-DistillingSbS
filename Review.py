from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, ClassVar, List, IO, Tuple, Union
import json
import pandas as pd

base = "/content/drive/MyDrive/ABSA-LLMs-DistillingSbS"
path_allowed_models = base + '/data/allowed_models.json'
path_generated_dataset_names = base + '/data/generated_dataset_names.json'
path_app_categories = base + '/data/app_categories.json'
path_general_aspects = base + '/data/general_aspects.json'
path_category_specific_aspects = base + '/data/category_specific_aspects.json'
path_app_names = base + '/data/app_names.json' 
Review_meta_data = {path_allowed_models : [], path_generated_dataset_names: [], path_app_categories : [], path_general_aspects : [], path_category_specific_aspects : [], path_app_names : []}

for path in Review_meta_data:
  with open(path, 'r') as _f:
    Review_meta_data[path] = json.load(_f)

class Review(BaseModel):
    user_review_id: str = Field(description="The review Id of the review")
    app_name: str = Field(description="The app name of the app in question")
    title: str = Field(description="The title of the review")
    body: str = Field(description="The body of the review")
    app_category : str = Field(description="The app category of the app in question")
    aspects : List[str] = Field(description="The possible aspects of the review in a list")
    aspects_and_scores: Optional[Dict[str, str]] = Field(description="The result from manual annotation of the review. ", default=None)
    generated_aspects_and_scores: Dict[str, Dict[str, Any]] = Field(description="A dictionary containing the results from openAI API-calls.", default={})
    _valid_sentiments: ClassVar[List[str]]  = ["positive", "negative", "neutral"]
    _allowed_models: ClassVar[List[str]]  = Review_meta_data[path_allowed_models]
    _generated_dataset_names: ClassVar[List[str]] = Review_meta_data[path_generated_dataset_names]
    _unique_app_names: ClassVar[List[str]] = Review_meta_data[path_app_names]
    _unique_app_categories: ClassVar[List[str]] = Review_meta_data[path_app_categories]
    _general_aspects : ClassVar[List[str]] = Review_meta_data[path_general_aspects]
    _category_specific_aspects : ClassVar[Dict[str, List[str]]] = Review_meta_data[path_category_specific_aspects]

    @classmethod
    def from_dataframe_row(cls, row : pd.Series):
        generated_aspects_and_scores = {column : json.loads(row[column]) if (column not in row.index or pd.isna(row[column]) or row[column]=='')  is False else {} for column in cls._generated_dataset_names}
        return cls(
            user_review_id=row['userReviewId'],
            app_name=row['appName'],
            title=row['title'],
            body=row['body'],
            app_category=row['appCategory'],
            aspects = (cls._general_aspects + cls._category_specific_aspects[row["appCategory"]]),
            aspects_and_scores = json.loads(row['aspectsAndScores']) if pd.isna(row['aspectsAndScores']) is False else None,
            generated_aspects_and_scores = generated_aspects_and_scores
        )

    def update_aspects_and_scores(self, column_name : str,  new_aspects_and_scores : Any, temp : Any = None, top_p : Any = None, predefined_aspects : bool = True, rationales : bool = False):
        if isinstance(new_aspects_and_scores, str):
          try:
              new_aspects_and_scores = json.loads(new_aspects_and_scores)
          except json.JSONDecodeError:
              raise ValueError("Invalid JSON format")
        if column_name not in self._generated_dataset_names:
              raise ValueError(f"Dataset name {column_name} is not in the list of allowed dataset names")
        if predefined_aspects and not rationales:
          for key, value in new_aspects_and_scores.items():
            if key not in self.aspects:
                raise ValueError(f"Aspect {key} is not in the list of aspects")
            if value not in self._valid_sentiments:
                raise ValueError(f"Sentiment {value} is not in the list of valid sentiments")
        if predefined_aspects and rationales:
          for key, value in new_aspects_and_scores.items():
            if key not in self.aspects:
                raise ValueError(f"Aspect {key} is not in the list of aspects")
            if value[0] not in self._valid_sentiments:
                raise ValueError(f"Sentiment {value} is not in the list of valid sentiments")
        if not predefined_aspects and not rationales:
          for key, value in new_aspects_and_scores.items():
            if value not in self._valid_sentiments:
                raise ValueError(f"Sentiment {value} is not in the list of valid sentiments")
        if not predefined_aspects and rationales:
          for key, value in new_aspects_and_scores.items():
            if value[0] not in self._valid_sentiments:
                raise ValueError(f"Sentiment {value} is not in the list of valid sentiments")
        if temp is not None:
          self.generated_aspects_and_scores[column_name] = {"temp": temp, "aspects_and_scores": new_aspects_and_scores}
        elif top_p is not None:
          self.generated_aspects_and_scores[column_name] = {"top_p": top_p, "aspects_and_scores": new_aspects_and_scores}


    @validator("app_name")
    def validate_app_name(cls, value):
        if value not in cls._unique_app_names:
            raise ValueError(f"App name {value} is not in the list of unique app names")
        return value

    @validator("app_category")
    def validate_app_category(cls, value):
        if value not in cls._unique_app_categories:
            raise ValueError(f"App category {value} is not in the list of unique app categories")
        return value

    @validator("aspects")
    def validate_aspects(cls, value, values):
        app_category = values["app_category"]
        valid_aspects = cls._general_aspects + cls._category_specific_aspects[app_category]
        for aspect in value:
            if aspect not in valid_aspects:
                raise ValueError(f"Aspect {aspect} is not in the list of valid aspects")
        return value

    @validator("aspects_and_scores")
    def validate_aspects_and_scores(cls, value, values):
        aspects_and_scores = value
        if aspects_and_scores is None:
          return aspects_and_scores
        field_dict = values
        aspects = field_dict["aspects"]
        for key, value in aspects_and_scores.items():
            if key not in aspects:
                raise ValueError(f"Aspect {key} is not in the list of aspects")
            if value not in cls._valid_sentiments:
                raise ValueError(f"Sentiment {value} is not in the list of valid sentiments")
        return aspects_and_scores