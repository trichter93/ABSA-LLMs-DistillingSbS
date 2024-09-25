# ABSA-LLMs-DistillingSbS
My work for my Masters Thesis in Natural Language Processing and the Aspect-Based Sentiment Analysis task. Started at Backtick.

This project is designed to be run in Google Colab. All code and dependencies have only been tested in Colab's environment and it will not run locally as is.

To run this project you need to upload the directory to your Google Drive. The directory should sit at /content/drive/MyDrive

Please note that:

1. The notebook preprocessing.ipynb was originally used with the complete dataset which I cannot share in its entirety as that is outside of what I have rights to. I've included a small sample of reviews to make it run and show how the functions work, albeit on a smaller scale.

2. The notebook annotation.ipynb utilizes the OpenAI API which is not free and requires you to have your own API-key. If you wish to use it, make sure that your API-key is stored in Secrets. Again, I've provided a small sample of reviews that allow you to see how reviews get annotated.

3. The notebook training_and_evaluation.ipynb accesses models via HuggingFace and you cannot perform the complete training as, again, you will not have the entire training dataset. Also, the complete training requires you to use an A100 GPU which you only have access to with a Google Colab Pro or Google Colab Pro+ subscription. 

The notebooks are intended to be run in the order above. You may of course adapt any or all of the code to run in your specific environment (via some cloud provider or if you have access to industry-grade GPU's locally) but that will require some work.