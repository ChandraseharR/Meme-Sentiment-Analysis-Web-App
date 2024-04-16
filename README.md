# Meme Classification Web Application Using Machine Learning:

This repository contains a web application associated with a collection of a few classification algorithms using machine learning in Python to determine the sentiments behind internet memes based on image and text data extracted from around 6,992 different internet memes.

### Dependencies:

- Jupyter Notebook ([install](https://docs.jupyter.org/en/latest/install.html))
- pandas ([install](https://pandas.pydata.org/docs/getting_started/install.html))
- NumPy ([install](https://numpy.org/install/))
- Matplotlib ([install](https://matplotlib.org/stable/users/installing/index.html))
- NLTK ([install](https://www.nltk.org/install.html))
- scikit-learn ([install](https://scikit-learn.org/stable/install.html))
- scikit-image ([install](https://scikit-image.org/docs/stable/install.html))
- Pillow (PIL Fork) ([install](https://pillow.readthedocs.io/en/stable/installation.html))
- Tesseract ([install](https://github.com/tesseract-ocr/tesseract))
- Pytesseract (install - [Anaconda](https://anaconda.org/conda-forge/pytesseract) | install - [PyPI](https://pypi.org/project/pytesseract/))
- Flask ([install](https://flask.palletsprojects.com/en/2.2.x/installation/))

## Introduction:

One of the most common applications of classification algorithms is image and text classification to determine which pre-determined categories certain image and/or text data is the most relevant to. While classification algorithms work for a variety of image and text data, I've trained certain image and text classification models specifically for the classification of internet memes to determine whether a certain meme relays one of five pre-categorised sentiments; neutral, positive, negative, very positive, and very negative.

### Classifiers Used (scikit-learn):

- sklearn.ensemble.RandomForestClassifier ([read](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html))
- sklearn.neighbors.KNeighborsClassifier ([read](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html))
- sklearn.ensemble.ExtraTreesClassifier ([read](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html))
- sklearn.linear_model.SGDClassifier ([read](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html))
- sklearn.naive_bayes.MultinomialNB ([read](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html))
- sklearn.linear_model.LogisticRegression ([read](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html))

## Usage:

- `Meme Classification.ipynb` — Contains the implementations (scikit-learn) of all trained and tested image and text classification models.
- `app.py` — Source code for the web application (Flask) associated with the classification algorithms using machine learning.
- `test_images` — Contains the images used for testing the trained image and text classification models.
- `templates` — Contains the source codes for the web pages (`home.html` and `predict.html`) rendered by the web application (Flask).
- `static\files` — Directory used by the web application (Flask) to store the uploaded images into.

### References:

- Gong, D. (2022, July 12). _Top 6 Machine Learning Algorithms for Classification._ Medium. https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501
- V, N. (2022, March 15). _Image Classification using Machine Learning._ Analytics Vidhya. https://www.analyticsvidhya.com/blog/2022/01/image-classification-using-machine-learning/
- EliteDataScience. (2022, July 6). _How to Handle Imbalanced Classes in Machine Learning._ https://elitedatascience.com/imbalanced-classes
- Ankit, U. (2022, January 6). _Image Classification of PCBs and its Web Application (Flask)._ Medium. https://towardsdatascience.com/image-classification-of-pcbs-and-its-web-application-flask-c2b26039924a
- GeeksforGeeks. (2020, December 26). _How to Extract Text from Images with Python?_ https://www.geeksforgeeks.org/how-to-extract-text-from-images-with-python/
