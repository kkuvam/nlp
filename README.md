# Natural Language Processing with NLTK & spaCy

This repository is an internship project undertaken as part of a winter internship.  The goal of the project was to explore a variety of Natural Language Processing (NLP) tasks using two of the most widely‑used Python libraries: **NLTK** and **spaCy**.  Through a series of hands‑on Jupyter notebooks you’ll see how basic text processing, language identification, tokenization, part‑of‑speech tagging, classification and sentiment analysis are implemented with NLTK, and how more advanced processing pipelines and custom models are built using spaCy.

## Repository structure

```
nlp/
├── notebook/
│   ├── nltk/        # Notebooks from the NLTK mini‑course
│   │   ├── data/    # Auxiliary data files used by some notebooks
│   │   └── *.ipynb  # Jupyter notebooks covering various NLP tasks
│   └── spacy/       # Notebooks from the spaCy mini‑course
│       ├── exercises/  # JSON data used in Chapter 4 training examples
│       ├── img/        # Images used in the course notebooks
│       └── Chapter_*.ipynb
├── requirements.txt # Python dependencies for this project【737895943819024†L0-L10】
└── README.md        # this file
```

### NLTK mini‑course

The `notebook/nltk` folder contains a set of Jupyter notebooks that gradually introduce NLTK’s core functionality and apply it to a variety of problems.  The notebooks are numbered to reflect their order in the mini‑course.  A brief overview is given below:

| Notebook | Description |
|---|---|
| **1‑1‑Downloading‑Libs‑and‑Testing‑That‑They‑Are‑Working.ipynb** | A short notebook to verify that NLTK and other dependencies install correctly. |
| **1‑2‑Text‑Analysis‑Using‑nltk.text.ipynb** | Introduces the `nltk.text` module for concordance and dispersion plots to explore word usage in corpora. |
| **2‑1‑Deriving‑N‑Grams‑from‑Text.ipynb** | Demonstrates how to create n‑grams from a tokenized sentence and explains what n‑grams represent. |
| **2‑2‑Detecting‑Text‑Language‑by‑Counting‑Stop‑Words.ipynb** | Shows how to detect the language of a piece of text by comparing stop‑word frequencies across languages. |
| **2‑3‑Language‑Identifier‑Using‑Word‑Bigrams.ipynb** | Builds a simple language identifier using word bigrams and frequency counts. |
| **3‑1‑Bigrams‑Stemming‑and‑Lemmatizing.ipynb** | Explores the Reuters corpus and demonstrates bigrams, stemming and lemmatization. |
| **3‑2‑Finding‑Unusual‑Words‑in‑Given‑Language.ipynb** | Uses word frequency analysis to find unusual or foreign words in a text. |
| **3‑3‑Creating‑a‑POS‑Tagger.ipynb** | Shows how to train and evaluate a custom part‑of‑speech (POS) tagger. |
| **3‑4‑Parts‑of‑Speech‑and‑Meaning.ipynb** | Investigates the connection between POS tags and semantics, using NLTK’s WordNet. |
| **4‑1‑Name‑Gender‑Identifier.ipynb** | Builds a Naïve Bayes classifier to predict a person’s gender from their name. |
| **4‑2‑Classifying‑News‑Documents‑into‑Categories.ipynb** | Applies machine‑learning methods to categorize Reuters news articles into topics. |
| **5‑1‑Sentiment‑Analysis.ipynb** | Introduces sentiment analysis on text using NLTK’s built‑in tools. |
| **5‑2‑Sentiment‑Analysis‑with‑nltk.sentiment.SentimentAnalyzer‑and‑VADER‑tools.ipynb** | Shows how to use NLTK’s `SentimentAnalyzer` and the VADER lexicon to perform fine‑grained sentiment analysis. |


### spaCy mini‑course

The `notebook/spacy` folder contains notebooks and supporting data from spaCy’s official course.  These notebooks follow the same four‑chapter structure as the online course and include exercises and images used in the lessons.  Each chapter introduces a distinct aspect of spaCy:

| Chapter | Highlights |
|---|---|
| **Chapter 01 – Finding words, phrases, names and concepts** | Introduces spaCy, shows how to create an `nlp` object, and explains the `Doc`, `Token` and `Span` objects.  The chapter covers tokenization, how to iterate over tokens, and how to create slices (spans) to represent phrases.  It also discusses lexical attributes like `is_alpha` and `like_num`. |
| **Chapter 02 – Large‑scale data analysis with spaCy** | Focuses on spaCy’s shared vocabulary and `StringStore`.  It explains how spaCy uses hash values to represent strings and how to access the vocabulary via `nlp.vocab.strings`.  The chapter also teaches how to efficiently process large corpora and combine statistical and rule‑based methods. |
| **Chapter 03 – Processing pipelines** | Describes spaCy’s processing pipeline, including tokenization, tagging, parsing and named entity recognition.  It explains the built‑in pipeline components (tagger, parser, attribute ruler, lemmatizer and NER) and shows how to view, customize or extend the pipeline by adding your own components. |
| **Chapter 04 – Training a neural network model** | Covers training and updating spaCy’s statistical models.  It explains why you might want to customize a model for your own domain, the mechanics of training (initializing weights, updating with examples, calculating gradients) and how to build a custom named entity recognizer.  The `exercises` folder contains JSON files such as `bookquotes.json`, `capitals.json` and `countries.json` which are used as training examples. |

Together, these chapters form a practical mini‑course on spaCy that progresses from basic text processing to custom model training.

## Getting started

1. **Clone the repository**

   ```bash
   git clone https://github.com/kkuvam/nlp.git
   cd nlp
   ```

2. **Create and activate a virtual environment (recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Windows use venv\Scripts\activate
   ```

3. **Install the dependencies**

   Install all required packages listed in `requirements.txt` using pip:

   ```bash
   pip install -r requirements.txt
   ```

   The key libraries used in this project include Jupyter notebooks for interactive coding, spaCy and NLTK for NLP functionality, NumPy for numerical computations, Matplotlib for plotting, Tweepy and TwitterSearch for fetching tweets, and additional utilities like `unidecode`, `langdetect`, `langid` and `gensim`.

4. **Download NLTK corpora**

   Many of the NLTK notebooks rely on corpora such as stop‑words lists and the Reuters corpus.  If you haven’t downloaded these resources before, run the following in a Python interpreter:

   ```python
   import nltk
   nltk.download('punkt')      # for tokenizers
   nltk.download('stopwords')  # stop words used in language detection
   nltk.download('averaged_perceptron_tagger')  # POS tagger
   nltk.download('wordnet')    # WordNet for lemmatization
   nltk.download('reuters')    # Reuters corpus used in 3-1 notebook
   ```

5. **Run the notebooks**

   Launch Jupyter Notebook (or JupyterLab) and open any of the notebooks under `notebook/nltk` or `notebook/spacy`:

   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

   Each notebook is self‑contained and includes explanatory markdown cells alongside executable code.

## Project outcomes

By the end of this internship project, the following objectives were achieved:

* **Familiarity with NLTK fundamentals:**  The early notebooks demonstrate how to install NLTK and perform basic text analysis using the `nltk.text` module.  Subsequent notebooks explore n‑grams and language detection using stop words, build a simple language identifier using word bigrams, and use classic NLP techniques such as stemming, lemmatization and part‑of‑speech tagging.
* **Applied machine‑learning techniques to NLP:**  Several notebooks move beyond simple preprocessing and implement classification tasks.  Notably, a Naïve Bayes classifier predicts gender from first names, Reuters documents are categorized by topic, and sentiment analysis is performed using NLTK’s `SentimentAnalyzer` and the VADER lexicon.
* **Hands‑on experience with spaCy:**  The spaCy mini‑course walks through the core API: creating an `nlp` object and iterating over tokens and spans, using vocabularies and string stores for efficient large‑scale text analysis【, understanding and customizing processing pipelines, and finally training and updating statistical models such as custom named entity recognizers.
* **Exposure to real‑world datasets:**  The project works with a variety of texts – from simple sentences and names to newswire articles (Reuters corpus) and JSON files containing quotes, countries and gadget names – giving practical insight into how NLP techniques perform on different kinds of data.


## License

No explicit license file is provided.  Unless otherwise stated in specific notebooks, assume that the notebooks and code are made available for educational purposes under an open‑source spirit.  Please respect the licenses of the underlying datasets and external libraries (e.g., Reuters corpus, stop‑word lists and spaCy models) when using this repository.
