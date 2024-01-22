# udacity-ds-nanodegree-pj-2
This repository contains Python files that are used to analyze data for a disaster response pipeline, the data was provided by figure eight.

# Motivation
In an era where timely and effective disaster response is crucial, this project embodies the intersection of data science and real-world applications. It meticulously executes a spectrum of data engineering tasks to develop a sophisticated supervised machine learning model. The primary objective is to categorize tweets and messages from actual disaster events efficiently. Such categorization is vital for disaster response teams to prioritize and address the most critical messages swiftly. This initiative is a fragment of a broader educational endeavor, spotlighting the transformative potential of data science in tackling real-life challenges.

# Libraries used
This project utilizes several Python libraries for data processing, analysis, visualization and deployment of a web application.

- numpy: For efficient numerical computations.
- pandas: For data manipulation and analysis.
- sklearn (Scikit-Learn): For machine learning and predictive data analysis.
- sqlite3: For CRUD operations.
- sqlalchemy: For CRUD operations on a sqlite database.
- nltk: For natural language processing.
- flask: Used to deploy a web application where users can interact with the ML Model.
- joblib: Used to load machine learning models.

# Files in the repository

## Data
process_data.py: Scripts responsible for cleaning the dataset. The output is a streamlined SQLite database file, optimized for subsequent operations.

### How to execute relevant commands:
From project root directory:

```bash
cd data 
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

## Models
train_classifier.py: Scripts dedicated to training the machine learning model using a comprehensive scikit-learn pipeline, ensuring accuracy and efficiency.

### How to execute relevant commands:
From project root directory:

```bash
cd models 
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```

## App
A dynamic Flask application that serves dual purposes: visualizing pertinent data and enabling users to classify messages through the trained machine learning model housed in the models directory.

### How to execute relevant commands:
From project root directory:

```bash
cd app 
python run.py
```

# Acknowledgements
This project was made possible thanks to figure eight and udacity.
