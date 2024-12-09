# Bias Breaker : An NLP Journey into Sexism Detection

## Executive Summary
The Sexism Detection Project aims to analyze and classify text data for identifying sexist remarks. Utilizing machine learning techniques, this project processes a dataset containing various comments and categorizes them based on their sexist nature. The project employs natural language processing (NLP) methods, including tokenization, vectorization, and model training, to achieve accurate classification results.

## Table of Contents
- [Bias Breaker : An NLP Journey into Sexism Detection](#bias-breaker--an-nlp-journey-into-sexism-detection)
  - [Executive Summary](#executive-summary)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [File Structure](#file-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Model Training](#model-training)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)
  
## Project Overview
This project is designed to detect and analyze sexist comments in English text. It involves data preprocessing, feature extraction, and model training using various machine learning algorithms. The primary goal is to create a reliable model that can classify comments as sexist or not.

## Dataset
The dataset used in this project consists of three CSV files:
- `dev.csv`: Development dataset for validation.
- `train.csv`: Training dataset for model training.
- `test.csv`: Testing dataset for evaluating model performance.

The dataset contains the following columns:
- **ID**: Unique identifier for each comment.
- **Text**: The comment text for analysis.
- **Sexism**: Label indicating whether the comment is sexist or not.
- **Category**: Type of sexist remarks.
- **Rationale**: Details on the sexist remarks.
- **Split**: Indicates the dataset split (train, dev, test).

## File Structure
```
Notebooks/
│
├── sexismDetection.ipynb       # Jupyter Notebook containing the analysis and model training
│
├── archive/
│   ├── dev.csv                  # Development dataset
│   ├── train.csv                # Training dataset
│   └── test.csv                 # Testing dataset
│
└── sexism_df.csv                # Processed dataset saved after cleaning and transformation
```

## Installation
To run this project, ensure you have the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- wordcloud
- scikit-learn
- nltk

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn nltk
```


## Usage
1. Clone the repository:
   ```bash
   git clone git@github.com:Darylwanji/Bias-Breaker--An-NLP-Journey-into-Sexism-Detection.git
   cd archive
   ```

2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Notebooks/sexismDetection.ipynb
   ```

3. Run the cells in the notebook to execute the analysis and model training.

## Model Training
The project utilizes a Logistic Regression model and a Decision Tree Classifier within a pipeline for classification tasks. The model is trained on the processed training dataset, and hyperparameter tuning is performed using GridSearchCV.

## Results
The model's performance is evaluated using accuracy scores on the training, testing, and validation datasets. The results are printed in the notebook for review.

## Contributing
Contributions are welcome! If you have suggestions for improvements or features, please create an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.