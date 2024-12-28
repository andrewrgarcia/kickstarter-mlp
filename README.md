# Kickstarter-MLP

This repository implements a deep learning pipeline to predict outcomes for Kickstarter projects using features such as `goal`, `category`, and `main_category`. The primary model is a shallow Multi-Layer Perceptron (MLP) built with PyTorch, compared against a Linear Regression baseline.

The dataset used is the [Kickstarter Projects dataset from Kaggle](https://www.kaggle.com/datasets/kemical/kickstarter-projects), which includes over 300,000 projects with features like funding goals, pledged amounts, categories, and backers. Key preprocessing steps include filtering, one-hot encoding of categorical features, and normalization of numerical data.

## License

This project is licensed under the [MIT License](LICENSE). 
