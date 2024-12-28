import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class KickstarterDataProcessor:
    def __init__(self, y_response='backers', rows_to_remove=None):
        self.y_response = y_response
        self.rows_to_remove = rows_to_remove
        self.encoder = None
        self.scaler = None

    def fit_transform(self, df):
        """
        Fit encoders and preprocess the dataset.
        """
        # Optionally remove random rows
        if self.rows_to_remove:
            df = df.sample(len(df) - self.rows_to_remove, random_state=42)

        df = self.data_selection(df)

        # Encode categorical columns
        categorical_columns = ['category', 'main_category']
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_categorical = self.encoder.fit_transform(df[categorical_columns])

        # Combine processed data
        X = np.hstack((encoded_categorical, df[['goal']].values))
        y = df[self.y_response].values

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize numerical features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def transform(self, df):
        """Preprocess a new dataset using the fitted encoder and scaler."""
        if not self.encoder or not self.scaler:
            raise ValueError("The processor has not been fitted. Call `fit_transform` first.")

        df = self.data_selection(df)

        # Encode categorical columns
        categorical_columns = ['category', 'main_category']
        encoded_categorical = self.encoder.transform(df[categorical_columns])

        # Combine processed data
        X = np.hstack((encoded_categorical, df[['goal']].values))

        # Normalize numerical features
        X = self.scaler.transform(X)

        return X
    
    def data_selection(self, df):
        # Filter: only projects with crowdfunding goals in USD
        df = df[df["currency"] == 'USD']

        # Select relevant columns
        columns_to_use = ['category', 'main_category', 'goal', 'backers', 'usd pledged']
        df = df[columns_to_use]

        # Drop missing values
        df = df.dropna()

        return df 
    
    def prepare_deployment_data(self, df):
        """Preprocess deployment data using the fitted encoder and scaler."""
        if not self.encoder or not self.scaler:
            raise ValueError("The processor has not been fitted. Call `fit_transform` first.")

        categorical_columns = ['category', 'main_category']

        # Encode categorical columns
        deployment_encoded = self.encoder.transform(df[categorical_columns])

        # Combine encoded features with 'goal'
        X_deployment = np.hstack((deployment_encoded, df[['goal']].values))

        # Normalize numerical features
        X_deployment = self.scaler.transform(X_deployment)

        return X_deployment