from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class TabularAdapter:
    """
    Base adapter for converting tabular data to PyTorch tensors.
    Handles missing values and encoding (StandardScaler for numerical, OneHot for categorical).
    """
    def __init__(self, numeric_features: List[str], categorical_features: List[str]):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.preprocessor = self._build_pipeline()

    def _build_pipeline(self) -> ColumnTransformer:
        """Builds the sklearn preprocessing pipeline."""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for dense tensor
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            verbose_feature_names_out=False # cleaner feature names
        )
        return preprocessor

    def fit(self, df: pd.DataFrame) -> 'TabularAdapter':
        """Fits the preprocessor on the dataframe."""
        self.preprocessor.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """Transforms the dataframe and returns a PyTorch tensor."""
        X_processed = self.preprocessor.transform(df)
        return torch.tensor(X_processed, dtype=torch.float32)
    
    def get_feature_names(self) -> List[str]:
        """Returns the list of feature names after transformation."""
        if hasattr(self.preprocessor, 'get_feature_names_out'):
             return list(self.preprocessor.get_feature_names_out())
        return []


class BRFSSDiabetesAdapter(TabularAdapter):
    """
    Specific adapter for the BRFSS Diabetes dataset.
    """
    def __init__(self):
        # Define features based on the dataset structure
        # Numerical features
        numeric_features = [
            'BMI', 'MentHlth', 'PhysHlth'
        ]
        
        # Categorical features
        categorical_features = [
            'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
            'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
        
        super().__init__(numeric_features, categorical_features)
