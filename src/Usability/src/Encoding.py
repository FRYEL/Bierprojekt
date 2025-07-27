from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

def one_hot_encode_dataframe(df, categorical_cols, encoder=None, return_encoder=False):
    """
    One-hot encodes the specified categorical columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        categorical_cols (list): List of column names to one-hot encode
        encoder (ColumnTransformer or None): If provided, uses existing encoder (e.g. for test set)
        return_encoder (bool): If True, returns the encoder as second return value

    Returns:
        pd.DataFrame: One-hot encoded DataFrame
        (optional) ColumnTransformer: Fitted encoder
    """

    if encoder is None:
        encoder = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols)
            ],
            remainder='passthrough'
        )
        encoder.fit(df)

    transformed_array = encoder.transform(df)
    cat_feature_names = encoder.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    passthrough_cols = [col for col in df.columns if col not in categorical_cols]
    all_columns = list(cat_feature_names) + passthrough_cols

    encoded_df = pd.DataFrame(transformed_array, columns=all_columns, index=df.index)

    if return_encoder:
        return encoded_df, encoder
    else:
        return encoded_df


def clean_alter_column(df):
    df = df.copy()
    if "alter" in df.columns:
        df["alter"] = df["alter"].astype(str).str.strip().str.lower()
        df["alter"] = df["alter"].apply(lambda x: x if x.isnumeric() else np.nan)
        df["alter"] = pd.to_numeric(df["alter"], errors="coerce")
        df["alter"] = df["alter"].fillna(df["alter"].median())
    return df
