import pandas as pd


class Preprocessor:
    def __init__(self, preprocessor_model, kwargs):
        self.preprocessor = preprocessor_model(**kwargs) if preprocessor_model else None

    def preprocess(self, df, is_fit=False):
        """Transform input data.

        Returns
        -------
        tuple
            Transformed DataFrame.
        """
        if self.preprocessor:
            if is_fit:
                self.preprocessor.fit(df)
            df = pd.DataFrame(
                self.preprocessor.transform(df),
                index=df.index,
                columns=df.columns,
            )

        return df
