import pandas as pd


class Preprocessor:
    def __init__(self, df, ann, preprocessor_model, kwargs):
        self.df = df
        self.ann = ann

        self.preprocessor = preprocessor_model
        self.preprocessor_kwargs = kwargs

    def preprocess(self, df, preprocessor=None):
        """Transform input data by passed preprocessor.

        Returns
        -------
        tuple
            Transformed DataFrame and its preprocessor model.
        """
        if preprocessor:
            index = df.index
            columns = df.columns
            df = pd.DataFrame(
                preprocessor.transform(df),
                index=index,
                columns=columns,
            )
        elif self.preprocessor:
            index = df.index
            columns = df.columns
            preprocessor = self.preprocessor(**self.preprocessor_kwargs)
            preprocessor.fit(df)
            df = pd.DataFrame(
                preprocessor.transform(df),
                index=index,
                columns=columns,
            )

        return df, preprocessor
