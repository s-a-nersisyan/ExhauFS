import pandas as pd


class Preprocessor:
    def __init__(self, preprocessor_model, kwargs):
        self.preprocessor = preprocessor_model(**kwargs) if preprocessor_model else None

    def preprocess(self, data, is_fit=False):
        """Transform input data.

        Returns
        -------
        tuple
            Transformed DataFrame.
        """
        if self.preprocessor:
            if is_fit:
                self.preprocessor.fit(data)
            if isinstance(data, pd.DataFrame):
                data = pd.DataFrame(
                    self.preprocessor.transform(data),
                    index=data.index,
                    columns=data.columns,
                )
            else:
                data = self.preprocessor.transform(data)

        return data
