from abc import ABC, abstractmethod

import pandas as pd


class EnsembleModel(ABC):
    """
    Base class for dataframe-based ensemble models.

    Subclasses operate on prediction columns that already exist in a pandas
    DataFrame, for example `model_a_valence` and `model_a_arousal`.
    """

    def __init__(self, path, valence_suffix="_valence", arousal_suffix="_arousal"):
        self.valence_suffix = valence_suffix
        self.arousal_suffix = arousal_suffix
        self.dataframe = pd.read_csv(path)

    def _get_prediction_columns(self):
        valence_cols = [
            col for col in self.dataframe.columns if col.endswith(self.valence_suffix)
        ]
        arousal_cols = [
            col for col in self.dataframe.columns if col.endswith(self.arousal_suffix)
        ]

        if not valence_cols:
            raise ValueError(
                f"No valence prediction columns found with suffix '{self.valence_suffix}'."
            )
        if not arousal_cols:
            raise ValueError(
                f"No arousal prediction columns found with suffix '{self.arousal_suffix}'."
            )
        if len(valence_cols) != len(arousal_cols):
            raise ValueError(
                "Found a different number of valence and arousal prediction columns."
            )

        if self.dataframe[valence_cols + arousal_cols].isna().any().any():
            raise ValueError(
                "Prediction columns contain null values."
            )

        return sorted(valence_cols), sorted(arousal_cols)

    @abstractmethod
    def predictions(self):
        """
        Return ensemble predictions : pred_v, pred_a, gold_v, gold_a
        """


class AverageEnsemble(EnsembleModel):
    """
    Average all model prediction columns for valence and arousal separately.
    """

    def predictions(self):
        valence_cols, arousal_cols = self._get_prediction_columns()

        pred_v = self.dataframe[valence_cols].mean(axis=1).to_numpy()
        pred_a = self.dataframe[arousal_cols].mean(axis=1).to_numpy()
        gold_v = self.dataframe["Valence"].to_numpy()
        gold_a = self.dataframe["Arousal"].to_numpy()

        return pred_v, pred_a, gold_v, gold_a

