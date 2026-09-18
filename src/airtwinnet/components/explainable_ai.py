import numpy as np


class ExplainableAI:

    def __init__(self, feature_names):

        if not feature_names:
            raise ValueError(
                "Feature names cannot be empty."
            )

        self.feature_names = feature_names

    def calculate_feature_contribution(
        self,
        feature_values,
        feature_weights
    ):

        feature_values = np.asarray(
            feature_values,
            dtype=float
        )

        feature_weights = np.asarray(
            feature_weights,
            dtype=float
        )

        if feature_values.ndim != 2:
            raise ValueError(
                "Feature values must be a 2D array."
            )

        if feature_weights.ndim != 1:
            raise ValueError(
                "Feature weights must be a 1D array."
            )

        if feature_values.shape[1] != len(
            self.feature_names
        ):
            raise ValueError(
                "Feature count does not match feature names."
            )

        if len(feature_weights) != len(
            self.feature_names
        ):
            raise ValueError(
                "Weight count does not match feature names."
            )

        contributions = (
            feature_values * feature_weights
        )

        return contributions

    def get_feature_importance(
        self,
        feature_values,
        feature_weights
    ):

        contributions = self.calculate_feature_contribution(
            feature_values,
            feature_weights
        )

        importance = np.mean(
            np.abs(contributions),
            axis=0
        )

        total = importance.sum()

        if total > 0:
            importance = importance / total

        return dict(
            zip(
                self.feature_names,
                importance
            )
        )

    def explain_model(
        self,
        model,
        feature_names=None
    ):

        if feature_names is None:
            feature_names = self.feature_names

        if len(feature_names) != len(
            self.feature_names
        ):
            raise ValueError(
                "Feature names count mismatch."
            )

        explanations = {}

        for index, target in enumerate(
            ["pm2_5", "pm10"]
        ):

            estimator = model.estimators_[index]

            importance = estimator.feature_importances_

            feature_importance = dict(
                zip(
                    feature_names,
                    importance
                )
            )

            sorted_features = sorted(
                feature_importance.items(),
                key=lambda item: item[1],
                reverse=True
            )

            dominant_feature = sorted_features[0]

            explanations[target] = {
                "feature_importance": feature_importance,
                "dominant_feature": dominant_feature[0],
                "dominant_importance": round(
                    float(dominant_feature[1]),
                    4
                )
            }

        return explanations

    def explain_separate_models(
        self,
        pm25_model,
        pm10_model,
        feature_names
    ):

        if len(feature_names) != len(
            self.feature_names
        ):
            raise ValueError(
                "Feature names count mismatch."
            )

        explanations = {}

        models = {
            "pm2_5": pm25_model,
            "pm10": pm10_model
        }

        for target, model in models.items():

            importance = model.feature_importances_

            feature_importance = dict(
                zip(
                    feature_names,
                    importance
                )
            )

            sorted_features = sorted(
                feature_importance.items(),
                key=lambda item: item[1],
                reverse=True
            )

            dominant_feature = sorted_features[0]

            explanations[target] = {
                "feature_importance": feature_importance,
                "dominant_feature": dominant_feature[0],
                "dominant_importance": round(
                    float(dominant_feature[1]),
                    4
                )
            }

        return explanations

    def generate_explanation(
        self,
        model
    ):

        explanations = self.explain_model(model)

        pm25_feature = explanations[
            "pm2_5"
        ]["dominant_feature"]

        pm10_feature = explanations[
            "pm10"
        ]["dominant_feature"]

        return (
            f"PM2.5 prediction is mainly influenced "
            f"by {pm25_feature}, while PM10 prediction "
            f"is mainly influenced by {pm10_feature}."
        )