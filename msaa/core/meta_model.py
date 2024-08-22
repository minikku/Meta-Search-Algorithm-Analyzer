import pandas as pd


class MetaModel:

    def __init__(self, predictor: object, training_data: object):
        """
        predictor is instance of sklearn regressor

        training_data is an object of the dataset for the model training
        """

        self.predictor = predictor
        self.training_data = training_data


    def get_training_data(self):
        return self.training_data


    def fit_model(self, input_features: list, output_feature: str):
        """
        Format data to fit the model
        """

        x = self.training_data[input_features]
        y = self.training_data[output_feature]
        self.predictor.fit(x, y)

    def make_prediction(self, x_input):
        return self.predictor.predict(x_input)
