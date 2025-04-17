from metaflow import FlowSpec, step
import mlflow
import pandas as pd
import dataprocessing

class ScoringFlow(FlowSpec):

    @step
    def start(self):
        self.df_new = pd.read_csv('../../data/bagpack/test.csv')
        self.next(self.process_data)

    @step
    def process_data(self):
        self.X_new, _ = dataprocessing.feature_transform(self.df_new)
        self.next(self.load_model)

    @step
    def load_model(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        model_uri = "models:/BagpackBestModel/latest"
        self.model = mlflow.sklearn.load_model(model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        self.predictions = self.model.predict(self.X_new)
        self.next(self.output_predictions)

    @step
    def output_predictions(self):
        results_df = pd.DataFrame(self.predictions, columns=['prediction'])
        results_df.to_csv('../../data/predictions_lab6.csv', index=False)
        print("Predictions saved successfully.")
        self.next(self.end)

    @step
    def end(self):
        print("Scoring flow completed.")

if __name__ == '__main__':
    ScoringFlow()

