from metaflow import FlowSpec, step, Parameter
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import dataprocessing

class TrainingFlow(FlowSpec):

    seed = Parameter('seed', default=20, help='Random seed for reproducibility')

    @step
    def start(self):
        self.df = pd.read_csv('../../data/bagpack/train.csv')
        self.df = self.df.dropna()
        print("Data Sample:")
        print(self.df.head())
        print("\nData Types:")
        print(self.df.dtypes)
        self.next(self.process_data)

    @step
    def process_data(self):
        # Modular data processing step
        self.X, self.y = dataprocessing.feature_transform(self.df)
        self.next(self.tune_model)

    @step
    def tune_model(self):
        # Hyperparameter tuning
        model = RandomForestRegressor(random_state=self.seed)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
        grid_search = GridSearchCV(model, param_grid, cv=3)
        grid_search.fit(self.X, self.y)
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        print(f"Best params: {self.best_params}")
        self.next(self.register_model)

    @step
    def register_model(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Bagpack_Model_Training")

        with mlflow.start_run():
            mlflow.log_params(self.best_params)
            mlflow.sklearn.log_model(self.best_model, "model")
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                "BagpackBestModel"
            )
        print("Model registered successfully.")
        self.next(self.end)

    @step
    def end(self):
        print("Training flow completed successfully.")

if __name__ == '__main__':
    TrainingFlow()
