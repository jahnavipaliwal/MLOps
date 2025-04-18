from metaflow import FlowSpec, step, Parameter, conda_base, kubernetes, resources, timeout, retry, catch
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@conda_base(
    python="3.9.16",
    libraries={
        "pandas": "1.4.4",
        "scikit-learn": "1.2.2",
        "mlflow": "2.11.1",
        "fsspec": "2023.6.0", 
        "gcsfs": "2023.6.0"  
    }
)
class TrainingFlow(FlowSpec):

    seed = Parameter('seed', default=20, help='Random seed for reproducibility')

    @step
    def start(self):
        self.df = pd.read_csv("gs://storage-my-ds-lab-metaflow-default/data/train.csv")
        self.df = self.df.dropna()
        print("Data Sample:")
        print(self.df.head())
        print("\nData Types:")
        print(self.df.dtypes)
        self.next(self.process_data)

    @catch(var="error")
    @retry(times=2)
    @timeout(seconds=300)
    @step
    def process_data(self):
        # Modular data processing step
        self.X, self.y = self.feature_transform(self.df)
        self.next(self.tune_model)

    @kubernetes(cpu=2, memory=4000)
    @resources(cpu=2, memory=4000)
    @retry(times=3)
    @timeout(seconds=600)
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

    @retry(times=3)
    @timeout(seconds=180)
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

    
    def feature_transform(self, df):
        # Drop ID as it's not useful
        df = df.drop(columns=["id"])

        if "Price" in df.columns:
            X = df.drop("Price", axis=1)
            y = df["Price"]
        else:
            X = df
            y = None

        # Identify numeric and categorical columns
        numeric_features = ["Compartments", "Weight Capacity (kg)"]
        categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]

        # Preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Fit and transform
        X_processed = preprocessor.fit_transform(X)

        return X_processed, y

    @step
    def end(self):
        print("Training flow completed successfully.")

if __name__ == '__main__':
    TrainingFlow()