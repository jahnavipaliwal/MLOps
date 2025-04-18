from metaflow import FlowSpec, step, Parameter, conda_base, kubernetes, resources, timeout, retry, catch
import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@conda_base(
    libraries={
        "pandas": "1.5.3",
        "scikit-learn": "1.2.2",
        "mlflow": "2.3.2"
    },
    python="3.9.16"
)
class ScoringFlowGCP(FlowSpec):

    @step
    def start(self):
        self.df_new = pd.read_csv('../../data/bagpack/test.csv')
        self.next(self.process_data)

    @catch(var="error")
    @retry(times=2)
    @timeout(seconds=300)
    @step
    def process_data(self):
        self.X_new, _ = self.feature_transform(self.df_new)
        self.next(self.load_model)

    @kubernetes(cpu=2, memory=4000)
    @resources(cpu=2, memory=4000)
    @retry(times=3)
    @timeout(seconds=600)
    @step
    def load_model(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Replace if MLflow is remote
        model_uri = "models:/BagpackBestModel/latest"
        self.model = mlflow.sklearn.load_model(model_uri)
        self.next(self.predict)

    @retry(times=3)
    @timeout(seconds=180)
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
        print("Scoring flow completed successfully.")

if __name__ == '__main__':
    ScoringFlowGCP()
