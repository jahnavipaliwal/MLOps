from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

app = FastAPI()
tracking_url = "sqlite:////Users/jahnavipaliwal/Desktop/ML_Ops/MLOps/Labs/mlflow.db"
run_id = "25ef3cddceaf4000942dcdc40ebf2d72"

class RequestBody(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float = Field(..., alias="od280/od315_of_diluted_wines")
    proline: float

@app.on_event("startup")
def load_model():
    global model

    client = MlflowClient()
    for exp in client.search_experiments():
        runs = client.search_runs([exp.experiment_id])
        for r in runs:
            artifacts = client.list_artifacts(r.info.run_id)
            for a in artifacts:
                if a.path == "model":
                    print(f"Run {r.info.run_id} has a logged model")

    mlflow.set_tracking_uri(tracking_url)
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

@app.post("/predict")
def predict(data: RequestBody):
    input_dict = data.dict(by_alias=True)
    prediction = model.predict([input_dict])
    return {"prediction": prediction.tolist()}

