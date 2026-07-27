import logging

logger = logging.getLogger(__name__)

import joblib
import os
import pandas as pd
import shutil
from fastapi import FastAPI , HTTPException, UploadFile, File
from pydantic import BaseModel , Field

app = FastAPI()

model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")

# input schema
class HouseFeatures(BaseModel):
    MedInc: float = Field(
        gt=0, 
        description="Median income in block group (in tens of thousands of USD, e.g. 3.5 = $35,000)"
    )
    HouseAge: float = Field(
        gt=0, 
        description="Median age of houses in block group"
    )
    AveRooms: float = Field(
        gt=0, 
        description="Average number of rooms per household"
    )
    AveBedrms: float = Field(
        gt=0, 
        description="Average number of bedrooms per household"
    )
    Population: float = Field(
        gt=0, 
        description="Block group population"
    )
    AveOccup: float = Field(
        gt=0, 
        description="Average number of household members (occupancy)"
    )
    Latitude: float = Field(
        ge=32.0, 
        le=43.0, 
        description="Block group latitude (California latitude boundaries)"
    )
    Longitude: float = Field(
        ge=-125.0, 
        le=-113.0, 
        description="Block group longitude (California longitude boundaries)"
    )

@app.get("/")
def home(): -> None:
    return{
        "Message":"Calefornia House Prediction API",
        "Status Code": "running",
        "endpoint":"sent post request to /predict"
    }

@app.get("/health")
def health():
    return {
        "status":"running",
        "model":"RandomForestRegressor",
        "features":features,
        "avg_error": "$39,000"
    }

# prediction
@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        # 1. Convert the incoming schema to a pandas DataFrame
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])
        
        # 2. Make the prediction
        prediction = model.predict(input_data)[0]
        
        # 3. Scale the price (target is in $100,000s in California Housing)
        actual_price = float(prediction) * 100000.0
        
        # Model evaluation metrics (based on your training results)
        r2_score = 0.805  # ~80.5% variance explained
        mae = 33000.0    # Mean Absolute Error
        rmse = 50000.0   # Root Mean Squared Error (used for confidence range)
        mse = 0.25       # Mean Squared Error
        
        # Calculate confidence range using RMSE
        lower_bound = max(10000.0, actual_price - rmse)
        upper_bound = actual_price + rmse
        
        # 4. Return a highly readable, human-friendly result
        return {
            "status": "success",
            "estimated_market_value": f"${actual_price:,.0f}",
            "expected_price_range": f"Between ${lower_bound:,.0f} and ${upper_bound:,.0f}",
            "pricing_confidence": f"High ({r2_score * 100:.1f}% accuracy rating)",
            "average_margin_of_error": f"+/- ${rmse:,.0f}",
            "simple_summary": (
                f"Based on our AI analysis of similar properties in California, this house has an estimated "
                f"market value of ${actual_price:,.0f}. Given normal market fluctuations, the price is highly "
                f"likely to fall between ${lower_bound:,.0f} and ${upper_bound:,.0f}."
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Model hot-reload endpoint: upload new house_model.joblib
@app.post("/upload-model")
def upload_model(file: UploadFile = File(...)):
    global model
    if not file.filename.endswith('.joblib'):
        raise HTTPException(status_code=400, detail="Only .joblib files are supported")
    try:
        # Save file to disk
        temp_file_path = "house_model_temp.joblib"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Try loading it first to verify it's a valid joblib file and model
        temp_model = joblib.load(temp_file_path)
        
        # If valid, replace old model file and reload in memory
        if os.path.exists("house_model.joblib"):
            os.remove("house_model.joblib")
        os.rename(temp_file_path, "house_model.joblib")
        
        model = temp_model
        return {
            "status": "success",
            "message": "Model updated and reloaded in memory successfully without restarting the server!"
        }
    except Exception as e:
        if os.path.exists("house_model_temp.joblib"):
            os.remove("house_model_temp.joblib")
        raise HTTPException(status_code=500, detail=f"Failed to load model file: {str(e)}")

# Batch predict endpoint: upload CSV file and get predictions back
@app.post("/predict-csv")
def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        # Read the uploaded CSV
        df_input = pd.read_csv(file.file)
        
        # Verify required features exist in the uploaded CSV
        missing_features = [f for f in features if f not in df_input.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"CSV file is missing required columns: {missing_features}"
            )
            
        # Get predictions
        predictions = model.predict(df_input[features])
        
        # Format output
        results = []
        for i, pred in enumerate(predictions):
            actual_price = float(pred) * 100000.0
            lower_bound = max(10000.0, actual_price - 50000.0) # Using $50,000 RMSE
            upper_bound = actual_price + 50000.0
            
            results.append({
                "row_index": i + 1,
                "estimated_market_value": f"${actual_price:,.0f}",
                "expected_price_range": f"Between ${lower_bound:,.0f} and ${upper_bound:,.0f}"
            })
            
        return {
            "status": "success",
            "total_predictions": len(results),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

