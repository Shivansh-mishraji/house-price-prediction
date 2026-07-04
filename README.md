# My First FastAPI Machine Learning Project: California House Price Predictor

Hey! Welcome to my practice machine learning API project. I built this completely from scratch to learn how backend APIs connect to real Machine Learning models using Python, FastAPI, and Scikit-Learn.

It uses a `RandomForestRegressor` trained on the famous California Housing dataset to predict home values based on location, age, size, and income.

---

## 🧠 What I Learned (The Journey)
I started with basic data analysis scripts, but I learned a ton of backend and MLOps concepts along the way:

*   **Pydantic Models with Validation:** I learned how to define the exact shape of incoming house feature data and used Pydantic's `Field` (like `gt=0`, `ge=32.0`, `le=43.0`) to validate inputs like California latitude/longitude boundaries.
*   **Encapsulating ML Models:** I figured out how to train a model in a separate script (`train.py`) and serialize it using `joblib` so that the FastAPI app (`main.py`) can load it instantly on startup.
*   **hot-Reloading (MLOps):** I built a `/upload-model` file upload endpoint using `python-multipart` to dynamically swap the trained model in memory. This means the server can run 24/7 without needing restarts to deploy new models!
*   **Batch Predictions:** I learned how to process `.csv` file uploads, parse them into a pandas DataFrame, and run batch predictions on multiple rows of data at once.
*   **User-Friendly API Responses:** I learned that raw floats (like `350964.0699`) are hard to read. I formatted values to clean currency strings and calculated realistic expected price ranges (confidence intervals) using Root Mean Squared Error (RMSE).

---

## 😅 Mistakes I Made (And Fixed!)
You can't learn without breaking things. Here are a few things that tripped me up, but I managed to conquer:

*   **The Escape Sequence Trap:** I ran into JSON syntax errors in `.vscode/settings.json` because of Windows backslashes (like `\91727` in my path). I fixed it by switching to double backslashes and using portable `${workspaceFolder}` variables.
*   **Function Signatures vs. Execution:** I accidentally printed `df.head` and `df.describe` in my python explore script without parentheses `()`, which printed the method descriptions instead of executing the functions.
*   **Dataset Download Delay:** My first data exploration script seemed to hang. I learned that `fetch_california_housing()` has to download the dataset from the internet on its first run before caching it locally.
*   **CORS Blockers:** I learned that modern browsers block cross-origin API calls by default, and I resolved it by setting up `CORSMiddleware` in FastAPI so my future frontend can connect smoothly.

---

## ✨ What I Did Best
*   **Designing for the User:** Instead of returning raw ML statistics like MSE and $R^2$ scores to non-technical users, I translated them into simple text ("High 80.5% accuracy rating") and clear price ranges.
*   **Virtual Environments:** I successfully set up a fast, clean virtual environment (`.venv`) using `uv` to keep my dependencies isolated and resolved IDE interpreter configuration path errors.

---

## 🛠️ Features (Fully Functional!)
*   `GET /` - Welcome page and endpoints map
*   `GET /health` - API server and model loading health check
*   `POST /predict` - Predict price for a single house (takes parameters like sqft, income, bedrooms, age)
*   `POST /predict-csv` - Upload a CSV of multiple houses for batch predictions
*   `POST /upload-model` - Upload a new `.joblib` model file to hot-swap the model in-memory 24/7

---

## 🏃 How to run this
1. Make sure you have your virtual environment active and dependencies installed:
   ```bash
   uv sync
   ```
2. Run the FastAPI server:
   ```bash
   uv run uvicorn main:app --reload
   ```
3. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to interactively test all endpoints using Swagger!

This is just the beginning. Next stop: connecting a real database and building a stunning frontend UI! 🚀
