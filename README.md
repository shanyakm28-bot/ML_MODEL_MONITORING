# 📉 Customer Churn Prediction with ML Monitoring

A production-ready machine learning system for customer churn prediction with built-in monitoring, drift detection, and real-time inference capabilities.

## 🚀 Features

- **End-to-End ML Pipeline**: Complete workflow from training to deployment
- **Real-Time Predictions**: FastAPI-powered REST API for instant predictions
- **Data Drift Detection**: Automated monitoring for model performance degradation
- **Interactive Web UI**: Streamlit-based frontend for easy interaction
- **Prediction Logging**: Comprehensive monitoring and audit trail
- **Production Ready**: Scalable architecture with proper error handling

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│   FastAPI API   │───▶│   ML Model      │
│   (Frontend)    │    │   (Backend)     │    │   (Inference)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Drift Monitor  │
                       │  & Logging      │
                       └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- pip package manager

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ml-monitoring
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if model.pkl doesn't exist)
   ```bash
   python train.py
   ```

## 🚀 Quick Start

### 1. Start the FastAPI Backend

```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Launch the Streamlit Frontend

```bash
streamlit run app.py
```

### 3. Access the Application

- **Web UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000

## 📊 API Endpoints

### `POST /predict`

Make churn predictions for customers.

**Request Body:**

```json
{
  "CustomerID": 1001,
  "Age": 35,
  "Gender": "Male",
  "Tenure": 24,
  "Usage_Frequency": 15,
  "Support_Calls": 2,
  "Payment_Delay": 5,
  "Subscription_Type": "Premium",
  "Contract_Length": "Yearly",
  "Total_Spend": 2500.0,
  "Last_Interaction": 3
}
```

**Response:**

```json
{
  "Churn_Prediction": 0,
  "Churn_Probability": 0.234,
  "Drift_Detected": false,
  "Drifted_Features": [],
  "Meaning": "Customer will NOT churn"
}
```

### `GET /stats`

Retrieve prediction statistics and monitoring data.

### `GET /`

Health check endpoint.

## 🔍 Monitoring & Drift Detection

The system automatically monitors for data drift by comparing incoming features against training data statistics:

- **Drift Threshold**: Configurable sensitivity (default: 10x deviation)
- **Feature Tracking**: Monitors all numerical and categorical features
- **Alert System**: Real-time drift notifications in UI
- **Audit Trail**: Complete prediction logging in `predictions.log`

## 📁 Project Structure

```
ml-monitoring/
├── app/
│   ├── __init__.py          # Package initialization
│   └── main.py              # FastAPI application
├── app.py                   # Streamlit frontend
├── train.py                 # Model training pipeline
├── churn.csv               # Training dataset
├── model.pkl               # Trained model artifacts
├── predictions.log         # Prediction monitoring logs
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── LICENSE                # MIT License
└── README.md              # Project documentation
```

## 🧠 Model Details

- **Algorithm**: Logistic Regression
- **Features**: Customer demographics, usage patterns, and engagement metrics
- **Target**: Binary churn classification (0: No Churn, 1: Churn)
- **Preprocessing**: One-hot encoding for categorical variables
- **Evaluation**: Accuracy-based performance metrics

## 📈 Usage Examples

### Python API Client

```python
import requests

payload = {
    "CustomerID": 1001,
    "Age": 35,
    "Gender": "Female",
    "Tenure": 12,
    "Usage_Frequency": 8,
    "Support_Calls": 1,
    "Payment_Delay": 0,
    "Subscription_Type": "Standard",
    "Contract_Length": "Monthly",
    "Total_Spend": 1200.0,
    "Last_Interaction": 7
}

response = requests.post("http://localhost:8000/predict", json=payload)
result = response.json()
print(f"Churn Risk: {result['Churn_Probability']}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"CustomerID": 1001, "Age": 35, "Gender": "Male", "Tenure": 24, "Usage_Frequency": 15, "Support_Calls": 2, "Payment_Delay": 5, "Subscription_Type": "Premium", "Contract_Length": "Yearly", "Total_Spend": 2500.0, "Last_Interaction": 3}'
```

## 🔧 Configuration

### Drift Detection Sensitivity

Modify the drift threshold in `app/main.py`:

```python
drifted_features = detect_drift(df, feature_means, threshold=10)  # Adjust threshold
```

### Model Retraining

To retrain with new data:

1. Update `churn.csv` with new training data
2. Run `python train.py`
3. Restart the FastAPI service

## 🚨 Monitoring Alerts

The system provides several monitoring capabilities:

- **Real-time Drift Detection**: Immediate alerts when input data deviates significantly
- **Prediction Logging**: All predictions logged with timestamps for audit
- **Performance Tracking**: Monitor prediction patterns over time
- **Feature Drift Analysis**: Identify which specific features are drifting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:

- Create an issue in the repository
- Check the API documentation at `/docs` endpoint
- Review the prediction logs in `predictions.log`

## 🔮 Future Enhancements

- [ ] Advanced drift detection algorithms (KS test, PSI)
- [ ] Model performance monitoring dashboard
- [ ] Automated model retraining pipeline
- [ ] A/B testing framework
- [ ] Integration with MLOps platforms
- [ ] Real-time streaming predictions
- [ ] Advanced feature importance tracking
