# DealSight AI

**AI-Powered Sales Deal Intelligence and Prediction Assistant**

> DealSight AI remembers customer interactions, predicts what is likely to happen with a deal, explains why, and recommends the next action.

---

## 📌 Problem

Enterprise sales representatives routinely juggle dozens of active deals simultaneously. Before every critical call or follow-up, sales reps must manually reconstruct the deal context by reviewing:
* Notes and logs from previous calls
* Past customer objections and pricing friction
* Competing vendors under customer evaluation
* Shifting customer sentiment and engagement velocity
* Current deal stage and overdue action items

Because these critical signals remain scattered across unstructured conversation transcripts and CRM notes, reps struggle to quickly answer three fundamental questions:
1. **What is happening with this deal?**
2. **Is it likely to move forward or stall?**
3. **What specific action should the salesperson take next?**

---

## 💡 Solution

**DealSight AI** is a sales decision assistant designed around one clear user story:

$$\text{\textbf{WHAT WILL HAPPEN?}} \longrightarrow \text{\textbf{WHY?}} \longrightarrow \text{\textbf{WHAT SHOULD I DO NEXT?}}$$

DealSight bridges customer memory with statistical machine learning:
* **Hindsight Memory** preserves long-term, customer-specific episodic context across calls.
* **Supervised Machine Learning** (trained on 2,500 real B2B deals) predicts progression and loss probabilities.
* **SHAP (Shapley Additive exPlanations)** decomposes predictions into plain-English positive drivers and risk factors.
* **Groq LLM** synthesizes the prediction, SHAP attributions, and customer memory into a concise, actionable next step for the sales representative.

---

## 🌟 Key Features

### 👤 Customer Intelligence
* **Hindsight Long-Term Customer Memory**: Persistent episodic memory across customer touchpoints.
* **Customer Profiles & Context**: Track company name, industry, size, and deal value.
* **Call Timeline & History**: Step-by-step chronology of past conversations and sentiment trends.
* **Friction & Objection Tracking**: Flags unresolved pricing objections, timeline delays, and competitor threats.

### 🤖 AI Assistance
* **Groq-Powered Pre-Call Brief**: Concise executive briefing generated before every customer interaction.
* **Context-Aware Recommendations**: Tailored playbooks synthesized from historical objections and winning strategies.
* **Plain-Language Translations**: Mathematical attributions translated into natural sales signals.

### 📊 Data Science & Machine Learning
* **Historical Sales Dataset**: 2,500 structured enterprise B2B sales records (`data/sales_data.csv`).
* **Data Cleaning & Preprocessing**: Deduplication, median imputation, standard scaling, and one-hot encoding without target leakage.
* **Feature Engineering**: Derives sales momentum, objection pressure, decision-maker engagement, and response health.
* **Multi-Model Evaluation**: Stratified 70/15/15 train/validation/test split evaluating Logistic Regression, Random Forest, and XGBoost.
* **Production Calibration**: Optimized for high ROC-AUC and maximum Recall on Lost Deals to catch slipping deals early.
* **Predictive Pipeline**: Complete scikit-learn `Pipeline` bundled into `models/deal_prediction_model.pkl`.

### 🧠 Explainability (SHAP)
* **Mathematical Attribution**: Exact linear decomposition (`LinearExplainer`) for calibrated production models.
* **Positive Signals**: Highlights key drivers moving the deal forward (e.g., completed demos, executive involvement).
* **Risk Factors**: Identifies deal friction points (e.g., unanswered pricing concerns, competitor evaluation).
* **Judge Details Drawer**: Full transparency for judges and ML engineers with exact Shapley values and feature weights.

### 🔮 Predictive Intelligence & Deal Forecast
* **Real Model Probabilities**: Progression and loss percentages generated directly from `predict_proba()`.
* **Dynamic Deal Health**: Automatically flags deals as *Healthy Deal*, *Needs Attention*, or *At Risk*.
* **Next Likely Milestone**: Multi-class transition model (`models/next_stage_model.pkl`) predicting the upcoming milestone with confidence score.
* **Interactive What-If Sandbox**: Live toggles to simulate changing buyer conditions (executive involvement, objections, response delay).

### 📈 Sales Insights Dashboard
* **Dynamic Dataset Metrics**: Total deals (2,500), outcome distribution (39% won / 61% lost), average calls (7.6), objections (1.0), and follow-ups (5.0).
* **4 Business Charts**:
  1. *Customer Interest vs Deal Outcome*
  2. *Pricing & Objections vs Deal Outcome*
  3. *Conversation Activity vs Deal Outcome*
  4. *Deal Stage vs Outcome*

---

## 🏗️ Architecture

```text
                    DealSight AI
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
   Hindsight Memory              Historical Data
   (Customer Context)          (2,500 B2B Sales Deals)
          │                             │
          ▼                             ▼
    Call History                 Data Processing
  (Timeline & Notes)       (Cleaning & Leakage Prevention)
          │                             │
          │                    Feature Engineering
          │                 (Velocity, Pressure, Health)
          │                             │
          │                         ML Models
          │                 (LogReg, RF, XGBoost)
          │                             │
          │                        Best Model
          │                   (Calibrated Pipeline)
          │                             │
          └───────────► Prediction ◄────┘
                       (predict_proba)
                              │
                              ▼
                         SHAP Explain
                       (LinearExplainer)
                              │
                              ▼
                      Key Driving Factors
                      (Positive & Concerns)
                              │
                              ▼
                          Groq LLM
                   (Synthesis & Guardrails)
                              │
                              ▼
                      Recommended Action
                              │
                              ▼
                         DealSight UI
             (Hero Forecast + Sales Intelligence)
```

---

## ⚙️ How It Works

1. **Customer Interaction Logged**: Notes, quotes, objections, and sentiment from calls are recorded.
2. **Hindsight Retains Memory**: Key customer details and previous friction points are stored in Hindsight memory.
3. **Data Cleaning & Engineering**: Signals such as objection pressure, response latency, and decision-maker involvement are engineered without target leakage.
4. **Model Predicts Outcome**: The calibrated machine-learning model computes the exact probability of progression versus loss using `predict_proba()`.
5. **Next Milestone Estimated**: The multi-class transition model predicts the most likely next pipeline stage (e.g., *Negotiation* with 78% confidence).
6. **SHAP Calculates Factor Impact**: SHAP attributes the specific positive factors driving the deal forward and the exact risk concerns creating friction.
7. **Customer Context Retrieved**: Relevant historical interactions and past successful objection resolutions are recalled from Hindsight.
8. **Groq Synthesizes Actionable Guidance**: Groq translates the structured ML prediction, SHAP drivers, and customer context into a concise, salesperson-friendly action recommendation.
9. **Sales Rep Executes**: The salesperson opens DealSight, immediately understands deal health in 5 seconds, and prepares a tailored follow-up.

---

## 🛠️ Technology Stack

### Frontend
* **React 18**: Component-driven UI architecture
* **Vite 6**: Fast build tool and development server
* **JavaScript (ES Modules)**: Application logic and state management
* **Tailwind CSS & Vanilla CSS**: Custom responsive design system with dark-mode aesthetic
* **Framer Motion**: Smooth micro-animations and transition states
* **Lucide React**: Consistent iconography

### Backend
* **Python 3.10+ / 3.14**: Backend runtime
* **FastAPI**: Asynchronous high-performance REST API
* **Uvicorn**: ASGI web server
* **Pydantic v2**: Strict schema validation and data parsing
* **python-dotenv**: Environment configuration management

### AI / Machine Learning
* **Scikit-learn**: Data preprocessing, ColumnTransformer, Logistic Regression, Random Forest, and model pipelines
* **XGBoost**: Gradient-boosted decision trees benchmark
* **SHAP**: Shapley Additive exPlanations for model explainability
* **Pandas & NumPy**: Tabular data manipulation and mathematical vector operations
* **Joblib**: Model serialization and persistence

### External AI Services
* **Hindsight**: Long-term episodic memory engine for customer interaction retention
* **Groq Cloud API**: High-speed LLM inference (`openai/gpt-oss-120b`) for plain-English synthesis

---

## 📂 Project Structure

```text
DealMind/
│
├── backend/
│   ├── main.py                     # FastAPI application endpoints & routing
│   ├── requirements.txt            # Python dependencies (UTF-8)
│   ├── .env.example                # Backend environment template
│   ├── services/
│   │   ├── ai_service.py           # Groq LLM integration
│   │   ├── hindsight_service.py    # Hindsight memory retention & recall
│   │   ├── prediction_service.py   # Unified ML prediction, SHAP & Groq synthesis
│   │   └── explainability_service.py# SHAP attribution service
│   ├── ml/
│   │   ├── train.py                # Multi-model training harness
│   │   ├── evaluate.py             # Holdout test set evaluation
│   │   ├── train_next_stage.py     # Multi-class milestone transition model
│   │   ├── data_analysis.py        # Dataset statistical insights & chart generation
│   │   └── model_results.json      # Dynamic benchmark results
│   └── test_hindsight.py          # Standalone Hindsight connectivity test
│
├── frontend/
│   ├── index.html                  # HTML entry point with metadata
│   ├── package.json                # Frontend dependencies and npm scripts
│   ├── vite.config.js              # Vite configuration
│   ├── .env.example                # Frontend environment template
│   └── src/
│       ├── main.jsx                # React root mount
│       ├── App.jsx                 # Main navigation & 4-tab container
│       ├── index.css               # Global theme & CSS variables
│       └── components/
│           ├── Header.jsx          # Top brand bar & backend status badge
│           ├── CustomerSelector.jsx# Interactive customer switcher
│           ├── CallTimeline.jsx    # Chronological call history
│           ├── BriefCard.jsx       # Pre-call executive brief
│           ├── DealForecast.jsx    # Hero forecast prediction card
│           ├── PredictionReasons.jsx# SHAP positive and risk reasons
│           ├── RecommendedAction.jsx# Groq next-step action playbook
│           ├── PredictiveDashboard.jsx # Full forecast screen + What-If sandbox + Judge drawer
│           ├── DatasetDashboard.jsx# Sales Insights dataset KPIs & 4 business charts
│           ├── ModelPerformance.jsx # Model comparison table, ROC-AUC, confusion matrix
│           └── AddCallModal.jsx    # Modal for logging new customer calls
│
├── data/
│   ├── sales_data.csv              # 2,500 enterprise B2B historical sales records
│   └── mock_calls.json             # Structured customer conversation records
│
├── models/
│   ├── deal_prediction_model.pkl   # Calibrated production ML pipeline
│   ├── next_stage_model.pkl        # Multi-class pipeline for next milestone prediction
│   ├── metrics.json                # Validated holdout test metrics
│   ├── model_comparison.json       # Multi-model benchmarking comparison
│   └── evaluation_report.json      # Holdout confusion matrix and report
│
├── src/
│   ├── data_preprocessing.py       # Feature engineering & ColumnTransformer
│   ├── deal_features.py            # Extracts features from call history
│   ├── explainability.py           # Feature impact calculation
│   └── train_model.py              # Pipeline training script
│
├── screenshots/                    # UI screenshots
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file (excludes .env and venv)
```

---

## 🏆 Model Evaluation & Comparison

All models were evaluated using a stratified 70/15/15 train/validation/test split on 2,500 historical deals ($N = 375$ unseen holdout deals).

### Evaluation Results

| Model Architecture | Accuracy | Precision | Recall (Prog) | Recall (Lost / Risk) | F1-Score | ROC-AUC | Production Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Calibrated) | **85.6%** | **79.9%** | **84.3%** | **86.5%** | **82.0%** | **0.952** | 🏆 **Selected Best Model** |
| **XGBoost Classifier** | 85.9% | 80.0% | 84.9% | 86.5% | 82.4% | 0.937 | Evaluated Benchmark |
| **Random Forest Classifier** | 81.3% | 74.7% | 78.8% | 83.0% | 76.7% | 0.892 | Evaluated Benchmark |
| **HistGradientBoosting** | 86.4% | 82.3% | 82.9% | 88.7% | 82.6% | 0.935 | Evaluated Benchmark |

### Why Logistic Regression Was Selected
In enterprise B2B sales forecasting, missed risks (false positives) lead to sudden quarter slippage and wasted executive attention. Logistic Regression achieved the highest **ROC-AUC (0.952)** and superior **Recall on Lost Deals (86.5%)**, while providing exact, unapproximated linear Shapley attributions with zero inference latency.

### Holdout Confusion Matrix ($N = 375$ Unseen Deals)
* **True Negatives (Correctly Caught Lost Deals)**: 198
* **True Positives (Correctly Predicted Progression)**: 123
* **False Positives (False Alarms)**: 31
* **False Negatives (Missed Lost Deals)**: 23

---

## 🔮 Prediction Example (Illustrative)

```text
🔮 DEAL FORECAST

Customer: Rahul Sharma
Company:  Zenith Textiles

PREDICTED OUTCOME
🔮 Likely to Progress — 99% chance
(1% probability of stalling or loss)

WHY WE THINK THIS (Positive Signals)
✓ Recent touchpoint within healthy sales cadence
✓ Customer interest is improving across conversations
✓ Product demonstration has been successfully completed
✓ Executive decision maker is involved in conversations

⚠ WATCH OUT FOR (Main Concern)
Pricing concerns remain an unresolved friction point

🔮 NEXT LIKELY STEP
Negotiation (99% confidence)

💡 WHAT SHOULD I DO NEXT?
"Address the pricing concern using the ROI comparison that worked in previous discussions."
```

*(Values shown above illustrate dynamic output generated via `predict_proba()`, `LinearExplainer`, and Groq).*

---

## 📸 Screenshots

| Feature | Screenshot |
| :--- | :--- |
| **Pre-Call Executive Brief** | ![Pre-Call Brief](./screenshots/img4.png) |
| **Customer Details & Context** | ![Customer Details](./screenshots/img2.png) |
| **Call Timeline & History** | ![Call History](./screenshots/img3.png) |
| **Hindsight Memory Layer** | ![Hindsight Memory](./screenshots/hindsight.jpeg) |
| **System Architecture** | ![Architecture](./screenshots/archi.jpeg) |

---

## 🚀 Local Setup Instructions

### Prerequisites
* **Python**: 3.10, 3.11, 3.12, or 3.14
* **Node.js**: v18+ and npm
* **Git**

---

### 1. Backend Setup

Open a terminal and navigate to the backend directory:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### Configure Environment Variables
Create `backend/.env` based on `backend/.env.example`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

HINDSIGHT_API_KEY=your_hindsight_api_key_here
HINDSIGHT_BASE_URL=https://api.hindsight.vectorize.io
HINDSIGHT_BANK_ID=dealsight-sales
```

*(Note: The server will run and serve ML predictions even if API keys are not supplied. Groq and Hindsight features will gracefully fall back if keys are absent).*

#### Start Backend Server
```powershell
uvicorn main:app --reload --port 8001
```

Backend will be available at:
* **API Server**: `http://localhost:8001`
* **Interactive API Docs (Swagger)**: `http://localhost:8001/docs`

---

### 2. Frontend Setup

Open a second terminal and navigate to the frontend directory:

```powershell
cd frontend
npm install
```

#### Configure Environment Variables
Create `frontend/.env` based on `frontend/.env.example`:

```env
VITE_API_URL=http://localhost:8001
```

#### Start Frontend Development Server
```powershell
npm run dev
```

Frontend will be available at:
* **Web Application**: `http://localhost:5173`

---

## 🔗 Local Service URLs

| Service | URL |
| :--- | :--- |
| **Frontend Application** | `http://localhost:5173` |
| **Backend API** | `http://localhost:8001` |
| **Interactive API Documentation** | `http://localhost:8001/docs` |
| **Cloud Deployment** | *Coming soon* |

---

## 🔒 Security & Privacy

* **Strict Secret Isolation**: All sensitive credentials (`GROQ_API_KEY`, `HINDSIGHT_API_KEY`) are loaded via backend environment variables and never exposed to the frontend client.
* **Git Protection**: `.env` and `*.env` files are strictly excluded from version control via root and backend `.gitignore` rules.
* **Sanitized Responses**: Raw API errors and stack traces are caught by custom exception handlers returning safe business-friendly messages.

---

## ⚖️ Technical Honesty & Limitations

* **Outcome vs Stage Separation**: The primary binary classifier predicts **Progression vs Loss**. A separate, dedicated multi-class model (`models/next_stage_model.pkl`) predicts specific pipeline milestone transitions.
* **Predictions vs Recommendations**: The machine learning model generates probabilities; the Groq LLM synthesizes natural-language advice. DealSight does not make automated sales commitments—the human salesperson always retains final decision authority.
* **Synthetic B2B Dataset**: The historical training dataset consists of 2,500 modeled enterprise deals designed to reflect typical sales cycle distributions, objections, and cadences.

---

## 🔮 Future Roadmap

* **Live CRM Integration**: Bi-directional synchronization with Salesforce, HubSpot, and Close.
* **Automated Meeting Transcription**: Direct ingestion of Zoom and Google Meet audio recordings.
* **Model Drift Monitoring**: Automatic tracking of feature distribution shifts and continuous model retraining.
* **Multi-Tenant Memory Isolation**: Enterprise team workspaces with role-based access control.

---

## 📄 License

This project is licensed under the MIT License.