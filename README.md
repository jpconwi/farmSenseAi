# 🌾 FarmSense AI — Farmer Report Analyzer

**CS 315 – Application Development and Emerging Technologies — Activity 3**

## 1. Project Purpose

FarmSense AI is a GenAI-powered Streamlit application that analyzes farmer
reports about agricultural problems. It loads a CSV dataset of unstructured
farmer reports, cleans it with Pandas, uses the Google Gemini API to classify and
summarize each report, displays interactive charts, and includes an AI
chatbot that can answer questions about the dataset.

## 2. Features

- Load and clean a CSV dataset with Pandas (remove empty rows, duplicates,
  fix dates, handle missing values)
- Filter reports by crop, location, and date range
- AI classification of each report into: **Pest, Disease, Water, Weather,
  Nutrient, Other** — plus severity (Low/Moderate/High), keywords, and a
  short summary
- Dashboard metric cards: total reports, number of crops, number of
  locations, most common problem
- Interactive Plotly charts: reports by crop, reports by location, problem
  categories
- **AI Report Analyzer**: paste in a brand-new report and get an instant
  AI classification
- **"Ask FarmSense AI" chatbot**: ask natural-language questions about the
  dataset, answered only from the actual data (no made-up answers)
- Beginner-friendly error handling everywhere (missing file, empty data,
  missing API key, API errors, bad AI responses, missing columns, no
  internet)

## 3. Dataset Description

File: `data/farmer_reports.csv`

| Column     | Description                                   |
|------------|------------------------------------------------|
| `date`     | Date the report was submitted (YYYY-MM-DD)      |
| `location` | Barangay/town where the report came from        |
| `crop`     | Crop affected (Rice, Corn, Coconut, Banana, Vegetables) |
| `report`   | Free-text description of the problem, in the farmer's own words |

The included sample dataset has 48 rows and intentionally contains a few
messy rows (a duplicate, an empty report, a missing date, extra whitespace)
so the cleaning step in `app.py` has real work to do. You can replace this
file with your own data as long as you keep the same 4 columns.

## 4. Technologies Used

- **Python 3**
- **Streamlit** — the web app framework/UI
- **Pandas** — data loading and cleaning
- **Google Gemini API** (`google-genai` Python package) — GenAI classification,
  summary, and chatbot
- **Plotly** — interactive charts

## 5. Project Structure

```
FarmSenseAI/
│
├── app.py                     # Main Streamlit app (UI + dashboard)
├── requirements.txt           # Python packages needed
├── README.md                  # This file
│
├── data/
│   └── farmer_reports.csv     # The dataset
│
├── utils/
│   ├── __init__.py            # Makes "utils" an importable Python package
│   └── ai_analysis.py         # All functions that call the Gemini API
│
└── .streamlit/
    └── secrets.toml           # Your Gemini API key goes here (kept private)
```

Where each file goes, in plain terms:
- Put the whole `FarmSenseAI` folder anywhere on your computer (e.g.
  Desktop or Documents).
- `app.py` stays in the top-level `FarmSenseAI/` folder.
- `farmer_reports.csv` must be inside `FarmSenseAI/data/`.
- `ai_analysis.py` must be inside `FarmSenseAI/utils/`.
- `secrets.toml` must be inside `FarmSenseAI/.streamlit/` (note the dot at
  the start of the folder name — it's a hidden folder).

## 6. Installation Instructions

Open a terminal (Command Prompt, PowerShell, or Terminal on Mac) and run:

```bash
# 1. Move into the project folder
cd path/to/FarmSenseAI

# 2. (Recommended) Create a virtual environment
python -m venv venv

# 3. Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 4. Install the required packages
pip install -r requirements.txt
```

## 7. How to Configure the Gemini API Key

1. Get a free API key from https://aistudio.google.com/app/apikey (sign in
   with a Google account — no billing setup required to start).
2. Open `.streamlit/secrets.toml` in a text editor.
3. Replace the placeholder with your real key:
   ```toml
   GEMINI_API_KEY = "your-real-key-here"
   GEMINI_MODEL = "gemini-3.6-flash"
   ```
4. Save the file. **Do not share this file or upload it to GitHub** — the
   included `.gitignore` already excludes it for you.
5. ⚠️ If your key was ever pasted into a chat, document, or committed to a
   public repo by mistake, treat it as compromised: go back to
   https://aistudio.google.com/app/apikey and delete/regenerate it, then
   update `secrets.toml` with the new one.

If you skip this step, the app will still run — it will show the raw
dataset and charts, but AI classification and the chatbot will be disabled
with a clear on-screen message (no fake results are ever generated).

## 8. How to Run the Application

From inside the `FarmSenseAI` folder, with your virtual environment
activated:

```bash
streamlit run app.py
```

Your browser should automatically open `http://localhost:8501`. If it
doesn't, copy that URL into your browser manually.

## 9. Deploy to Streamlit Community Cloud

1. Create a free account at https://share.streamlit.io (sign in with
   GitHub).
2. Push your `FarmSenseAI` project to a GitHub repository.
   ```bash
   git init
   git add .
   git commit -m "Initial commit - FarmSense AI"
   git branch -M main
   git remote add origin https://github.com/your-username/FarmSenseAI.git
   git push -u origin main
   ```
   (Because `.streamlit/secrets.toml` is in `.gitignore`, your real API key
   will **not** be uploaded to GitHub — this is intentional and safe.)
3. On share.streamlit.io, click **"New app"**.
4. Select your repository, branch (`main`), and set the main file path to
   `app.py`.
5. Before clicking Deploy, open **"Advanced settings" → "Secrets"** and
   paste:
   ```toml
   GEMINI_API_KEY = "your-real-key-here"
   GEMINI_MODEL = "gemini-3.6-flash"
   ```
6. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   automatically and launch your app with a public URL.

## 10. Common Errors and Solutions

| Problem | Likely Cause | Solution |
|---|---|---|
| "Could not find the dataset file" | CSV missing or wrong location | Make sure `farmer_reports.csv` is inside `data/` |
| "No Gemini API key found" | `secrets.toml` not set up | Follow section 7 above |
| "AI request failed" | Invalid key, no internet, or the Gemini API is down | Check your key, check your internet connection, try again later |
| "ModuleNotFoundError: No module named 'utils'" | The `utils/` folder wasn't pushed to GitHub | Confirm `utils/__init__.py` and `utils/ai_analysis.py` appear in your GitHub repo, then reboot the app |
| "The AI returned a response that wasn't valid JSON" | Rare AI formatting hiccup | Click Analyze again — the app never crashes, it just skips that report |
| App won't start / `ModuleNotFoundError` | Packages not installed | Run `pip install -r requirements.txt` again inside your activated virtual environment |
| Chart is empty | Filters are too narrow | Widen your Crop/Location/Date filters in the sidebar |

## 11. Future Improvements

- Add a map view showing report locations
- Let users upload their own CSV file directly in the app
- Cache AI analysis results per report so re-running filters doesn't
  re-call the API for reports already analyzed
- Add multi-language support (Bisaya/Tagalog report input)
- Add authentication so multiple farmers/agencies can log in and see only
  their own reports

## 12. How This Project Satisfies CS 315 Activity 3

| Requirement | Where it's implemented |
|---|---|
| Dataset loading & cleaning with Pandas | `load_and_clean_data()` in `app.py` — removes empty rows, duplicates, converts dates, handles missing values |
| GenAI-powered text analysis | `utils/ai_analysis.py` — `analyze_report()` classifies category/severity/keywords/summary via Gemini |
| Streamlit interactive UI | Sidebar filters, columns, expanders, metrics, buttons, text areas throughout `app.py` |
| Data visualization | Plotly bar charts (crop, location) and pie chart (categories) |
| Dataset filtering | Sidebar crop/location/date filters applied before analysis and display |
| AI chatbot functionality | "Ask FarmSense AI" section — answers questions using a summarized dataset, never invents facts |
| Deployment readiness for Streamlit Community Cloud | `requirements.txt`, `.streamlit/secrets.toml` pattern, `.gitignore`, and step-by-step deploy instructions above |
