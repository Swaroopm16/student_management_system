# DevLevel — Developer Skill Classifier

An ML-powered web app that classifies developers as **Beginner**, **Intermediate**, or **Expert** using a trained RandomForest model.

## Project Structure

```
devlevel-app/
├── app.py                  ← Flask backend
├── requirements.txt
├── models/
│   ├── student_model.pkl   ← Trained RandomForest pipeline
│   └── label_encoder.pkl   ← Label encoder
└── templates/
    └── index.html          ← Frontend UI
```

## Features Used by Model

The model uses **6 features**:
1. `leetcode_total` — Total LeetCode problems solved
2. `contest_rating` — LeetCode contest rating
3. `github_repos` — Number of GitHub repositories
4. `github_commits` — Total GitHub commits
5. `commit_streak` — Longest commit streak (days)
6. `open_source_prs` — Open source pull requests merged

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Flask server
```bash
python app.py
```

### 3. Open the app
Visit [http://localhost:5000](http://localhost:5000) in your browser.

## API Endpoint

**POST /predict**

```json
{
  "leetcode_total": 150,
  "contest_rating": 1500,
  "github_repos": 20,
  "github_commits": 500,
  "commit_streak": 30,
  "open_source_prs": 5
}
```

**Response:**
```json
{
  "success": true,
  "label": "Intermediate",
  "top_confidence": 78.5,
  "confidence": {
    "Beginner": 10.0,
    "Intermediate": 78.5,
    "Expert": 11.5
  }
}
```
