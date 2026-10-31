# 📄 AI Resume ATS Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) compatibility and gives specific improvements, using Google's **Gemini Flash** model.

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** for a targeted keyword match
- **ATS score (0-100)** with a breakdown: keyword match, formatting, content quality, section completeness
- Strengths, **prioritised improvements**, missing keywords and before/after rewrite examples
- Download the report as JSON

## Project structure
```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Run locally
1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. Install and run:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Provide the key either way:
   - paste it into the sidebar when the app opens, **or**
   - create `.streamlit/secrets.toml`:
     ```toml
     GEMINI_API_KEY = "your-key-here"
     # GEMINI_MODEL = "gemini-2.5-flash"   # optional
     ```

> ⚠️ Never commit `.streamlit/secrets.toml` or your API key to GitHub. Add `.streamlit/secrets.toml` to `.gitignore`.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and click **Create app**.
3. Select your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Notes
- Scanned/image-only resumes can't be read (ATS systems can't read them either). Use a text-based PDF or DOCX.
- The score is an AI-generated estimate, not an official score from any real ATS.
- If Google retires the default model, set `GEMINI_MODEL` to a current Flash model name.
