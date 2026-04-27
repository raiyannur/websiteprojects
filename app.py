"""
GradeAI — Flask backend
POST /api/predict/csv  — accepts clean CSV (semester, gpa, cgpa)
POST /api/predict/pdf  — accepts a university transcript PDF, extracts GPAs automatically
POST /api/register     — create a new user account
POST /api/login        — authenticate and start a session
POST /api/logout       — end the current session
GET  /api/me           — return current user info
POST /api/me/name      — update display name
"""

from flask import Flask, request, jsonify, send_from_directory, session
import os
from flask_cors import CORS
from flask_login import (
    LoginManager, UserMixin,
    login_user, logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
import psycopg2.errors
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import r2_score
import pdfplumber
import re
import io

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "gradeai-dev-secret-change-in-prod")
CORS(app, supports_credentials=True)

# ─────────────────────────────────────────────────────────────
# FLASK-LOGIN SETUP
# ─────────────────────────────────────────────────────────────

login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({"error": "Authentication required."}), 401


# ─────────────────────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────────────────────

def _db_url():
    url = os.environ.get("DATABASE_URL", "")
    # Render gives postgres:// but psycopg2 needs postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url

def get_db():
    return psycopg2.connect(_db_url())

def init_db():
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                name          VARCHAR(100)  NOT NULL,
                email         VARCHAR(255)  UNIQUE NOT NULL,
                password_hash VARCHAR(255)  NOT NULL,
                created_at    TIMESTAMP     DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        print(f"[GradeAI] DB init warning: {exc}")

init_db()


# ─────────────────────────────────────────────────────────────
# USER MODEL
# ─────────────────────────────────────────────────────────────

class User(UserMixin):
    def __init__(self, id, name, email, created_at):
        self.id         = id
        self.name       = name
        self.email      = email
        self.created_at = created_at

@login_manager.user_loader
def load_user(user_id):
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "SELECT id, name, email, created_at FROM users WHERE id = %s",
            (user_id,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return User(*row)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    name     = data.get("name",     "").strip()
    email    = data.get("email",    "").strip().lower()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({"error": "Name, email, and password are required."}), 400
    if "@" not in email or "." not in email.split("@")[-1]:
        return jsonify({"error": "Please enter a valid email address."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400

    pw_hash = generate_password_hash(password)
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s) "
            "RETURNING id, name, email, created_at",
            (name, email, pw_hash)
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
    except psycopg2.errors.UniqueViolation:
        return jsonify({"error": "An account with this email already exists."}), 409
    except Exception as exc:
        return jsonify({"error": "Registration failed. Please try again."}), 500

    user = User(*row)
    login_user(user, remember=True)
    return jsonify({"message": "Account created.", "name": user.name}), 201


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email    = data.get("email",    "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "SELECT id, name, email, password_hash, created_at FROM users WHERE email = %s",
            (email,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception:
        return jsonify({"error": "Login failed. Please try again."}), 500

    # Deliberate vague message — do not hint which field is wrong (FR-AUTH-07)
    if not row or not check_password_hash(row[3], password):
        return jsonify({"error": "Invalid email or password."}), 401

    user = User(row[0], row[1], row[2], row[4])
    login_user(user, remember=True)
    return jsonify({"message": "Login successful.", "name": user.name})


@app.route("/api/logout", methods=["POST"])
def logout():
    logout_user()
    return jsonify({"message": "Logged out."})


@app.route("/api/me", methods=["GET"])
@login_required
def me():
    return jsonify({
        "id":         current_user.id,
        "name":       current_user.name,
        "email":      current_user.email,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
    })


@app.route("/api/me/name", methods=["POST"])
@login_required
def update_name():
    data     = request.get_json(silent=True) or {}
    new_name = data.get("name", "").strip()
    if not new_name or len(new_name) > 100:
        return jsonify({"error": "Invalid name."}), 400
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("UPDATE users SET name = %s WHERE id = %s", (new_name, current_user.id))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"name": new_name})
    except Exception:
        return jsonify({"error": "Failed to update name."}), 500


# ─────────────────────────────────────────────────────────────
# SHARED PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────

def run_prediction(semesters: list[dict]) -> dict:
    """
    semesters: list of {"semester": int, "gpa": float, "cgpa": float}
    Returns prediction dict.

    Current-semester detection:
      If the LAST row has GPA = 0.00 it is treated as the current in-progress
      semester (grades not yet posted).  The prediction is made FOR that
      semester (next_semester == its number, not max+1).  All regression and
      history logic operates only on the preceding completed semesters.

    Outlier handling for completed semesters (three layers):
      1. Zero-GPA rows in the middle of the history (all-W/Z terms) are
         stripped from the regression fit — they are administrative records.
      2. IQR fence (1.5×IQR) removes remaining statistical outliers.
         Relaxed automatically when too few points survive.
      3. HuberRegressor (epsilon=1.35) provides final residual-outlier
         robustness without discarding data.

    CGPA projection:
      Uses the authoritative pre-current CGPA as the weighted base.
      For PDF uploads this comes from the transcript footer (credit-weighted).
      Formula: (current_cgpa × n_valid + predicted_gpa) / (n_valid + 1)
    """
    df = pd.DataFrame(semesters).sort_values("semester").reset_index(drop=True)

    if len(df) < 3:
        raise ValueError("At least 3 semesters of data are required.")

    # ── Detect current in-progress semester ───────────────────────────────
    # A trailing GPA = 0.00 means the semester is ongoing — no grades posted.
    # We predict FOR this semester, not the next one after it.
    last_is_current = (float(df.iloc[-1]["gpa"]) == 0.0)

    if last_is_current:
        next_sem     = int(df.iloc[-1]["semester"])
        # For PDF uploads: extract_gpa_from_pdf already overrides the last
        # row's CGPA with the authoritative transcript-footer value, which
        # reflects only completed semesters — perfect as a projection base.
        current_cgpa = float(df.iloc[-1]["cgpa"])
        df_history   = df.iloc[:-1].copy().reset_index(drop=True)
    else:
        next_sem     = int(df["semester"].max()) + 1
        current_cgpa = float(df["cgpa"].iloc[-1])
        df_history   = df.copy().reset_index(drop=True)

    if len(df_history) < 3:
        raise ValueError("At least 3 completed semesters of data are required.")

    # ── Layer 1: strip zero-GPA semesters from completed history ──────────
    df_valid      = df_history[df_history["gpa"] > 0.0].copy().reset_index(drop=True)
    excluded_sems = df_history[df_history["gpa"] == 0.0]["semester"].tolist()

    if len(df_valid) < 3:
        raise ValueError(
            "At least 3 semesters with earned credits are required. "
            f"Only {len(df_valid)} non-zero GPA semester(s) found."
        )

    # CSV fallback: if current_cgpa is still 0 (user left the field blank),
    # use the last completed semester's CGPA instead.
    if current_cgpa == 0.0:
        current_cgpa = float(df_valid.iloc[-1]["cgpa"])

    # ── Layer 2: IQR fence on remaining GPAs ──────────────────────────────
    y_valid = df_valid["gpa"].values
    q1, q3  = float(np.percentile(y_valid, 25)), float(np.percentile(y_valid, 75))
    iqr     = q3 - q1
    lo, hi  = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df_fit  = df_valid[(df_valid["gpa"] >= lo) & (df_valid["gpa"] <= hi)].copy()

    # Fall back to all valid data if IQR removes too many points
    if len(df_fit) < 3:
        df_fit = df_valid

    X_fit = df_fit[["semester"]].values
    y_fit = df_fit["gpa"].values

    # ── Layer 3: Huber regression (resistant to residual outliers) ─────────
    reg_gpa  = HuberRegressor(epsilon=1.35, max_iter=300).fit(X_fit, y_fit)
    pred_gpa = float(reg_gpa.predict([[next_sem]])[0])
    pred_gpa = round(max(0.0, min(4.0, pred_gpa)), 4)
    r2_gpa   = float(r2_score(y_fit, reg_gpa.predict(X_fit)))

    # ── CGPA projection ────────────────────────────────────────────────────
    n_valid   = len(df_valid)
    proj_cgpa = round((current_cgpa * n_valid + pred_gpa) / (n_valid + 1), 4)
    proj_cgpa = max(0.0, min(4.0, proj_cgpa))

    # ── CGPA regression (on valid semesters only) ──────────────────────────
    X_valid       = df_valid[["semester"]].values
    y_cgpa        = df_valid["cgpa"].values
    reg_cgpa      = LinearRegression().fit(X_valid, y_cgpa)
    pred_cgpa_reg = round(float(max(0.0, min(4.0, reg_cgpa.predict([[next_sem]])[0]))), 4)
    r2_cgpa       = float(r2_score(y_cgpa, reg_cgpa.predict(X_valid)))

    # ── Confidence interval from Huber residuals ───────────────────────────
    residuals = y_fit - reg_gpa.predict(X_fit)
    std_error = float(np.std(residuals, ddof=min(2, len(residuals) - 1)))
    ci_lower  = round(max(0.0, pred_gpa - std_error), 4)
    ci_upper  = round(min(4.0, pred_gpa + std_error), 4)

    return {
        "next_semester":             next_sem,
        "predicted_gpa":             pred_gpa,
        "predicted_gpa_ci_lower":    ci_lower,
        "predicted_gpa_ci_upper":    ci_upper,
        "projected_cgpa_arithmetic": proj_cgpa,
        "projected_cgpa_regression": pred_cgpa_reg,
        "r2_gpa":                    round(r2_gpa, 4),
        "r2_cgpa":                   round(r2_cgpa, 4),
        "slope_gpa":                 round(float(reg_gpa.coef_[0]), 4),
        "total_semesters_used":      len(df_fit),    # semesters used in regression fit
        "total_semesters_valid":     n_valid,         # completed semesters with earned credits
        "excluded_semesters":        excluded_sems,   # mid-history zero-GPA semesters skipped
        "is_predicting_current":     last_is_current, # True = predicting the ongoing semester
        "history":                   df_history[["semester", "gpa", "cgpa"]].to_dict(orient="records"),
    }


# ─────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_student_info_from_lines(lines: list[str]) -> dict:
    """
    Extract student name, ID, program (major), and minor from transcript header.

    Handles NSU-style layout where multiple fields share one line, e.g.:
        ID: 2211463 Time: 16-Apr-2026 04:08:14 AM
        Name: FARHA YESMIN Major(s): Computer Science and Engineering
        Address: ...  Minor: Management Information Systems

    Also handles generic single-field-per-line transcripts.
    Values are None when not found.
    """
    info = {
        'student_name': None,
        'student_id': None,
        'program': None,
        'minor': None,
    }

    # Lookahead that stops at the next field label ("Word:" or "Word(s):") on
    # the same line, or at end-of-line — whichever comes first.
    _STOP = r'(?=\s+\w+[\w()]*\s*:|$)'

    for line in lines[:100]:
        s = line.strip()
        if not s:
            continue

        # ── Student ID ─────────────────────────────────────────────────────
        # NSU: "ID: 2211463 Time: ..."  /  generic: "Student ID: 12345678"
        if not info['student_id']:
            m = re.search(r'\b(?:Student\s+)?ID\s*:\s*(\d{4,12})\b', s, re.IGNORECASE)
            if m:
                info['student_id'] = m.group(1).strip()

        # ── Student Name ───────────────────────────────────────────────────
        # Stop before the next "Word:" / "Word(s):" on the same line.
        if not info['student_name']:
            m = re.search(r'\b(?:Student\s+)?Name\s*:\s*(.+?)' + _STOP, s, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if 2 < len(val) < 80:
                    info['student_name'] = val

        # ── Program / Major ────────────────────────────────────────────────
        # NSU uses "Major(s):" — handle the "(s)" suffix.
        if not info['program']:
            m = re.search(r'\bMajor[s()]*\s*:\s*(.+?)' + _STOP, s, re.IGNORECASE)
            if not m:
                m = re.search(
                    r'(?:Program|Department|Degree|Field\s+of\s+Study)\s*:\s*(.+?)' + _STOP,
                    s, re.IGNORECASE
                )
            if m:
                val = m.group(1).strip()
                if 2 < len(val) < 120:
                    info['program'] = val

        # ── Minor ──────────────────────────────────────────────────────────
        if not info['minor']:
            m = re.search(r'\bMinor[s()]*\s*:\s*(.+?)' + _STOP, s, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if 2 < len(val) < 80 and val.upper() not in ('N/A', 'NONE', 'NIL', '-', 'NA'):
                    info['minor'] = val

    return info


def extract_gpa_from_pdf(pdf_bytes: bytes) -> tuple[list[dict], dict]:
    """
    Parse a university transcript PDF and extract per-semester GPAs.

    Strategy:
      - Read every page's text with pdfplumber
      - Find blocks that look like semester headers (e.g. "AUTUMN 2022", "SPRING 2023")
      - After each block find the "GPA : X.XX" line
      - Also read cumulative CGPA from the last page

    Returns list of {"semester": int, "gpa": float, "cgpa": float}
    """
    text_pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_pages.append(t)

    full_text = "\n".join(text_pages)
    lines = full_text.split("\n")

    # ── Extract cumulative GPA from transcript footer ─────────────────────────
    cumulative_gpa = None
    total_gp = None
    total_credits_gpa = None

    # Also search the full text for cases where pdfplumber splits table cells
    # across lines — join every pair of adjacent lines and search those too.
    for line in lines:
        # "Cumulative GPA : 3.85" or "CGPA : 3.85"
        m = re.search(r'(?:Cumulative\s+GPA|CGPA)\s*[:\-]\s*([\d.]+)', line, re.IGNORECASE)
        if m:
            cumulative_gpa = float(m.group(1))
        # "Total Grade Point : 435.30"
        m2 = re.search(r'Total\s+Grade\s+Point\s*[:\-]\s*([\d.]+)', line, re.IGNORECASE)
        if m2:
            total_gp = float(m2.group(1))
        # "Total Credits Earned : 113.00" (credits counted for GPA — excludes W)
        m3 = re.search(r'Total\s+Credits\s+Earned\s*[:\-]\s*([\d.]+)', line, re.IGNORECASE)
        if m3:
            total_credits_gpa = float(m3.group(1))

    # Also search pairs of adjacent lines in case the label and value are split
    for i in range(len(lines) - 1):
        joined = lines[i].strip() + " " + lines[i + 1].strip()
        if cumulative_gpa is None:
            m = re.search(r'(?:Cumulative\s+GPA|CGPA)\s*[:\-]?\s*([\d.]+)', joined, re.IGNORECASE)
            if m:
                cumulative_gpa = float(m.group(1))
        if total_gp is None:
            m2 = re.search(r'Total\s+Grade\s+Point\s*[:\-]?\s*([\d.]+)', joined, re.IGNORECASE)
            if m2:
                total_gp = float(m2.group(1))
        if total_credits_gpa is None:
            m3 = re.search(r'Total\s+Credits\s+Earned\s*[:\-]?\s*([\d.]+)', joined, re.IGNORECASE)
            if m3:
                total_credits_gpa = float(m3.group(1))

    # Fallback: compute CGPA from Total Grade Point / Total Credits Earned
    if cumulative_gpa is None and total_gp is not None and total_credits_gpa is not None and total_credits_gpa > 0:
        cumulative_gpa = round(total_gp / total_credits_gpa, 2)

    # ── Extract semester blocks ───────────────────────────────────────────────
    # Semester header pattern: "AUTUMN 2022", "SPRING 2023", "SUMMER 1 2024", etc.
    semester_header_re = re.compile(
        r'^(AUTUMN|SPRING|SUMMER\s*\d*|FALL|WINTER)\s+\d{4}', re.IGNORECASE
    )
    # GPA line: "GPA : 3.91" or "Semester GPA: 3.91"
    gpa_line_re = re.compile(r'^GPA\s*[:\-]\s*([\d.]+)', re.IGNORECASE)

    semesters_raw = []   # list of {"label": str, "gpa": float}
    current_label = None

    for line in lines:
        stripped = line.strip()
        if semester_header_re.match(stripped):
            current_label = stripped
        elif gpa_line_re.match(stripped) and current_label:
            m = gpa_line_re.match(stripped)
            gpa_val = float(m.group(1))
            semesters_raw.append({"label": current_label, "gpa": gpa_val})
            current_label = None  # reset — avoid duplicate GPA for same semester

    if not semesters_raw:
        raise ValueError(
            "Could not extract any semester GPA data from this PDF. "
            "Make sure it is a text-based transcript (not scanned) with 'GPA :' lines per semester."
        )

    # ── Build running CGPA ────────────────────────────────────────────────────
    # We compute running CGPA from the extracted GPAs.
    # If we have total_gp and total_credits, the final CGPA is authoritative — 
    # use it to back-check.
    result = []
    running_sum   = 0.0
    running_count = 0
    for i, s in enumerate(semesters_raw):
        if s["gpa"] > 0.0:          # skip zero-GPA semesters (W/F/Z terms)
            running_sum   += s["gpa"]
            running_count += 1
        cgpa = round(running_sum / running_count, 4) if running_count > 0 else 0.0
        result.append({
            "semester": i + 1,
            "gpa": s["gpa"],
            "cgpa": cgpa,
        })

    # Override final CGPA with the authoritative value from the transcript footer
    if cumulative_gpa is not None:
        result[-1]["cgpa"] = cumulative_gpa

    # Extract student profile info
    student_info = extract_student_info_from_lines(lines)

    return result, student_info


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/api/predict/csv", methods=["POST"])
def predict_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a CSV as form-data with key 'file'."}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv files are accepted on this endpoint."}), 400

    try:
        content = file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    df.columns = df.columns.str.strip().str.lower()
    missing = {"semester", "gpa", "cgpa"} - set(df.columns)
    if missing:
        return jsonify({"error": f"Missing required columns: {missing}"}), 400

    for col in ["semester", "gpa", "cgpa"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return jsonify({"error": f"Column '{col}' must be numeric."}), 400

    try:
        result = run_prediction(df.to_dict(orient="records"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify(result)


@app.route("/api/predict/pdf", methods=["POST"])
def predict_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a PDF as form-data with key 'file'."}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only .pdf files are accepted on this endpoint."}), 400

    pdf_bytes = file.read()

    try:
        semesters, student_info = extract_gpa_from_pdf(pdf_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"PDF parsing failed: {e}"}), 500

    try:
        result = run_prediction(semesters)
        result["source"] = "pdf"
        result["semesters_extracted"] = len(semesters)
        result.update(student_info)   # adds student_name, student_id, program, minor
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify(result)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/<path:filename>", methods=["GET"])
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "GradeAI prediction API", "version": "2.0"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
