
import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

# Database location
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "phishguard.db"


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            input_type TEXT NOT NULL,
            analyzed_input TEXT NOT NULL,
            risk_score INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            ai_status TEXT,
            ai_confidence REAL,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def create_user(email, password):
    conn = get_db_connection()

    password_hash = generate_password_hash(password)

    try:
        conn.execute(
            """
            INSERT INTO users (email, password_hash)
            VALUES (?, ?)
            """,
            (email, password_hash)
        )

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()


def verify_user(email, password):
    conn = get_db_connection()

    user = conn.execute(
        "SELECT * FROM users WHERE email = ?",
        (email,)
    ).fetchone()

    conn.close()

    if user and check_password_hash(
        user["password_hash"], password
    ):
        return user

    return None


def save_scan(
    user_id,
    input_type,
    analyzed_input,
    risk_score,
    risk_level,
    ai_status,
    ai_confidence,
    explanation
):
    conn = get_db_connection()

    cursor = conn.execute(
        """
        INSERT INTO scans (
            user_id,
            input_type,
            analyzed_input,
            risk_score,
            risk_level,
            ai_status,
            ai_confidence,
            explanation
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            input_type,
            analyzed_input,
            risk_score,
            risk_level,
            ai_status,
            ai_confidence,
            explanation
        )
    )

    scan_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return scan_id


def get_user_scans(user_id):
    conn = get_db_connection()

    scans = conn.execute(
        """
        SELECT *
        FROM scans
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 100
        """,
        (user_id,)
    ).fetchall()

    conn.close()

    return scans
