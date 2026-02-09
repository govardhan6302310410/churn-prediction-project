import sqlite3

def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        international INTEGER,
        voicemail INTEGER,
        vmail_messages INTEGER,
        day_minutes REAL,
        eve_minutes REAL,
        night_minutes REAL,
        service_calls INTEGER,
        result TEXT,
        probability REAL,
        date TEXT
    )
    """)

    conn.commit()
    conn.close()
