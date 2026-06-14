"""
load_data.py - Initializes the SQLite database and loads cell-count.csv.
Run with: python load_data.py
"""

import sqlite3
import csv
import os

DB_PATH = "teiko.db"
CSV_PATH = "cell-count.csv"


def init_db(conn: sqlite3.Connection) -> None:
    """Create the relational schema."""
    conn.executescript("""
        PRAGMA foreign_keys = ON;

        -- Projects: top-level grouping for clinical trials
        CREATE TABLE IF NOT EXISTS projects (
            project_id   TEXT PRIMARY KEY
        );

        -- Subjects: one row per patient
        CREATE TABLE IF NOT EXISTS subjects (
            subject_id   TEXT PRIMARY KEY,
            project_id   TEXT NOT NULL REFERENCES projects(project_id),
            condition    TEXT NOT NULL,
            age          INTEGER,
            sex          TEXT CHECK(sex IN ('M', 'F')),
            treatment    TEXT NOT NULL,
            response     TEXT CHECK(response IN ('yes', 'no'))
        );

        -- Samples: one row per biological sample
        CREATE TABLE IF NOT EXISTS samples (
            sample_id                   TEXT PRIMARY KEY,
            subject_id                  TEXT NOT NULL REFERENCES subjects(subject_id),
            sample_type                 TEXT NOT NULL,
            time_from_treatment_start   INTEGER NOT NULL
        );

        -- Cell counts: one row per sample (wide format for fast aggregation)
        CREATE TABLE IF NOT EXISTS cell_counts (
            sample_id    TEXT PRIMARY KEY REFERENCES samples(sample_id),
            b_cell       INTEGER NOT NULL,
            cd8_t_cell   INTEGER NOT NULL,
            cd4_t_cell   INTEGER NOT NULL,
            nk_cell      INTEGER NOT NULL,
            monocyte     INTEGER NOT NULL
        );

        -- Indexes to accelerate common analytical queries
        CREATE INDEX IF NOT EXISTS idx_subjects_project   ON subjects(project_id);
        CREATE INDEX IF NOT EXISTS idx_subjects_condition ON subjects(condition);
        CREATE INDEX IF NOT EXISTS idx_subjects_treatment ON subjects(treatment);
        CREATE INDEX IF NOT EXISTS idx_subjects_response  ON subjects(response);
        CREATE INDEX IF NOT EXISTS idx_samples_subject    ON samples(subject_id);
        CREATE INDEX IF NOT EXISTS idx_samples_type       ON samples(sample_type);
        CREATE INDEX IF NOT EXISTS idx_samples_time       ON samples(time_from_treatment_start);
    """)
    conn.commit()


def load_csv(conn: sqlite3.Connection, csv_path: str) -> int:
    """Load all rows from cell-count.csv into the database. Returns row count."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows_loaded = 0
        for row in reader:
            project_id = row["project"]
            subject_id = row["subject"]
            sample_id  = row["sample"]

            conn.execute(
                "INSERT OR IGNORE INTO projects VALUES (?)",
                (project_id,)
            )
            conn.execute(
                """INSERT OR IGNORE INTO subjects
                   (subject_id, project_id, condition, age, sex, treatment, response)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    subject_id, project_id,
                    row["condition"],
                    int(row["age"]) if row["age"] else None,
                    row["sex"] or None,
                    row["treatment"],
                    row["response"] or None,
                )
            )
            conn.execute(
                """INSERT OR IGNORE INTO samples
                   (sample_id, subject_id, sample_type, time_from_treatment_start)
                   VALUES (?, ?, ?, ?)""",
                (
                    sample_id, subject_id,
                    row["sample_type"],
                    int(row["time_from_treatment_start"]),
                )
            )
            conn.execute(
                """INSERT OR IGNORE INTO cell_counts
                   (sample_id, b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    sample_id,
                    int(row["b_cell"]),
                    int(row["cd8_t_cell"]),
                    int(row["cd4_t_cell"]),
                    int(row["nk_cell"]),
                    int(row["monocyte"]),
                )
            )
            rows_loaded += 1

        conn.commit()
    return rows_loaded


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find {CSV_PATH}. Make sure it is in the same directory.")

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        init_db(conn)
        print("Schema initialized.")
        n = load_csv(conn, CSV_PATH)
        print(f"Loaded {n} rows into {DB_PATH}.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
