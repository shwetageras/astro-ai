import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

def get_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    return conn

# -------------------------------
# JOB FUNCTIONS
# -------------------------------

def insert_job(job_id, file_id, file_name, status, created_at):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO jobs (job_id, file_id, file_name, status, created_at)
        VALUES (%s, %s, %s, %s, %s)
    """, (job_id, file_id, file_name, status, created_at))

    conn.commit()
    cursor.close()
    conn.close()

def get_job(job_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT job_id, file_id, file_name, status, created_at, completed_at, error
        FROM jobs
        WHERE job_id = %s
    """, (job_id,))

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if row:
        return {
            "job_id": row[0],
            "file_id": row[1],
            "file_name": row[2],
            "status": row[3],
            "created_at": row[4],
            "completed_at": row[5],
            "error": row[6]
        }
    
    return None

def update_job(job_id, status, completed_at=None, error=None):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE jobs
        SET status = %s,
            completed_at = %s,
            error = %s
        WHERE job_id = %s
    """, (status, completed_at, error, job_id))

    conn.commit()
    cursor.close()
    conn.close()


# -------------------------------
# CHART JOB FUNCTIONS
# -------------------------------

def insert_chart_job(
    job_id,
    file_id,
    chart_id,
    user_id,
    profile_id,
    file_name,
    status,
    created_at
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO charts_jobs (
            job_id,
            file_id,
            chart_id,
            user_id,
            profile_id,
            file_name,
            status,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        job_id,
        file_id,
        chart_id,
        user_id,
        profile_id,
        file_name,
        status,
        created_at
    ))

    conn.commit()
    cursor.close()
    conn.close()


def get_chart_job(job_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT job_id, file_id, chart_id, user_id, profile_id, file_name, status, created_at, completed_at, error
        FROM charts_jobs
        WHERE job_id = %s
    """, (job_id,))

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if row:
        return {
            "job_id": row[0],
            "file_id": row[1],
            "chart_id": row[2],
            "user_id": row[3],
            "profile_id": row[4],
            "file_name": row[5],
            "status": row[6],
            "created_at": row[7],
            "completed_at": row[8],
            "error": row[9]
        }

    return None


def update_chart_job(job_id, status, completed_at=None, error=None):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE charts_jobs
        SET status = %s,
            completed_at = %s,
            error = %s
        WHERE job_id = %s
    """, (status, completed_at, error, job_id))

    conn.commit()
    cursor.close()
    conn.close()


# -------------------------------
# QnA LOG FUNCTIONS
# -------------------------------

def insert_qna(user_id, profile_id, chart_id, question):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO qna_logs (user_id, profile_id, chart_id, question)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (user_id, profile_id, chart_id, question))

        row = cursor.fetchone()

        if not row:
            raise Exception("Insert failed: No ID returned")

        conn.commit()
        return row[0]

    finally:
        cursor.close()
        conn.close()


def update_qna_answer(qna_id, answer):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE qna_logs
        SET answer = %s
        WHERE id = %s
    """, (answer, qna_id))

    conn.commit()
    cursor.close()
    conn.close()


def get_chart_details_bulk(job_ids):
    from psycopg2.extras import RealDictCursor

    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    query = """
        SELECT job_id, chart_id, user_id, profile_id
        FROM charts_jobs
        WHERE chart_id = ANY(%s)
        AND is_deleted = FALSE
    """

    cursor.execute(query, (job_ids,))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results


def soft_delete_chart_job(job_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE charts_jobs
        SET is_deleted = TRUE
        WHERE job_id = %s
    """, (job_id,))

    conn.commit()
    cursor.close()
    conn.close()


# -------------------------------
# QNA SL FUNCTIONS
# -------------------------------

def insert_qna_sl(kb_id, question, llm_answer):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO qna_sl_logs (kb_id, question, llm_answer, created_at)
        VALUES (%s, %s, %s, EXTRACT(EPOCH FROM NOW())::BIGINT)
        RETURNING id
    """, (kb_id, question, llm_answer))

    row = cursor.fetchone()

    if row is None:
        cursor.close()
        conn.close()
        raise Exception("Insert failed: No ID returned")

    qna_id = row[0]

    conn.commit()
    cursor.close()
    conn.close()

    return qna_id


def update_qna_sl_validation(
    table_name,
    qna_id,
    is_valid,
    llm_answer=None,
    corrected_answer=None
):

    conn = get_connection()
    cursor = conn.cursor()

    # -----------------------------
    # VALID ANSWER
    # -----------------------------
    if is_valid is True:

        cursor.execute(f"""
            UPDATE {table_name}
            SET
                is_valid = %s,
                corrected_answer = %s,
                tr_sl = TRUE
            WHERE id = %s
        """, (
            is_valid,
            llm_answer,
            qna_id
        ))

    # -----------------------------
    # INVALID ANSWER
    # -----------------------------
    elif is_valid is False:

        cursor.execute(f"""
            UPDATE {table_name}
            SET
                is_valid = %s,
                corrected_answer = %s,
                tr_sl = TRUE
            WHERE id = %s
        """, (
            is_valid,
            corrected_answer,
            qna_id
        ))

    conn.commit()

    cursor.close()
    conn.close()


def get_qna_sl(table_name, qna_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT *
        FROM {table_name}
        WHERE id = %s
    """, (qna_id,))

    row = cursor.fetchone()

    columns = [desc[0] for desc in cursor.description]

    cursor.close()
    conn.close()

    if row:
        return dict(zip(columns, row))

    return None


def mark_qna_ml_ready(qna_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE qna_sl_logs
        SET tr_ml = TRUE
        WHERE id = %s
    """, (qna_id,))

    conn.commit()
    cursor.close()
    conn.close()


def insert_generated_qna(kb_id, question, answer):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO generated_qnas (
                kb_id,
                question,
                answer,
                created_at
            )
            VALUES (%s, %s, %s, EXTRACT(EPOCH FROM NOW())::BIGINT)
            RETURNING id
            """,
            (kb_id, question, answer)
        )

        row = cursor.fetchone()

        if row is None:
            raise Exception("Insert failed: No ID returned")

        conn.commit()

        return row[0]

    finally:

        cursor.close()
        conn.close()


def get_file_id_from_job(job_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT file_id
        FROM jobs
        WHERE job_id = %s
    """, (job_id,))

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if row:
        return row[0]

    return None


def delete_qna_record(table_name, qna_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(f"""
        DELETE FROM {table_name}
        WHERE id = %s
    """, (qna_id,))

    conn.commit()

    cursor.close()
    conn.close()