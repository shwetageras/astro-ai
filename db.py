import psycopg2

def get_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="astro_ai_db",
        user="postgres",
        password="postgres123"  # replace if different
    )
    return conn

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

def insert_chart_job(job_id, chart_id, user_id, profile_id, file_name, status, created_at):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO charts_jobs (
            job_id, chart_id, user_id, profile_id, file_name, status, created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (job_id, chart_id, user_id, profile_id, file_name, status, created_at))

    conn.commit()
    cursor.close()
    conn.close()


def get_chart_job(job_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT job_id, chart_id, user_id, profile_id, file_name, status, created_at, completed_at, error
        FROM charts_jobs
        WHERE job_id = %s
    """, (job_id,))

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if row:
        return {
            "job_id": row[0],
            "chart_id": row[1],
            "user_id": row[2],
            "profile_id": row[3],
            "file_name": row[4],
            "status": row[5],
            "created_at": row[6],
            "completed_at": row[7],
            "error": row[8]
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

    cursor.execute("""
        INSERT INTO qna_logs (user_id, profile_id, chart_id, question)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """, (user_id, profile_id, chart_id, question))

    qna_id = cursor.fetchone()[0]

    conn.commit()
    cursor.close()
    conn.close()

    return qna_id


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