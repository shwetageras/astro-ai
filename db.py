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
    import psycopg2

    conn = psycopg2.connect(
        dbname="astro_ai_db",
        user="postgres",
        password="postgres123",
        host="localhost",
        port="5432"
    )

    cursor = conn.cursor(cursor_factory=RealDictCursor)

    query = """
        SELECT job_id, chart_id, user_id, profile_id
        FROM charts_jobs
        WHERE job_id = ANY(%s)
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


def update_qna_sl_validation(qna_id, is_valid, corrected_answer=None):
    conn = get_connection()
    cursor = conn.cursor()

    # If VALID → use llm_answer as corrected_answer
    if is_valid is True:
        cursor.execute("""
            UPDATE qna_sl_logs
            SET 
                is_valid = %s,
                corrected_answer = llm_answer,
                tr_sl = TRUE
            WHERE id = %s
        """, (is_valid, qna_id))

    # If NOT VALID → use user provided corrected_answer
    elif is_valid is False:
        cursor.execute("""
            UPDATE qna_sl_logs
            SET 
                is_valid = %s,
                corrected_answer = %s,
                tr_sl = TRUE
            WHERE id = %s
        """, (is_valid, corrected_answer, qna_id))

    # If NOT REQUIRED → do nothing to corrected_answer
    else:
        cursor.execute("""
            UPDATE qna_sl_logs
            SET 
                is_valid = NULL,
                tr_sl = TRUE
            WHERE id = %s
        """, (qna_id,))

    conn.commit()
    cursor.close()
    conn.close()


def get_qna_sl(qna_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT kb_id, question, llm_answer, corrected_answer, is_valid
        FROM qna_sl_logs
        WHERE id = %s
    """, (qna_id,))

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if row:
        return {
            "kb_id": row[0],
            "question": row[1],
            "llm_answer": row[2],
            "corrected_answer": row[3],
            "is_valid": row[4]
        }

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