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