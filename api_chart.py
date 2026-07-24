import time
import uuid
import os
import json
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from storage import save_file, save_metadata
from kb_builder import read_pdf, chunk_text, create_embeddings, build_kb, save_kb, chunk_json_text
from db import get_job
from vector_db import upsert_embeddings
from notifier import notify_chart_status
from db import insert_chart_job, update_chart_job
from fastapi import HTTPException


app = FastAPI()

def make_safe_filename(name: str):
    return name.replace(" ", "_").replace("/", "_")


def json_to_semantic_text(data, prefix=""):

    lines = []

    if isinstance(data, dict):

        for k, v in data.items():

            current = f"{prefix} {k}".strip()

            lines.extend(
                json_to_semantic_text(v, current)
            )

    elif isinstance(data, list):

        for item in data:

            lines.extend(
                json_to_semantic_text(item, prefix)
            )

    else:

        lines.append(
            f"{prefix} is {data}"
        )

    return lines


def process_chart(file_bytes, file_id, file_name, job_id, chart_id, user_id, profile_id, timestamp):

    print("🚀 PROCESS_CHART STARTED", flush=True)


    temp_file_path = None

    try:
        temp_file_path = f"temp_{file_id}.{file_name.split('.')[-1]}"
        print("FILE SAVING START", flush=True)

        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        print("FILE SAVED", flush=True)

        file_ext = file_name.split(".")[-1].lower()
        print("FILE TYPE:", file_ext, flush=True)

        if file_ext == "pdf":

            text = read_pdf(temp_file_path)

        elif file_ext in ["md", "txt"]:

            from kb_builder import read_text_file
            text = read_text_file(temp_file_path)

        elif file_ext == "json":

            with open(temp_file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            print("TOP LEVEL KEYS:")
            print(list(json_data.keys()))

            print("TOTAL TOP LEVEL KEYS:", len(json_data.keys()))

            chunks = []

            for key, value in json_data.items():

                original_chunk = json.dumps(
                    {key: value},
                    ensure_ascii=False,
                    indent=2
                )

                if key == "05_planets_in_houses":

                    chunks.append(
                        original_chunk
                    )

                    for planet, house in value.items():

                        chunks.append(
                            f"{planet} is placed in house {house}\n"
                            f"{planet} is in the {house}th house\n"
                            f"House {house} contains {planet}"
                        )

                    continue

                semantic_lines = json_to_semantic_text(
                    {key: value}
                )

                semantic_text = "\n".join(
                    semantic_lines
                )

                chunk = original_chunk + "\n\n" + semantic_text

                if len(chunk) > 10000:

                    chunks.extend(
                        chunk_json_text(chunk)
                    )

                else:

                    chunks.append(chunk)

        else:

            raise ValueError(
                f"Unsupported file type: {file_ext}"
            )

        if file_ext != "json":

            print("TEXT EXTRACTED")
            print("TEXT LENGTH:", len(text))

            chunks = chunk_text(text)

        else:

            print("JSON CHUNKS CREATED")

        print("TOTAL CHUNKS:", len(chunks))

        for i in range(min(5, len(chunks))):
            print(f"\n========== CHUNK {i+1} ==========")
            print(chunks[i][:1000])

        embeddings = create_embeddings(chunks)
        print("EMBEDDINGS:", len(embeddings))

        print("UPSERT METADATA:")
        print({
            "chart_id": str(chart_id),
            "user_id": str(user_id),
            "profile_id": str(profile_id)
        })

        upsert_embeddings(
            file_id,
            chunks,
            embeddings,
            metadata={
                "chart_id": str(chart_id),
                "user_id": str(user_id),
                "profile_id": str(profile_id)
            }
        )

        print("UPSERT DONE")

        kb = build_kb(chunks, embeddings)
        save_kb(kb, file_id)
        print("KB SAVED")

        save_metadata(file_id, file_name, int(time.time()))
        print("METADATA SAVED")

        update_chart_job(job_id, "completed", int(time.time()))
        print("DB UPDATED")

        notify_chart_status(job_id, chart_id, file_id)
        print("CALLBACK SENT")

    except Exception as e:

        print(f"❌ ERROR in chart job {job_id}: {e}")

        update_chart_job(
            job_id,
            "failed",
            int(time.time()),
            str(e)
        )

    finally:

        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def process_chart_text(content, file_id, job_id, chart_id, user_id, profile_id, timestamp):

    print("PROCESS_CHART_TEXT STARTED", flush=True)

    try:
        print("CONTENT LENGTH:", len(content))

        print("\n========== RAW CHART CONTENT ==========")
        print(content[:2000])
        print("=======================================")

        # Step 1: Chunk
        chunks = chunk_text(content)
        print("CHUNKS:", len(chunks))

        # Step 2: Embeddings
        embeddings = create_embeddings(chunks)
        print("EMBEDDINGS:", len(embeddings))

        # Step 3: Store in Pinecone
        upsert_embeddings(
            file_id,
            chunks,
            embeddings,
            metadata={
                "user_id": str(user_id),
                "profile_id": str(profile_id),
                "chart_id": str(chart_id)
            }
        )
        print("UPSERT DONE")

        # Step 4: Build KB
        kb = build_kb(chunks, embeddings)
        save_kb(kb, file_id)
        print("KB SAVED")

        # Step 5: Metadata
        save_metadata(file_id, "chart_text", int(time.time()))
        print("METADATA SAVED")

        # Step 6: DB update
        update_chart_job(job_id, "completed", int(time.time()))
        print("DB UPDATED")

        # Step 7: Callback
        notify_chart_status(job_id, chart_id, file_id)
        print("CALLBACK SENT")

        print("PROCESS COMPLETE:", job_id)

    except Exception as e:
        print(f"❌ ERROR in chart text job {job_id}: {e}")
        update_chart_job(job_id, "failed", int(time.time()), str(e))



@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = get_job(job_id)

    if not job:
        return {"error": "Job not found"}

    return job


# Create NEW API → /upload_chart
@app.post("/upload_chart")
async def upload_chart(
    background_tasks: BackgroundTasks,
    isCharttype: str = Form(...),
    name: str = Form(...),
    user_id: int = Form(...),
    profile_id: int = Form(...),
    chart_id: int = Form(...),
    content: str = Form(None),
    file: UploadFile = File(None)
):
    print("========== UPLOAD CHART HIT ==========")


    timestamp = int(time.time())

    safe_name = make_safe_filename("chart")
    file_id = f"{timestamp}_{uuid.uuid4().hex}_{safe_name}"

    job_id = f"job_{timestamp}_{uuid.uuid4().hex}"

    insert_chart_job(
        job_id,
        file_id,
        chart_id,
        user_id,
        profile_id,
        name,
        "processing",
        timestamp
    )

    # 🔥 CASE 1: TEXT INPUT
    if isCharttype == "article":

        if not content:
            raise HTTPException(status_code=400, detail="Content required for text chart")

        # Start background processing
        background_tasks.add_task(
            process_chart_text,
            content,
            file_id,
            job_id,
            chart_id,   
            user_id,
            profile_id,
            timestamp
        )

    # 🔥 CASE 2: FILE INPUT
    elif isCharttype == "file":

        print("FILE UPLOAD MODE")

        if not file:
            raise HTTPException(status_code=400, detail="File required for chart upload")

        print("FILE NAME:", file.filename)

        file_bytes = await file.read()

        print(
            "FILE SIZE MB:",
            round(len(file_bytes) / (1024 * 1024), 2)
        )

        # Save to S3
        save_file(file_bytes, file_id)

        print("ADDING BACKGROUND TASK", flush=True)

        background_tasks.add_task(
            process_chart,
            file_bytes,
            file_id,
            file.filename,
            job_id,
            chart_id,
            user_id,
            profile_id,
            timestamp
        )

        print("BACKGROUND TASK ADDED", flush=True)

    else:
        raise HTTPException(status_code=400, detail="Invalid isCharttype")

    return {
        "job_id": job_id,
        "status": "processing"
    }