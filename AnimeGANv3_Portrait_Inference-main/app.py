from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
import subprocess
import glob

app = FastAPI()

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "out"

# Создаём папки, если их нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MODEL_PATH = os.path.abspath("deploy/models/AnimeGANv3_Hayao_36.onnx")

@app.post("/stylize")
async def stylize_image(file: UploadFile = File(...)):
    # Сохраняем загруженный файл
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return {"error": f"Failed to save uploaded file: {str(e)}"}

    try:
        # Запускаем скрипт инференса
        subprocess.run([
            "python", os.path.join(os.getcwd(), "onnx_infer.py"),
            "-i", os.path.abspath(input_path),
            "-m", MODEL_PATH,
            "-o", os.path.abspath(OUTPUT_FOLDER),
            "--background"
        ], check=True, capture_output=True, text=True)

        # Рекурсивно ищем все файлы в OUTPUT_FOLDER
        list_of_files = glob.glob(os.path.join(OUTPUT_FOLDER, "**", "*.*"), recursive=True)
        # Отфильтровываем только файлы
        list_of_files = [f for f in list_of_files if os.path.isfile(f)]

        if not list_of_files:
            return {"error": "No output file found after inference"}

        # Берём последний созданный файл
        latest_file = max(list_of_files, key=os.path.getctime)
        return FileResponse(latest_file)

    except subprocess.CalledProcessError as e:
        return {
            "error": "Inference failed",
            "stdout": e.stdout,
            "stderr": e.stderr
        }
    except Exception as e:
        return {"error": str(e)}
