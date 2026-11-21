import os
import uuid
import textwrap
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip

app = FastAPI(title="Bartox AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "generated_videos"
VIDEO_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = "gemini-2.5-flash"


class ChatRequest(BaseModel):
    message: str


class VideoRequest(BaseModel):
    topic: str


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = BASE_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>index.html tidak ditemukan</h1>", status_code=404)
    return index_file.read_text(encoding="utf-8")


@app.get("/style.css")
async def serve_css():
    css_file = BASE_DIR / "style.css"
    if not css_file.exists():
        raise HTTPException(status_code=404, detail="style.css tidak ditemukan")
    return FileResponse(css_file, media_type="text/css")


async def ask_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY belum diatur di Environment Variables.",
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.post(url, json=payload)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Gemini API error: {resp.text}")

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="Respons dari Gemini kosong atau tidak valid.")


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Pesan tidak boleh kosong.")
    reply = await ask_gemini(user_msg)
    return {"reply": reply}


def create_slide_image(text: str, output_path: Path):
    width, height = 640, 360
    img = Image.new("RGB", (width, height), color=(24, 24, 37))
    draw = ImageDraw.Draw(img)

    # Frame dekoratif ringan
    draw.rectangle([16, 16, width - 16, height - 16], outline=(244, 114, 182), width=3)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    draw.text((32, 28), "BARTOX AI", fill=(244, 114, 182), font=font_title)

    wrapped = textwrap.fill(text, width=44)
    lines = wrapped.splitlines()[:9]
    y_text = 75
    for line in lines:
        draw.text((32, y_text), line, fill=(248, 250, 252), font=font_body)
        y_text += 26

    img.save(output_path)


@app.post("/generate-video")
async def generate_video_endpoint(req: VideoRequest):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topik video tidak boleh kosong.")

    script_prompt = (
        f"Buatkan narasi singkat berbahasa Indonesia maksimal 3 kalimat (sekitar 35-45 kata) "
        f"tentang: {topic}. Langsung tulis isi narasinya tanpa judul atau simbol markdown."
    )
    narration = await ask_gemini(script_prompt)

    job_id = uuid.uuid4().hex[:10]
    audio_path = VIDEO_DIR / f"{job_id}.mp3"
    image_path = VIDEO_DIR / f"{job_id}.png"
    video_path = VIDEO_DIR / f"{job_id}.mp4"

    try:
        # 1. Buat suara (TTS)
        tts = gTTS(text=narration, lang="id")
        tts.save(str(audio_path))

        # 2. Buat gambar slide
        create_slide_image(narration, image_path)

        # 3. Gabungkan menjadi video MP4 hemat RAM
        audio_clip = AudioFileClip(str(audio_path))
        image_clip = ImageClip(str(image_path)).set_duration(audio_clip.duration)
        video_clip = image_clip.set_audio(audio_clip)

        video_clip.write_videofile(
            str(video_path),
            fps=15,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=1,
            logger=None,
        )

        video_clip.close()
        audio_clip.close()
        image_clip.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membuat video: {str(e)}")
    finally:
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)
        if image_path.exists():
            image_path.unlink(missing_ok=True)

    return {
        "script": narration,
        "video_url": f"/videos/{video_path.name}",
    }


@app.get("/videos/{filename}")
async def get_video(filename: str):
    safe_name = Path(filename).name
    target = VIDEO_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Video tidak ditemukan.")
    return FileResponse(target, media_type="video/mp4")