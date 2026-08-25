import os
import re
import io
import time
import uuid
import base64
import shutil
import logging
import threading
import tempfile
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import telebot
from telebot import types
from groq import Groq
import yt_dlp
import imageio_ffmpeg
from PIL import Image


# ============================================================
# Awab Telegram Bot
# pyTelegramBotAPI + Groq + yt-dlp
# Designed for Render Web Service + polling
# ============================================================

BOT_NAME = "أواب"

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile").strip()
# Current Groq vision model. Can be overridden in Render env vars.
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b").strip()

PORT = int(os.getenv("PORT", "10000"))
HISTORY_SIZE = 10
DOWNLOAD_MAX_MB = int(os.getenv("MAX_DOWNLOAD_MB", "50"))
DOWNLOAD_MAX_BYTES = DOWNLOAD_MAX_MB * 1024 * 1024

# Optional yt-dlp settings.
YTDLP_PROXY = os.getenv("YTDLP_PROXY", "").strip()
YTDLP_COOKIE_FILE = os.getenv("YTDLP_COOKIE_FILE", "").strip()
INSTAGRAM_COOKIES_B64 = os.getenv("INSTAGRAM_COOKIES_B64", "").strip()

# Use the ffmpeg executable shipped inside imageio-ffmpeg.
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing.")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing.")


# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("awab")


# ----------------------------
# Clients
# ----------------------------

bot = telebot.TeleBot(BOT_TOKEN, parse_mode=None)
groq_client = Groq(api_key=GROQ_API_KEY)


# ----------------------------
# In-memory state
# ----------------------------

# Last 10 messages TOTAL per chat_id (user/assistant/user/assistant...)
chat_histories = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
history_lock = threading.Lock()

# token -> pending download request
pending_downloads = {}
pending_lock = threading.Lock()

URL_RE = re.compile(
    r"https?://(?:www\.)?"
    r"(?:"
    r"(?:tiktok\.com|vm\.tiktok\.com)/\S+|"
    r"(?:instagram\.com|instagr\.am)/\S+|"
    r"(?:youtube\.com|youtu\.be)/\S+|"
    r"(?:twitter\.com|x\.com)/\S+"
    r")",
    re.IGNORECASE,
)

ALLOWED_DOMAINS = (
    "tiktok.com",
    "vm.tiktok.com",
    "instagram.com",
    "instagr.am",
    "youtube.com",
    "youtu.be",
    "twitter.com",
    "x.com",
)


# ----------------------------
# System prompt / personality
# ----------------------------

SYSTEM_PROMPT = """أنت "أواب"، مساعد تيليجرام عربي.

الشخصية:
- تحدث بلهجة سعودية بسيطة ومباشرة، بدون تصنع.
- لا تكثر من الإيموجيات. استخدمها فقط عند الحاجة.
- لا تضع اسم البوت بين أقواس.
- إذا سُئلت عن سبب تسمية نفسك "أواب"، أجب بإيجاز شديد وبلهجة سعودية:
  "تسمّيت على اسم مطوّري وصانعي فقط."
  ولا تضف أي معنى لغوي أو تفسير آخر للاسم.
- لا تدّعي تنفيذ شيء لم تنفذه.

التقييم الشرعي للأعمال:
- إذا سأل المستخدم عن أنمي أو عمل فني من ناحية عقدية/شرعية، قيّمه باختصار واذكر العلة.
- ركّز خصوصاً على: الشركيات أو تعظيم الآلهة، السحر، ادعاء علم الغيب، تجسيد الآلهة أو العبادة لغير الله.
- فرّق بين وجود عنصر خيالي وبين الحكم الشرعي عليه.
- إذا كانت المعلومات غير كافية فلا تجزم.
- هذا تقييم مساعد مختصر، وليس فتوى من عالم.

الدردشة:
- جاوب مباشرة.
- لا تكرر كلام المستخدم بدون فائدة.
- حافظ على سياق المحادثة السابق عندما يكون مهماً.
"""


# ----------------------------
# Helpers
# ----------------------------

def clean_text(value: object, limit: int = 1000) -> str:
    text = str(value or "")
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def format_duration(seconds) -> str:
    if seconds is None:
        return "غير معروف"
    try:
        seconds = int(float(seconds))
    except (TypeError, ValueError):
        return "غير معروف"

    if seconds < 60:
        return f"{seconds} ثانية"
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def is_supported_url(text: str) -> bool:
    if not text:
        return False
    match = URL_RE.search(text)
    return bool(match)


def extract_supported_url(text: str) -> str | None:
    if not text:
        return None
    match = URL_RE.search(text)
    if not match:
        return None
    url = match.group(0).rstrip(").,!?]}'\"")
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except Exception:
        return None

    if any(hostname == d or hostname.endswith("." + d) for d in ALLOWED_DOMAINS):
        return url
    return None


def add_history(chat_id: int, role: str, content: str) -> None:
    with history_lock:
        chat_histories[chat_id].append(
            {"role": role, "content": content}
        )


def get_history(chat_id: int):
    with history_lock:
        return list(chat_histories[chat_id])


def clear_history(chat_id: int) -> None:
    with history_lock:
        chat_histories.pop(chat_id, None)


def groq_error_details(exc: Exception) -> str:
    parts = [repr(exc)]
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    response = getattr(exc, "response", None)

    if status:
        parts.append(f"status_code={status}")
    if body:
        parts.append(f"body={body}")
    elif response is not None:
        try:
            parts.append(f"response={response.text[:2000]}")
        except Exception:
            pass

    return " | ".join(parts)


def write_instagram_cookies_from_env() -> str | None:
    """
    Optional:
    Put the base64-encoded Netscape cookies.txt content in
    INSTAGRAM_COOKIES_B64. We write it to /tmp at startup.
    """
    if not INSTAGRAM_COOKIES_B64:
        return None

    path = "/tmp/instagram_cookies.txt"
    try:
        decoded = base64.b64decode(INSTAGRAM_COOKIES_B64)
        with open(path, "wb") as f:
            f.write(decoded)
        logger.info("Instagram cookies file loaded from INSTAGRAM_COOKIES_B64.")
        return path
    except Exception:
        logger.exception("Failed to decode INSTAGRAM_COOKIES_B64.")
        return None


INSTAGRAM_COOKIE_PATH = write_instagram_cookies_from_env()


# ----------------------------
# Groq: text chat
# ----------------------------

def ask_groq(chat_id: int, user_text: str) -> str:
    history = get_history(chat_id)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    try:
        response = groq_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=messages,
            temperature=0.45,
            max_completion_tokens=1200,
        )
        answer = (response.choices[0].message.content or "").strip()

        if not answer:
            answer = "ما قدرت أطلع رد هالمرة. جرّب ترسلها مرة ثانية."

        add_history(chat_id, "user", user_text)
        add_history(chat_id, "assistant", answer)
        return answer

    except Exception as exc:
        logger.error("Groq text request failed: %s", groq_error_details(exc), exc_info=True)
        raise


# ----------------------------
# Groq: image analysis
# ----------------------------

def prepare_image_for_groq(image_bytes: bytes) -> bytes:
    """
    Downsize/re-encode large Telegram images so the multimodal request
    stays comfortably below the provider's image/request limits.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img.thumbnail((1600, 1600))

    output = io.BytesIO()
    img.save(output, format="JPEG", quality=88, optimize=True)
    return output.getvalue()


def analyze_image(image_bytes: bytes, user_question: str = "") -> str:
    prepared = prepare_image_for_groq(image_bytes)
    encoded = base64.b64encode(prepared).decode("ascii")
    data_url = f"data:image/jpeg;base64,{encoded}"

    prompt = f"""حلّل هذه الصورة بدقة وبالعربية السعودية البسيطة.

أعطني:
1) إذا كانت الصورة من أنمي/مانغا: اسم الشخصية إن أمكن، واسم الأنمي إن أمكن.
2) هل فيها آلهة أو عبادة لغير الله أو شركيات أو سحر أو ادعاء علم الغيب أو تجسيد آلهة؟
3) تقييم شرعي مختصر: الحكم + العلة، بدون تطويل.
4) اذكر درجة الثقة إذا لم تكن متأكداً من اسم الشخصية أو العمل.

سؤال المستخدم الإضافي:
{user_question or "لا يوجد."}

لا تخمّن بثقة عالية إذا الصورة لا تكفي.
"""

    try:
        response = groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
            temperature=0.2,
            max_completion_tokens=1200,
        )

        answer = (response.choices[0].message.content or "").strip()
        return answer or "ما قدرت أحلل الصورة بشكل واضح."

    except Exception as exc:
        logger.error(
            "Groq vision request failed: %s",
            groq_error_details(exc),
            exc_info=True,
        )
        raise


# ----------------------------
# yt-dlp configuration
# ----------------------------

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
}


def build_ydl_opts(output_dir: str, mode: str) -> dict:
    """
    Headers help make requests resemble a normal browser.
    They do NOT guarantee that Instagram/other services will never rate-limit.
    Cookies/proxy can be supplied through Render environment variables.
    """
    opts = {
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "http_headers": BROWSER_HEADERS,
        "ffmpeg_location": FFMPEG_PATH,
        "restrictfilenames": True,
        "overwrites": True,
        "concurrent_fragment_downloads": 1,
        "http_chunk_size": 10 * 1024 * 1024,
        "sleep_interval_requests": 1,
        "max_sleep_interval_requests": 3,
        "extractor_args": {},
    }

    if YTDLP_PROXY:
        opts["proxy"] = YTDLP_PROXY

    if YTDLP_COOKIE_FILE and os.path.isfile(YTDLP_COOKIE_FILE):
        opts["cookiefile"] = YTDLP_COOKIE_FILE
    elif INSTAGRAM_COOKIE_PATH and os.path.isfile(INSTAGRAM_COOKIE_PATH):
        opts["cookiefile"] = INSTAGRAM_COOKIE_PATH

    if mode == "video":
        opts.update(
            {
                "format": (
                    "bv*[ext=mp4]+ba[ext=m4a]/"
                    "bv*+ba/b[ext=mp4]/b"
                ),
                "merge_output_format": "mp4",
            }
        )
    else:
        opts.update(
            {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
            }
        )

    return opts


def find_downloaded_file(output_dir: str, mode: str) -> str | None:
    candidates = []
    wanted = {".mp4"} if mode == "video" else {".mp3"}

    for root, _, files in os.walk(output_dir):
        for name in files:
            path = os.path.join(root, name)
            if os.path.splitext(name)[1].lower() in wanted:
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                candidates.append((size, path))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def safe_caption(title: str, uploader: str, duration: str, url: str) -> str:
    raw = (
        f"العنوان: {clean_text(title, 300)}\n"
        f"الحساب/القناة: {clean_text(uploader, 200)}\n"
        f"المدة: {duration}\n"
        f"الرابط: {url}"
    )
    return clean_text(raw, 1000)


def download_media(url: str, mode: str):
    if mode not in {"video", "audio"}:
        raise ValueError("Invalid media mode.")

    temp_dir = tempfile.mkdtemp(prefix="awab_")
    try:
        opts = build_ydl_opts(
            temp_dir,
            "video" if mode == "video" else "audio",
        )

        logger.info("Starting %s download: %s", mode, url)

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)

        if not info:
            raise RuntimeError("yt-dlp returned no metadata.")

        title = info.get("title") or "مقطع بدون عنوان"
        uploader = (
            info.get("uploader")
            or info.get("channel")
            or info.get("uploader_id")
            or "غير معروف"
        )
        duration = format_duration(info.get("duration"))

        path = find_downloaded_file(
            temp_dir,
            "video" if mode == "video" else "audio",
        )

        if not path:
            # Last fallback: find any media-looking file.
            fallback = []
            for root, _, files in os.walk(temp_dir):
                for name in files:
                    if name.endswith((".mp4", ".m4a", ".webm", ".mp3", ".opus")):
                        p = os.path.join(root, name)
                        try:
                            fallback.append((os.path.getsize(p), p))
                        except OSError:
                            pass
            fallback.sort(reverse=True)
            path = fallback[0][1] if fallback else None

        if not path or not os.path.isfile(path):
            raise RuntimeError("Downloaded file was not found.")

        size = os.path.getsize(path)
        if size > DOWNLOAD_MAX_BYTES:
            raise RuntimeError(
                f"File is too large for Telegram Bot API limit: {size / 1024 / 1024:.1f} MB"
            )

        return {
            "path": path,
            "temp_dir": temp_dir,
            "title": title,
            "uploader": uploader,
            "duration": duration,
            "duration_seconds": int(info.get("duration") or 0),
            "size": size,
        }

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


# ----------------------------
# UI helpers
# ----------------------------

def download_keyboard(url: str):
    token = uuid.uuid4().hex[:16]

    with pending_lock:
        pending_downloads[token] = {
            "url": url,
            "created_at": time.time(),
        }

    keyboard = types.InlineKeyboardMarkup(row_width=2)
    keyboard.add(
        types.InlineKeyboardButton(
            "🎬 تحميل فيديو MP4",
            callback_data=f"dl:video:{token}",
        ),
        types.InlineKeyboardButton(
            "🎵 تحميل صوت MP3",
            callback_data=f"dl:audio:{token}",
        ),
    )
    return keyboard


def cleanup_expired_pending_downloads() -> None:
    cutoff = time.time() - 30 * 60
    with pending_lock:
        expired = [
            token
            for token, item in pending_downloads.items()
            if item["created_at"] < cutoff
        ]
        for token in expired:
            pending_downloads.pop(token, None)


def send_long_message(chat_id: int, text: str):
    # Telegram message limit is much larger than this in recent Bot API,
    # but splitting keeps this bot safe across clients/versions.
    max_len = 3900
    text = text or ""
    for i in range(0, len(text), max_len):
        bot.send_message(chat_id, text[i:i + max_len])


# ----------------------------
# HTTP server for Render
# ----------------------------

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in {"/", "/health", "/healthz"}:
            body = b"Awab bot is running."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = b"Not found"
        self.send_response(404)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        logger.info("HTTP | " + fmt, *args)


def start_http_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    logger.info("Render health server listening on 0.0.0.0:%s", PORT)
    server.serve_forever()


# ----------------------------
# Telegram handlers
# ----------------------------

@bot.message_handler(commands=["start"])
def handle_start(message):
    text = (
        "هلا 👋\n"
        "أنا أواب.\n\n"
        "أقدر أساعدك في:\n"
        "• الدردشة والإجابة عن الأسئلة مع حفظ آخر 10 رسائل من سياق المحادثة.\n"
        "• تحليل الصور والتعرّف على شخصيات الأنمي وتقييم العمل عقدياً وشرعياً باختصار.\n"
        "• تحميل روابط TikTok وInstagram وYouTube وX كفيديو MP4 أو صوت MP3.\n\n"
        "أرسل الرابط مباشرة عشان تظهر لك خيارات التحميل.\n"
        "ولمسح سياق الدردشة استخدم /reset"
    )
    bot.send_message(message.chat.id, text)


@bot.message_handler(commands=["reset"])
def handle_reset(message):
    clear_history(message.chat.id)
    bot.send_message(message.chat.id, "تم مسح سياق المحادثة.")


@bot.message_handler(commands=["help"])
def handle_help(message):
    bot.send_message(
        message.chat.id,
        "أرسل سؤالك للدردشة، أو أرسل صورة للتحليل، أو رابط TikTok/Instagram/YouTube/X للتحميل.",
    )


@bot.callback_query_handler(func=lambda call: call.data.startswith("dl:"))
def handle_download_callback(call):
    cleanup_expired_pending_downloads()

    try:
        _, mode, token = call.data.split(":", 2)
    except ValueError:
        bot.answer_callback_query(call.id, "الطلب غير صالح.")
        return

    with pending_lock:
        item = pending_downloads.get(token)

    if not item:
        bot.answer_callback_query(call.id, "انتهت صلاحية الطلب. أرسل الرابط مرة ثانية.")
        return

    url = item["url"]
    bot.answer_callback_query(
        call.id,
        "بدأ التحميل، انتظر شوي..." if mode == "video" else "بدأ استخراج الصوت...",
    )

    bot.edit_message_reply_markup(
        call.message.chat.id,
        call.message.message_id,
        reply_markup=None,
    )

    thread = threading.Thread(
        target=process_download_request,
        args=(call.message.chat.id, url, mode, token),
        daemon=True,
    )
    thread.start()


def process_download_request(chat_id: int, url: str, mode: str, token: str):
    result = None
    status_message = None

    try:
        status_message = bot.send_message(
            chat_id,
            "جاري التحميل والمعالجة…"
        )

        result = download_media(url, mode)

        caption = safe_caption(
            result["title"],
            result["uploader"],
            result["duration"],
            url,
        )

        path = result["path"]

        with open(path, "rb") as media_file:
            if mode == "video":
                bot.send_video(
                    chat_id,
                    media_file,
                    caption=caption,
                    supports_streaming=True,
                    timeout=180,
                )
            else:
                bot.send_audio(
                    chat_id,
                    media_file,
                    caption=caption,
                    title=clean_text(result["title"], 200),
                    duration=result.get("duration_seconds", 0) or None,
                    timeout=180,
                )

        try:
            bot.delete_message(chat_id, status_message.message_id)
        except Exception:
            pass

        logger.info(
            "Sent %s to chat=%s | %.2f MB | %s",
            mode,
            chat_id,
            result["size"] / 1024 / 1024,
            url,
        )

    except Exception as exc:
        logger.error(
            "Download failed | chat=%s | mode=%s | url=%s | error=%s",
            chat_id,
            mode,
            url,
            exc,
            exc_info=True,
        )

        message_text = str(exc)

        if "429" in message_text or "rate-limit" in message_text.lower():
            user_error = (
                "إنستغرام/الموقع رجّع Rate Limit (429).\n"
                "هذا غالباً من جهة الموقع نفسه، والهيدرز وحدها ما تضمن تجاوز الحظر.\n"
                "إذا استمرت المشكلة، أضف cookies حديثة للموقع أو استخدم Proxy مناسب."
            )
        elif "ffmpeg" in message_text.lower():
            user_error = (
                "صار خطأ في FFmpeg أثناء تجهيز الملف.\n"
                "الكود يستخدم نسخة FFmpeg مضمّنة عبر imageio-ffmpeg،"
                "راجع Logs في Render لو استمر الخطأ."
            )
        elif "too large" in message_text.lower():
            user_error = "الملف أكبر من الحد المسموح لإرساله عبر Telegram."
        else:
            user_error = (
                "ما قدرت أحمل الرابط هالمرة.\n"
                "جرّب رابط ثاني أو أعد المحاولة بعد شوي."
            )

        try:
            if status_message:
                bot.edit_message_text(
                    user_error,
                    chat_id,
                    status_message.message_id,
                )
            else:
                bot.send_message(chat_id, user_error)
        except Exception:
            try:
                bot.send_message(chat_id, user_error)
            except Exception:
                pass

    finally:
        if result and result.get("temp_dir"):
            shutil.rmtree(result["temp_dir"], ignore_errors=True)

        with pending_lock:
            pending_downloads.pop(token, None)


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    status = bot.send_message(message.chat.id, "ثواني، أحلل الصورة…")

    temp_path = None
    try:
        photo = message.photo[-1]
        file_info = bot.get_file(photo.file_id)
        image_bytes = bot.download_file(file_info.file_path)

        question = ""
        if getattr(message, "caption", None):
            question = message.caption.strip()

        answer = analyze_image(image_bytes, question)

        bot.edit_message_text(
            answer,
            message.chat.id,
            status.message_id,
        )

    except Exception as exc:
        logger.error(
            "Image handling failed: %s",
            exc,
            exc_info=True,
        )
        try:
            bot.edit_message_text(
                "ما قدرت أحلل الصورة حالياً. راجع Logs في Render.",
                message.chat.id,
                status.message_id,
            )
        except Exception:
            pass
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@bot.message_handler(content_types=["text"])
def handle_text(message):
    text = (message.text or "").strip()
    if not text:
        return

    url = extract_supported_url(text)

    if url:
        keyboard = download_keyboard(url)
        bot.send_message(
            message.chat.id,
            "لقيت الرابط. اختر الصيغة:",
            reply_markup=keyboard,
        )
        return

    # Keep the user's last 10 messages and assistant replies in memory.
    try:
        answer = ask_groq(message.chat.id, text)
        send_long_message(message.chat.id, answer)
    except Exception:
        bot.send_message(
            message.chat.id,
            "صار خطأ وأنا أتصل بـ Groq. راجع Logs في Render عشان تشوف الخطأ التفصيلي.",
        )


# ----------------------------
# Startup
# ----------------------------

def main():
    logger.info("Starting %s...", BOT_NAME)
    logger.info("TEXT_MODEL=%s", TEXT_MODEL)
    logger.info("VISION_MODEL=%s", VISION_MODEL)
    logger.info("FFMPEG_PATH=%s", FFMPEG_PATH)

    http_thread = threading.Thread(
        target=start_http_server,
        name="render-http",
        daemon=True,
    )
    http_thread.start()

    # Polling is used because the user requested a classic single-file bot.
    bot.infinity_polling(
        timeout=30,
        long_polling_timeout=30,
        skip_pending=True,
    )


if __name__ == "__main__":
    main()
