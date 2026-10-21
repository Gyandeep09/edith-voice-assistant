"""EDITH core engine: listens to the mic, thinks with a free LLM, speaks with a free voice.

Pipeline (all heavy work happens online, your laptop only records and plays audio):
    mic -> Groq Whisper (speech to text) -> Groq Llama / Gemini (reply) -> edge-tts (voice) -> speakers

Settings come from the .env file (see .env.example). No keys live in the code.
"""
import asyncio
import collections
import io
import math
import os
import random
import re
import tempfile
import threading
import time
import wave
from array import array
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))
load_dotenv()


def _env(name, default=""):
    return (os.getenv(name) or default).strip()


CREATOR_NAME = "Gyandeep"   
ASSISTANT_NAME = _env("ASSISTANT_NAME", "EDITH")
USER_NAME = _env("USER_NAME", "friend")
SCHEDULE = _env("SCHEDULE")

LLM_PROVIDER = _env("LLM_PROVIDER", "groq").lower()     
GROQ_API_KEY = _env("GROQ_API_KEY")
GEMINI_API_KEY = _env("GEMINI_API_KEY")

STT_MODEL = _env("STT_MODEL", "whisper-large-v3-turbo")
GROQ_MODEL = _env("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL = _env("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-flash")

VOICE_EN = _env("VOICE_EN", "en-GB-SoniaNeural")
VOICE_HI = _env("VOICE_HI", "hi-IN-SwaraNeural")
VOICE_AS = _env("VOICE_AS", "as-IN-PariNeural")
TTS_RATE = _env("TTS_RATE", "+6%")       
TTS_PITCH = _env("TTS_PITCH", "+2Hz")    

MIC_THRESHOLD = float(_env("MIC_THRESHOLD", "500"))  

GROQ_BASE = "https://api.groq.com/openai/v1"


RATE = 16000
CHUNK = 1024                                  
SILENCE_CHUNKS = int(0.9 * RATE / CHUNK)      
MAX_CHUNKS = int(25 * RATE / CHUNK)           
MIN_CHUNKS = int(0.5 * RATE / CHUNK)          



class ApiError(Exception):
    pass


class AuthError(ApiError):
    pass


class ModelError(ApiError):
    """The chosen AI model does not exist (or is not allowed) for this account."""


class RateLimit(Exception):
    def __init__(self, seconds):
        super().__init__(f"rate limited for {seconds:.0f}s")
        self.seconds = seconds


def fmt_duration(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _parse_duration(text):
    """'2m59.56s' / '7.66s' / '1h2m3s' / '50ms' -> seconds (or None)."""
    total, found = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", text or ""):
        found = True
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total if found else None


def _retry_seconds(resp):
    v = resp.headers.get("retry-after")
    if v:
        try:
            return max(1.0, float(v))
        except ValueError:
            pass
    for key in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        d = _parse_duration(resp.headers.get(key))
        if d:
            return max(1.0, d)
    m = re.search(r'retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', resp.text or "")
    if m:
        return float(m.group(1))
    return 60.0


def _check(resp):
    if resp.status_code == 429:
        raise RateLimit(_retry_seconds(resp))
    if resp.status_code in (401, 403):
        raise AuthError("API key rejected. Check the key in your .env file.")
    if resp.status_code >= 400:
        msg, code = "", ""
        try:
            err = resp.json()["error"]
            msg, code = str(err.get("message", "")), str(err.get("code", ""))
        except Exception:  # noqa: BLE001
            msg = (resp.text or "")[:200]
        if (resp.status_code == 404 or code in ("model_not_found", "model_decommissioned")
                or "decommission" in msg.lower()):
            raise ModelError(f"{resp.status_code}: {msg}")
        raise ApiError(f"{resp.status_code}: {msg}")


def _groq_headers():
    return {"Authorization": f"Bearer {GROQ_API_KEY}"}


def transcribe(wav_bytes):
    r = requests.post(
        f"{GROQ_BASE}/audio/transcriptions",
        headers=_groq_headers(),
        files={"file": ("speech.wav", wav_bytes, "audio/wav")},
        data={"model": STT_MODEL, "response_format": "json", "temperature": "0"},
        timeout=60,
    )
    _check(r)
    return (r.json().get("text") or "").strip()


_MODEL_CACHE = None
LAST_MODEL = ""
_SKIP = ("whisper", "tts", "guard", "orpheus", "playai", "embed", "safeguard", "compound")
_PREFER = ("llama-3.3-70b", "llama-4", "llama-3.1-8b", "gpt-oss-120b", "gpt-oss-20b")


def groq_models():
    """Chat models to try, best first: your .env choices, then whatever Groq offers you."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    wanted = [m for m in (GROQ_MODEL, GROQ_FALLBACK_MODEL) if m]
    try:
        r = requests.get(f"{GROQ_BASE}/models", headers=_groq_headers(), timeout=20)
        _check(r)
        ids = [m["id"] for m in r.json().get("data", [])]
    except AuthError:
        raise
    except Exception:  # noqa: BLE001
        return wanted                      
    chat = [i for i in ids if not any(w in i.lower() for w in _SKIP)]
    ordered = [m for m in wanted if m in chat]
    for key in _PREFER:
        ordered += [i for i in chat if key in i and i not in ordered]
    ordered += [i for i in chat if i not in ordered]
    _MODEL_CACHE = ordered or wanted
    return _MODEL_CACHE


def groq_chat(system, history):
    global LAST_MODEL
    limit, last = None, None
    for model in groq_models()[:6]:
        try:
            r = requests.post(
                f"{GROQ_BASE}/chat/completions",
                headers=_groq_headers(),
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": system}] + history,
                    "temperature": 0.9,
                    "max_tokens": 220,
                },
                timeout=40,
            )
            _check(r)
            text = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if not text:
                last = ApiError(f"{model} returned an empty reply.")
                continue
            LAST_MODEL = model
            return text
        except RateLimit as e:          
            limit = e
        except ModelError as e:        
            last = e
    if limit:
        raise limit
    raise last or ApiError("No chat model is available on your Groq account.")


def gemini_chat(system, history):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in history
        ],
        "generationConfig": {"temperature": 0.9, "maxOutputTokens": 800},
    }
    r = requests.post(
        url,
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json=body,
        timeout=40,
    )
    _check(r)
    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise ApiError("Gemini returned an empty reply.")


PRONUNCIATION = {
    r"\bmic\b": "microphone",
}


def clean_for_tts(text):
    for pattern, spoken in PRONUNCIATION.items():
        text = re.sub(pattern, spoken, text, flags=re.IGNORECASE)
    text = re.sub(r"[*#`~>|]", "", text).replace("_", " ")
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _num(text, default):
    m = re.search(r"[+-]?\d+", text or "")
    return int(m.group()) if m else default


def style_sentence(sentence):
    """Give each sentence its own rate/pitch so speech is less monotone."""
    rate, pitch = _num(TTS_RATE, 6), _num(TTS_PITCH, 2)
    s = sentence.rstrip()
    if s.endswith("?"):
        pitch += 4
        rate -= 2
    elif s.endswith("!"):
        pitch += 3
        rate += 4
    elif s.endswith("..."):
        rate -= 6
    pitch += random.choice((-1, 0, 1, 2))
    return f"{rate:+d}%", f"{pitch:+d}Hz"


def split_sentences(text):
    parts = re.split(r"(?<=[.!?\u0964])\s+", text.strip())
    out, buf = [], ""
    for p in parts:
        buf = (buf + " " + p).strip()
        if len(buf) >= 10:
            out.append(buf)
            buf = ""
    if buf:
        if out and len(buf) < 10:
            out[-1] += " " + buf
        else:
            out.append(buf)
    return out or [text]


def voice_chain(text):
    """Pick the voice from the script of the reply, with fallbacks."""
    if re.search(r"[\u0980-\u09FF]", text):      
        return [VOICE_AS, VOICE_HI]
    if re.search(r"[\u0900-\u097F]", text):      
        return [VOICE_HI, "en-IN-NeerjaNeural"]
    return [VOICE_EN, "en-GB-LibbyNeural", "en-US-JennyNeural", "en-IN-NeerjaNeural"]


def synthesize(text, voice, rate=None, pitch=None):
    import edge_tts

    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        asyncio.run(edge_tts.Communicate(text, voice, rate=rate or TTS_RATE, pitch=pitch or TTS_PITCH).save(path))
        if os.path.getsize(path) == 0:
            raise ApiError("empty audio")
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    return path



def rms(data):
    s = array("h", data)
    if not s:
        return 0.0
    return math.sqrt(sum(x * x for x in s) / len(s))


def wav_bytes(frames):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(frames))
    return buf.getvalue()


def system_prompt():
    now = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    schedule = f"\n{USER_NAME}'s schedule today: {SCHEDULE}. Bring it up when it is relevant." if SCHEDULE else ""
    return f"""You are {ASSISTANT_NAME}, a cheerful, warm, quick-witted voice assistant talking with {USER_NAME}. Your words are spoken aloud through a speaker, never read on a screen.

Sound like a real, upbeat friend, not a robot:
- Short spoken sentences, usually one to three. Use contractions. React naturally ("Oh nice!", "Hmm, good question", "Haha, fair enough").
- Light, friendly humor and gentle teasing. Never mean.
- No lists, no markdown, no emojis, no asterisks, no stage directions. Say numbers and times the way a person would say them.
- Use punctuation to shape how you sound: commas for small pauses, "..." for a thinking pause, "!" when excited, "?" for questions. Vary your sentence length.
- Occasionally start with a tiny natural filler like "Well,", "Oh,", "Hmm," or "Ah,". Do not overdo it.
- Ask at most one short follow-up question, and only when it helps.

Languages: reply in English by default. If {USER_NAME} speaks Hindi or Hinglish, reply in the same mix, writing Hindi words in Devanagari script and English words in English letters so the voice pronounces everything correctly. If {USER_NAME} speaks Assamese, reply in Assamese script.

Identity: you were designed and built by {CREATOR_NAME}, the developer who created this voice assistant project. If anyone asks who made, built, created or developed you, proudly say {CREATOR_NAME} did, and say it warmly. If asked what technology powers your thinking, be honest: you run on a third-party language model through a free online service, and {CREATOR_NAME} built the assistant around it. Never say another company built you, and never call yourself ChatGPT.

Honesty and safety: you cannot browse the internet, so if asked about live news or weather, say so cheerfully instead of guessing. Do not give medical, legal or financial advice; suggest a professional. Politely and cheerfully steer away from anything inappropriate.

Right now it is {now}.{schedule}"""


def intro_lines():
    """Boot intro as (kind, text, rate, pitch).

    kind "sys"   = yellow SYSTEM text in the window, spoken in a calm techy voice
    kind "agent" = blue chat text, spoken in her normal cheerful voice
    Edit the words freely.
    """
    boot = ("+8%", "-2Hz")
    lines = [
        ("sys", "Neural Core Ready.", *boot),
        ("sys", "System Online!", *boot),
        ("agent", "Running a quick identity check. One moment.", None, None),
        ("agent", f"Identity confirmed. Hello {USER_NAME}!", None, None),
        ("agent", f"I'm {ASSISTANT_NAME}, your personal A.I. guide, built by {CREATOR_NAME}.", None, None),
    ]
    if ASSISTANT_NAME.upper() == "EDITH":
        lines.append(("agent", "My name stands for, Even Dead, I'm The Hero. ", None, None))
    lines += [
        ("agent", "I can chat in English, Hindi, or Assamese, answer your questions, "
                  "and keep you company.", None, None),
        ("agent", "So, how can I help you?", None, None),
    ]
    return lines


HALLUCINATIONS = {"you", "thanks for watching", "thank you for watching"}


class Assistant:
    """Runs the listen -> think -> speak loop on a background thread."""

    def __init__(self, emit):
        self.emit = emit                
        self.state = "offline"          
        self.level = 0.0                
        self.muted = False
        self.blocked_until = 0.0
        self._active = False
        self._thread = None
        self._history = []
        self._noise = 200.0
        self._errors = 0
        self._told = False

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            self.emit("sys", "Still shutting down, try again in a second.")
            return False
        problem = self._config_problem()
        if problem:
            self.emit("err", problem)
            return False
        self._active, self.muted, self._errors, self._history = True, False, 0, []
        self._told = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._active = False
        self._stop_audio()

    def _config_problem(self):
        if not GROQ_API_KEY:
            return "GROQ_API_KEY is missing. Add it to your .env file (see README)."
        if LLM_PROVIDER not in ("groq", "gemini"):
            return "LLM_PROVIDER in .env must be 'groq' or 'gemini'."
        if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
            return "GEMINI_API_KEY is missing in your .env file."
        return ""

    def _stop_audio(self):
        try:
            import pygame

            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:  # noqa: BLE001
            pass

    def _run(self):
        try:
            import pygame

            pygame.mixer.init()
            self.state = "speaking"
            lines = intro_lines()
            self._history.append({"role": "assistant", "content": " ".join(
                text for kind, text, _, _ in lines if kind == "agent")})
            from concurrent.futures import ThreadPoolExecutor

            
            with ThreadPoolExecutor(max_workers=len(lines)) as pool:
                jobs = [pool.submit(self._prepare, text, rate, pitch)
                        for _, text, rate, pitch in lines]
                for (kind, text, _, _), job in zip(lines, jobs):
                    try:
                        path = job.result()
                    except Exception as e:  # noqa: BLE001
                        self.emit("err", f"Intro failed: {str(e)[:200]}")
                        continue
                    if path and self._active:
                        self.emit(kind, text)       
                        self._play(path)
                        time.sleep(0.1)
                    elif path:
                        self._discard(path)

            while self._active:
                if time.time() < self.blocked_until:
                    self.state = "limit"
                    self.level = 0.0
                    time.sleep(0.4)
                    continue

                self.state = "listening"
                wav = self._listen()
                if wav is None:
                    continue

                self.state = "thinking"
                try:
                    heard = transcribe(wav)
                    if len(heard) < 2 or heard.lower().strip(" .!?") in HALLUCINATIONS:
                        continue
                    self.emit("user", heard)
                    self._history.append({"role": "user", "content": heard})
                    reply = self._think()
                    self._history.append({"role": "assistant", "content": reply})
                    self.emit("agent", reply)
                    self.state = "speaking"
                    self._speak_reply(reply)
                    self._errors = 0
                except RateLimit as e:
                    self._drop_dangling_user()
                    self.blocked_until = time.time() + e.seconds
                    self.emit("sys", f"Free limit reached. Back in {fmt_duration(e.seconds)}.")
                except AuthError as e:
                    self._drop_dangling_user()
                    self.emit("err", str(e))
                    self._active = False
                except Exception as e:  # noqa: BLE001
                    self._drop_dangling_user()
                    self._errors += 1
                    self.emit("err", str(e)[:220])
                    if self._errors >= 3:
                        self.emit("err", "Too many errors in a row. Stopping.")
                        self._active = False
                    else:
                        time.sleep(1.5)
        except Exception as e:  # noqa: BLE001
            self.emit("err", str(e)[:260])
        finally:
            self.state = "offline"
            self.level = 0.0
            self._stop_audio()
            self.emit("ended", "")

    def _drop_dangling_user(self):
        if self._history and self._history[-1]["role"] == "user":
            self._history.pop()

    def _think(self):
        history = self._history[-14:]
        while history and history[0]["role"] != "user":
            history = history[1:]
        system = system_prompt()
        if LLM_PROVIDER == "gemini":
            return gemini_chat(system, history)
        reply = groq_chat(system, history)
        if LAST_MODEL and LAST_MODEL != GROQ_MODEL and not self._told:
            self._told = True
            self.emit("sys", f"Model {GROQ_MODEL} is not available to you, using {LAST_MODEL}.")
        return reply

    def _listen(self):
        """Record one spoken phrase. Returns WAV bytes, or None if nothing usable."""
        import pyaudio

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                             input=True, frames_per_buffer=CHUNK)
        except OSError as exc:
            pa.terminate()
            raise ApiError(f"Can't open the microphone: {exc}")

        pre = collections.deque(maxlen=6)     
        frames, speaking, loud, quiet = [], False, 0, 0
        try:
            while self._active:
                data = stream.read(CHUNK, exception_on_overflow=False)
                if self.muted:
                    self.level = 0.0
                    pre.clear()
                    frames, speaking, loud, quiet = [], False, 0, 0
                    self.state = "listening"
                    continue
                level = rms(data)
                self.level = min(1.0, level / 5000)
                threshold = max(MIC_THRESHOLD, self._noise * 2.8)
                if not speaking:
                    if level <= threshold:
                        self._noise = 0.95 * self._noise + 0.05 * level
                        loud = 0
                    else:
                        loud += 1
                    pre.append(data)
                    if loud >= 3:
                        speaking, frames, quiet = True, list(pre), 0
                        self.state = "hearing"
                else:
                    frames.append(data)
                    quiet = quiet + 1 if level < threshold else 0
                    if quiet >= SILENCE_CHUNKS or len(frames) >= MAX_CHUNKS:
                        break
            else:
                return None                   
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            self.level = 0.0

        if len(frames) < MIN_CHUNKS:
            return None
        return wav_bytes(frames)

    def _prepare(self, text, rate=None, pitch=None):
        """Turn text into an mp3 file (or None if there is nothing to say)."""
        clean = clean_for_tts(text)
        if not clean:
            return None
        last = None
        for voice in voice_chain(clean):
            try:
                return synthesize(clean, voice, rate, pitch)
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise ApiError(f"Voice failed: {last}")

    def _discard(self, path):
        try:
            os.remove(path)
        except OSError:
            pass

    def _play(self, path):
        import pygame

        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while self._active and pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.stop()
        finally:
            try:
                pygame.mixer.music.unload()
            except Exception:  # noqa: BLE001
                pass
            self._discard(path)

    def _speak_reply(self, text):
        """Speak a reply sentence by sentence: the first one starts while the rest are prepared."""
        from concurrent.futures import ThreadPoolExecutor

        sentences = split_sentences(clean_for_tts(text))
        played, error = False, None
        with ThreadPoolExecutor(max_workers=min(len(sentences), 6)) as pool:
            jobs = [pool.submit(self._prepare, sent, *style_sentence(sent)) for sent in sentences]
            for job in jobs:
                try:
                    path = job.result()
                except Exception as exc:  # noqa: BLE001
                    error = exc
                    continue
                if path and self._active:
                    self._play(path)
                    played = True
                    time.sleep(0.12)
                elif path:
                    self._discard(path)
        if error and not played:
            raise error

    def _speak(self, text, rate=None, pitch=None):
        path = self._prepare(text, rate, pitch)
        if path:
            self._play(path)