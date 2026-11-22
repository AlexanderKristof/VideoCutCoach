"""
Az annotált elemzés-videó írása (KIMENETI KÓDOLÁS): ffmpeg rawvideo-csövön, külön író szálon.

* Kódoló: a fast_cut felderítése szerint (h264_nvenc > h264_qsv > h264_amf > libx264, az utóbbi
  -preset veryfast), és ha a (pl. LGPL) ffmpeg-ben nincs libx264, a libopenh264. A FAST_CUT_ENCODER
  környezeti változó itt is érvényes ("cpu" = szoftveres).
* Minőség: nvenc -preset p4 -rc vbr -cq 23; qsv -global_quality 23; amf cqp 22/24; libx264 -crf 23.
* output_scale (pl. 0.5): kicsinyített annotált videó (1920x1080 -> 960x540) – kevesebb kódolás /
  csőforgalom / lemez. A kicsinyítés az író szálon történik (az OpenCV elengedi a GIL-t).
* Az író szál kis, korlátos sorból dolgozik (alap 4 képkocka): a fő ciklus nem vár a cső írására,
  amíg a sor nem telik meg (nagy, 6 MB-os 1080p képkockáknál ez számít).
"""
import queue
import subprocess
import threading

import cv2

DEFAULT_FFMPEG = "ffmpeg.exe"   # a main.py eddig is így hívta (a tesztek szimbolikus linket tesznek a PATH-ra)

_choice_cache = {}


def _available(ffmpeg_bin):
    try:
        import fast_cut
        return fast_cut.list_encoders(ffmpeg_bin)
    except Exception:
        return set()


def choose_h264_encoder(ffmpeg_bin=DEFAULT_FFMPEG):
    """A használható H.264 kódoló neve (gyorsítótárazva)."""
    if ffmpeg_bin in _choice_cache:
        return _choice_cache[ffmpeg_bin]
    name = "libx264"
    try:
        import fast_cut
        cands = fast_cut.encoder_candidates("h264", ffmpeg_bin)
        name = cands[0] if cands else "libx264"
        avail = fast_cut.list_encoders(ffmpeg_bin)
        if name == "libx264" and avail and "libx264" not in avail and "libopenh264" in avail:
            name = "libopenh264"
    except Exception:
        pass
    _choice_cache[ffmpeg_bin] = name
    return name


def software_h264_encoder(ffmpeg_bin=DEFAULT_FFMPEG):
    """Szoftveres H.264 tartalék: libx264 (GPL ffmpeg) vagy – LGPL ffmpeg-ben – libopenh264."""
    try:
        import fast_cut
        avail = fast_cut.list_encoders(ffmpeg_bin)
        if avail and "libx264" not in avail and "libopenh264" in avail:
            return "libopenh264"
    except Exception:
        pass
    return "libx264"


def encoder_args(name, width, height):
    if name == "h264_nvenc":
        return ["-c:v", name, "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0", "-pix_fmt", "yuv420p"]
    if name == "h264_qsv":
        return ["-c:v", name, "-preset", "veryfast", "-global_quality", "23", "-pix_fmt", "nv12"]
    if name == "h264_amf":
        return ["-c:v", name, "-quality", "speed", "-rc", "cqp", "-qp_i", "22", "-qp_p", "24", "-pix_fmt", "nv12"]
    if name == "libopenh264":
        mbps = max(1.0, 4.0 * width * height / (1920.0 * 1080.0))
        return ["-c:v", name, "-b:v", f"{mbps:.1f}M", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]


def output_size(width, height, scale):
    scale = float(scale or 1.0)
    if not (0.05 < scale <= 1.0):
        scale = 1.0
    w = max(2, int(round(width * scale / 2.0)) * 2)
    h = max(2, int(round(height * scale / 2.0)) * 2)
    return w, h


class AnnotatedVideoWriter:
    def __init__(self, path, width, height, fps, output_scale=1.0, ffmpeg_bin=DEFAULT_FFMPEG,
                 encoder=None, queue_size=4):
        self.path = path
        self.in_size = (int(width), int(height))
        self.size = output_size(width, height, output_scale)
        self.resize = self.size != self.in_size
        self.encoder = encoder or choose_h264_encoder(ffmpeg_bin)
        self.ffmpeg_bin = ffmpeg_bin
        self.fps = fps
        self.frames = 0
        self.error = None
        self.fallback_used = False
        self._q = queue.Queue(maxsize=max(1, int(queue_size)))
        self.proc = self._start(self.encoder)
        self._thread = threading.Thread(target=self._run, name="annotalt-video-iro", daemon=True)
        self._thread.start()

    def _start(self, enc):
        w, h = self.size
        cmd = [self.ffmpeg_bin, "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(self.fps), "-i", "-"] + \
            encoder_args(enc, w, h) + ["-movflags", "+faststart", self.path]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def describe(self):
        w, h = self.size
        return (f"{self.encoder}{' (tartalék)' if self.fallback_used else ''}, {w}x{h}"
                + (f" (kicsinyítve {self.in_size[0]}x{self.in_size[1]}-ről)" if self.resize else ""))

    def _run(self):
        while True:
            frame = self._q.get()
            if frame is None:
                break
            if self.error is not None:
                continue
            try:
                if self.resize:
                    frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
                self.proc.stdin.write(memoryview(frame).cast("B") if frame.flags["C_CONTIGUOUS"] else frame.tobytes())
                self.frames += 1
            except (BrokenPipeError, OSError, ValueError) as ex:
                # a hardveres kódoló menet közben hibázott: újraindítás szoftveresen (a már írt rész elvész,
                # de az elemzés nem áll le) – csak az elején érdemes, később csak naplózunk
                sw_enc = software_h264_encoder(self.ffmpeg_bin)
                if self.frames < 5 and self.encoder != sw_enc and not self.fallback_used:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                    self.fallback_used = True
                    self.encoder = sw_enc
                    self.proc = self._start(sw_enc)
                    self.frames = 0
                    try:
                        self.proc.stdin.write(frame.tobytes())
                        self.frames += 1
                        continue
                    except Exception as ex2:
                        ex = ex2
                self.error = f"{type(ex).__name__}: {ex}"

    def write(self, frame):
        """A képkocka sorba állítása (a hívó utána NEM módosíthatja a tömböt)."""
        self._q.put(frame)

    def close(self):
        self._q.put(None)
        self._thread.join()
        err = b""
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            err = self.proc.stderr.read() or b""
            self.proc.wait(timeout=600)
        except Exception:
            pass
        if self.proc.returncode not in (0, None) and self.error is None:
            self.error = (err or b"").decode("utf-8", "replace").strip()[-300:] or f"ffmpeg kilépési kód {self.proc.returncode}"
        return self.error

