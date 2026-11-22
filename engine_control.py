"""
A motor vezérlése egy asztali gazdaalkalmazásból (MAUI): állapot, verzió, leállítás, modellek.

Végpontok (install_endpoints(app, engine) regisztrálja; engine = a main modul):
  GET  /health          – mindig gyors (nem tölt be modellt, nem importál torch-ot):
                          {"status": "ok", "version", "model_loaded", "model_loading", "model_error",
                           "device", "ffmpeg", "data_dir", "mode", "jobs_active", "shutting_down", "pid"}
  GET  /version         – {"name", "version", "api_version", "frozen", "python"}
  POST /shutdown        – csak helyi gépről (127.0.0.1 / ::1); ha TACTICAL_CONTROL_TOKEN be van
                          állítva, az X-TacAnaly-Token fejléc is kell. Kecses leállás: a futó jobok
                          megszakítása, legfeljebb grace_sec várakozás, a maradék "megszakadt"
                          állapotba, majd a szerver leáll. Válasz: {"status": "shutting_down", ...}
  GET  /models/status   – a beállított súlyok megvannak-e, a models mappa tartalma
  POST /models/download – háttérfeladat (job) a hiányzó hivatalos súly letöltésére, haladással;
                          paraméter (form / JSON / query): name (alap: az aktuális eszköz súlya)
  POST /models/load     – háttérfeladat: a modell betöltése előre (bemelegítés)

A leállítás és a szerver-indítás (serve) a launcherből és a python main.py-ból is ezt használja.
"""
import os
import sys
import json
import time
import signal
import threading

import app_paths
import model_manager

ENGINE_NAME = "TacAnaly Engine"
ENGINE_VERSION = "1.0.0"
API_VERSION = 1
ENV_CONTROL_TOKEN = "TACTICAL_CONTROL_TOKEN"
LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1")
ACTIVE_STATUSES = ("queued", "running")

_state = {"server": None, "shutting_down": False, "reason": None}
_state_lock = threading.Lock()


def version_info():
    return {
        "name": ENGINE_NAME,
        "version": ENGINE_VERSION,
        "api_version": API_VERSION,
        "frozen": app_paths.is_frozen(),
        "python": sys.version.split()[0],
    }


def is_local_client(host):
    if not host:
        return False
    return host in LOCAL_HOSTS or str(host).startswith("127.")


def token_ok(headers, env=None):
    expected = (os.environ if env is None else env).get(ENV_CONTROL_TOKEN)
    if not expected:
        return True
    got = headers.get("x-tacanaly-token") if headers is not None else None
    return got == expected


# ---------------------------------------------------------------- szerver / leállítás
def register_server(server):
    with _state_lock:
        _state["server"] = server


def is_shutting_down():
    return _state["shutting_down"]


def active_jobs(job_manager):
    if job_manager is None:
        return []
    return [m for m in job_manager.list(limit=0) if m.get("status") in ACTIVE_STATUSES]


def cancel_active_jobs(job_manager):
    ids = []
    for m in active_jobs(job_manager):
        job_manager.cancel(m["id"])
        ids.append(m["id"])
    return ids


def mark_interrupted(job_manager, reason="A motor leállt, mielőtt a feladat befejeződött."):
    """A még futó jobok lezárása 'megszakadt' állapottal (a következő indulás is ezt tenné)."""
    ids = []
    finish = getattr(job_manager, "_finish", None)
    for m in active_jobs(job_manager):
        if finish is not None:
            try:
                finish(m["id"], "megszakadt", error=reason, phase="Megszakadt (leállítás)")
                ids.append(m["id"])
            except Exception as ex:  # a leállást ne akassza meg
                print(f"⚠️ A job nem zárható le ({m['id']}): {ex}")
    return ids


def stop_server():
    server = _state["server"]
    if server is not None:
        server.should_exit = True
        return True
    # Nincs regisztrált szerver (pl. 'uvicorn main:app' indítás): a uvicorn SIGINT-kezelője áll le
    try:
        signal.raise_signal(signal.SIGINT)
        return True
    except Exception:
        return False


def _shutdown_worker(job_manager, grace_sec, delay_sec):
    time.sleep(delay_sec)  # a /shutdown válasza még kimehessen
    deadline = time.monotonic() + max(0.0, grace_sec)
    while active_jobs(job_manager) and time.monotonic() < deadline:
        cancel_active_jobs(job_manager)
        time.sleep(0.2)
    left = mark_interrupted(job_manager)
    if left:
        print(f"⚠️ Leállítás: {len(left)} feladat nem állt le időben – megszakadt állapotba téve.")
    stop_server()


def request_shutdown(job_manager=None, grace_sec=10.0, reason="", delay_sec=0.3):
    """
    Kecses leállás indítása háttérszálon. Visszatér: a megszakítást kérő job azonosítók
    (vagy None, ha a leállás már folyamatban van).
    """
    with _state_lock:
        if _state["shutting_down"]:
            return None
        _state["shutting_down"] = True
        _state["reason"] = reason
    print(f"🛑 Leállítás kérve{': ' + reason if reason else ''}")
    ids = cancel_active_jobs(job_manager)
    threading.Thread(target=_shutdown_worker, args=(job_manager, grace_sec, delay_sec),
                     name="engine-shutdown", daemon=True).start()
    return ids


def serve(app, host="127.0.0.1", port=8000, sockets=None, on_started=None, log_config=None,
          log_level="info"):
    """
    uvicorn szerver futtatása (blokkol), a /shutdown számára regisztrálva.
    on_started(server): a szerver indulása (a socket figyel) után egyszer meghívódik.
    """
    import uvicorn
    kwargs = {"host": host, "port": port, "log_level": log_level}
    if log_config is not False:
        kwargs["log_config"] = log_config if log_config is not None else uvicorn.config.LOGGING_CONFIG
    else:
        kwargs["log_config"] = None
    config = uvicorn.Config(app, **kwargs)
    server = uvicorn.Server(config)
    register_server(server)
    if on_started is not None:
        def _watch():
            while not server.started and not server.should_exit:
                time.sleep(0.05)
            if server.started:
                on_started(server)
        threading.Thread(target=_watch, name="engine-ready", daemon=True).start()
    server.run(sockets=sockets)
    return server


# ---------------------------------------------------------------- végpontok
def configured_model(cfg):
    """A beállított (még nem feltétlenül betöltött) modell rövid leírása – a program kijelzéséhez.
    A modell lustán, az első elemzéskor töltődik be; addig ebből látszik, melyik lesz."""
    if not isinstance(cfg, dict):
        return None
    kind = str(cfg.get("backend") or "ultralytics").lower()
    if kind == "onnx":
        return f"ONNX: {os.path.basename(str(cfg.get('onnx_path') or '?'))}"
    w = cfg.get("weights") or f"{cfg.get('weights_cuda') or 'yolov8x.pt'} (GPU) / {cfg.get('weights_cpu') or 'yolov8n.pt'} (CPU)"
    return f"{kind}: {w}"


def health_info(engine):
    model_state = getattr(engine, "model_state", {}) or {}
    return {
        "status": "shutting_down" if _state["shutting_down"] else "ok",
        "version": ENGINE_VERSION,
        "model_loaded": getattr(engine, "model", None) is not None,
        "model_loading": bool(model_state.get("loading")),
        "model_error": model_state.get("error"),
        "model_weights": getattr(engine, "model_weights", None),
        "device": getattr(engine, "device", None),
        "model_configured": configured_model(getattr(engine, "model_cfg", None)),
        "ffmpeg": app_paths.find_ffmpeg(),
        "data_dir": os.path.abspath(app_paths.data_dir()),
        "mode": "app" if app_paths.app_mode() else "dev",
        "jobs_active": len(active_jobs(getattr(engine, "job_manager", None))),
        "shutting_down": _state["shutting_down"],
        "pid": os.getpid(),
    }


async def _params(request):
    """Paraméterek a query-ből, form-ból vagy JSON törzsből (ami van)."""
    out = dict(request.query_params)
    ctype = (request.headers.get("content-type") or "").lower()
    try:
        if "application/json" in ctype:
            body = await request.body()
            if body:
                data = json.loads(body)
                if isinstance(data, dict):
                    out.update({k: v for k, v in data.items() if v is not None})
        elif "form" in ctype:
            form = await request.form()
            out.update({k: v for k, v in form.items() if isinstance(v, str)})
    except ValueError:
        pass
    return out


def install_endpoints(app, engine):
    from fastapi import Request
    from fastapi.responses import JSONResponse

    def _err(code, message):
        return JSONResponse(status_code=code, content={"status": "error", "message": message})

    @app.get("/health")
    async def health_endpoint():
        return health_info(engine)

    @app.get("/version")
    async def version_endpoint():
        return version_info()

    @app.post("/shutdown")
    async def shutdown_endpoint(request: Request):
        client = getattr(request, "client", None)
        host = getattr(client, "host", None)
        if not is_local_client(host):
            return _err(403, "A leállítás csak a helyi gépről kérhető.")
        if not token_ok(request.headers):
            return _err(403, "Hibás vagy hiányzó vezérlő token (X-TacAnaly-Token).")
        params = await _params(request)
        try:
            grace = float(params.get("grace_sec", 10.0))
        except (TypeError, ValueError):
            grace = 10.0
        ids = request_shutdown(getattr(engine, "job_manager", None), grace_sec=grace, reason="/shutdown")
        return {"status": "shutting_down", "already": ids is None, "cancelled_jobs": ids or [],
                "grace_sec": grace}

    @app.get("/models/status")
    async def models_status_endpoint():
        info = model_manager.status(engine.model_cfg)
        info.update({"status": "success", "model_loaded": engine.model is not None,
                     "model_weights": engine.model_weights, "device": engine.device,
                     "model_error": (engine.model_state or {}).get("error")})
        return info

    @app.post("/models/download")
    async def models_download_endpoint(request: Request):
        params = await _params(request)
        name = (params.get("name") or "").strip()

        def _job(progress, name):
            if not name:
                progress(0.0, "Eszköz felismerése…")
                from model_config import resolve_weights
                name = resolve_weights(engine.model_cfg, engine.detect_device())
            if model_manager.find_weights(name):
                path = model_manager.find_weights(name)
                return {"name": os.path.basename(name), "path": path, "downloaded": False,
                        "size": os.path.getsize(path)}
            return model_manager.download_weights(name, progress=progress)

        if name and not model_manager.find_weights(name) and not model_manager.official_url(name):
            return _err(400, f"Ismeretlen súlyfájl: {name} – csak a hivatalos súlyok tölthetők le "
                             f"({', '.join(model_manager.OFFICIAL_WEIGHTS)}); egyedi modellt másolj a "
                             f"models mappába.")
        job_id = engine.job_manager.submit("model_download", _job, name, params={"name": name or None})
        return {"status": "success", "job_id": job_id, "name": name or None}

    @app.post("/models/load")
    async def models_load_endpoint():
        def _job(progress):
            engine.ensure_model(progress=progress)
            return {"model_weights": engine.model_weights, "device": engine.device}
        job_id = engine.job_manager.submit("model_load", _job)
        return {"status": "success", "job_id": job_id}

    return app
