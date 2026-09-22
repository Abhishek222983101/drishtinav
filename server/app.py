"""DrishtiNav server: dashboard, phone app (PWA) and the Python engine as a live API.

Local:   python -m drishtinav serve            -> http://localhost:5000  (dashboard)  /app/ (phone)
Render:  gunicorn server.app:app                -> the live engine API used by the Vercel site (render.yaml)

Static data (identical to the files the Vercel build ships, see server/datagen.py)
    GET /data/scenarios.json   /data/runs/<key>.json   /data/drives/<key>.json   /data/benchmark.json
    GET /data/osm/<file>       /models/<file>          /config.js
Live engine API (CORS enabled)
    GET  /api/health
    GET  /api/scenarios
    GET  /api/run?scenario=<id>&outage=60&seed=7&configs=full[,ai,ins]
    POST /api/navigate?profile=smartphone      body: CSV in the generic format (docs/EDGE_ENGINE.md)
"""
from __future__ import annotations

import os
import sys
import time
from functools import lru_cache
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import datagen  # noqa: E402

WEB = ROOT / "web"
OSM = ROOT / "data" / "osm"
STARTED = time.time()


@lru_cache(maxsize=32)
def _cached_run(key: str):
    for s in datagen.scenarios():
        for outage, k in s["runs"].items():
            if k == key:
                return datagen.run_result(s["id"], 0 if outage == "tunnel" else int(outage))
    abort(404)


@lru_cache(maxsize=8)
def _cached_drive(key: str):
    for s in datagen.scenarios():
        if s["drive"] == key:
            return datagen.drive_stream(s["id"])
    abort(404)


def create_app():
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024        # uploads to /api/navigate
    app.config["MAX_FORM_MEMORY_SIZE"] = 8 * 1024 * 1024

    @app.errorhandler(413)
    def too_large(_):
        resp = jsonify({"error": "upload too large (max 8 MB / 30 000 samples) - send a shorter drive"})
        resp.status_code = 413
        return resp

    @app.after_request
    def cors(resp):
        if request.path.startswith("/api/") or request.path.startswith("/data/"):
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

    # ------------------------------------------------------------ front-ends
    @app.route("/")
    def dashboard():
        return send_from_directory(WEB / "dashboard", "index.html")

    @app.route("/dashboard/<path:p>")
    def dashboard_files(p):
        return send_from_directory(WEB / "dashboard", p)

    @app.route("/app/")
    def mobile_index():
        return send_from_directory(WEB / "mobile", "index.html")

    @app.route("/app/<path:p>")
    def mobile_files(p):
        resp = send_from_directory(WEB / "mobile", p)
        if p == "sw.js":
            resp.headers["Service-Worker-Allowed"] = "/app/"
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.route("/config.js")
    def config_js():
        # same-origin engine when served by this server; the static build writes the Render URL
        engine = os.environ.get("DRISHTI_ENGINE_URL", "")
        return Response(f'window.DRISHTI = {{ data: "/data", engine: "{engine}" }};\n',
                        mimetype="application/javascript", headers={"Cache-Control": "no-cache"})

    @app.route("/vendor/<path:p>")
    def vendor(p):
        return send_from_directory(WEB / "vendor", p)

    @app.route("/favicon.ico")
    def favicon():
        return send_from_directory(WEB / "mobile", "icon.svg", mimetype="image/svg+xml")

    @app.route("/models/<path:p>")
    def models(p):
        return send_from_directory(ROOT / "models", p)

    @app.route("/results/<path:p>")
    def results(p):
        return send_from_directory(ROOT / "results", p)

    # ------------------------------------------------------------ static data
    @app.route("/data/osm/<path:p>")
    def osm(p):
        return send_from_directory(OSM, p)

    @app.route("/data/scenarios.json")
    def data_scenarios():
        return jsonify(datagen.scenarios())

    @app.route("/data/runs/<key>.json")
    def data_run(key):
        return jsonify(_cached_run(key))

    @app.route("/data/drives/<key>.json")
    def data_drive(key):
        return jsonify(_cached_drive(key))

    @app.route("/data/benchmark.json")
    def data_benchmark():
        return jsonify(datagen.benchmark())

    # ------------------------------------------------------------ live engine API
    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "engine": "drishtinav", "uptime_s": round(time.time() - STARTED, 1)})

    @app.route("/api/scenarios")
    def api_scenarios():
        return jsonify(datagen.scenarios())

    @app.route("/api/run")
    def api_run():
        sid = request.args.get("scenario", "sim:delhi_pragati_tunnel:smartphone")
        if sid not in datagen.scenario_ids():
            return jsonify({"error": f"unknown scenario {sid!r}", "scenarios": sorted(datagen.scenario_ids())}), 404
        try:
            outage = int(request.args.get("outage", 60))
            seed = int(request.args.get("seed", 7))
            configs = tuple(c for c in request.args.get("configs", "ins,ai,full").split(",") if c in datagen.ABLATIONS)
            if outage not in (30, 60, 120, 180) and not sid.startswith("sim:"):
                return jsonify({"error": "outage must be 30, 60, 120 or 180"}), 400
            t0 = time.time()
            res = datagen.run_result(sid, outage, seed, configs or ("full",))
            res["compute_s"] = round(time.time() - t0, 2)
            return jsonify(res)
        except Exception as ex:  # surface engine errors to the UI
            return jsonify({"error": f"{type(ex).__name__}: {ex}"}), 500

    @app.route("/api/navigate", methods=["POST", "OPTIONS"])
    def api_navigate():
        if request.method == "OPTIONS":
            return ("", 204)
        profile = request.args.get("profile", "smartphone")
        if profile not in ("smartphone", "mems_edge", "fog"):
            return jsonify({"error": "profile must be smartphone, mems_edge or fog"}), 400
        body = request.files["file"].read() if "file" in request.files else request.get_data()
        if not body or len(body) > 8_000_000:
            return jsonify({"error": "send a generic-format CSV (<= 8 MB) as the request body or a 'file' field"}), 400
        try:
            t0 = time.time()
            res = datagen.navigate_csv(body.decode("utf-8", "replace"), profile)
            res["compute_s"] = round(time.time() - t0, 2)
            return jsonify(res)
        except Exception as ex:
            return jsonify({"error": f"{type(ex).__name__}: {ex}"}), 400

    return app


app = create_app()          # for `gunicorn server.app:app`

if __name__ == "__main__":
    print("DrishtiNav server on http://localhost:5000  (phone app: /app/)")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
