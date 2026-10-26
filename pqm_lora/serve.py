"""``python -m pqm_lora.serve --port 7878``: the caption service on the pod.

PQM's LoRA extension declares this as its pod service (``provides.remote.service``):
the agent starts it after the install steps and waits for ``GET /health``. The
extension's ``tag`` and ``caption`` jobs then ask it about one image at a time,
so each model loads once per pod rather than once per image -- JoyCaption alone is
16 GB. Reached on ``127.0.0.1`` only.

    GET  /health                 {"ok": true, "version", "loaded": {"tagger", "joycaption"}}
    POST /tag      {"path", "min_score"?}              -> the tagger's report
    POST /caption  {"path", "prompt", "max_new_tokens"?} -> {"sentence", ...}

A model loads on its first request (the health check answers at once), and the
GPU work is one request at a time. A bad request is 400 with ``{"error"}``; a
model that fails is 500 with the same. Standard library, plus the two model
modules, which import torch only when first asked.
"""

import argparse
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, models

DEFAULT_PORT = 7878


class Models:
    """The two models, each made on first use by its factory. Tests pass fakes."""

    def __init__(self, tagger_factory=None, joycaption_factory=None):
        self._factories = {"tagger": tagger_factory or _real_tagger,
                           "joycaption": joycaption_factory or _real_joycaption}
        self._made = {}
        self.lock = threading.Lock()

    def get(self, name):
        if name not in self._made:
            self._made[name] = self._factories[name]()
        return self._made[name]

    def loaded(self):
        return {name: name in self._made for name in self._factories}


def _real_tagger():
    from .tagger import Tagger

    return Tagger()


def _real_joycaption():
    from .joycaption import JoyCaption

    return JoyCaption()


class BadRequest(ValueError):
    pass


def _image_path(body):
    path = body.get("path")
    if not isinstance(path, str) or not path:
        raise BadRequest("a request names its image by an absolute path")
    if not os.path.isabs(path):
        raise BadRequest("%r is not an absolute path" % path)
    if not os.path.isfile(path):
        raise BadRequest("there is no file at %s" % path)
    return path


def handle(models_, route, body):
    """One request's answer as ``(status, dict)``: the service without HTTP."""
    if route == "/tag":
        path = _image_path(body)
        try:
            min_score = float(body.get("min_score", models.REPORT_MIN_SCORE))
        except (TypeError, ValueError):
            raise BadRequest("min_score is not a number")
        with models_.lock:
            return 200, models_.get("tagger").tag(path, min_score)
    if route == "/caption":
        path = _image_path(body)
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise BadRequest("a caption request needs a prompt")
        try:
            tokens = int(body.get("max_new_tokens", models.MAX_NEW_TOKENS))
        except (TypeError, ValueError):
            raise BadRequest("max_new_tokens is not a whole number")
        if not 1 <= tokens <= 512:
            raise BadRequest("max_new_tokens must be 1 to 512")
        with models_.lock:
            return 200, models_.get("joycaption").caption(path, prompt, tokens)
    return 404, {"error": "no such route: %s" % route}


def make_handler(models_):
    class Handler(BaseHTTPRequestHandler):
        server_version = "pqm-lora/" + __version__

        def log_message(self, fmt, *args):
            sys.stdout.write("%s %s\n" % (self.address_string(), fmt % args))
            sys.stdout.flush()

        def _send(self, status, data):
            raw = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.split("?", 1)[0] == "/health":
                self._send(200, {"ok": True, "version": __version__,
                                 "loaded": models_.loaded()})
            else:
                self._send(404, {"error": "no such route: %s" % self.path})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise BadRequest("the body is not a JSON object")
                status, data = handle(models_, self.path.split("?", 1)[0], body)
            except (BadRequest, ValueError) as exc:
                status, data = 400, {"error": str(exc)}
            except Exception as exc:  # noqa: BLE001 - a model failure is an answer
                traceback.print_exc()
                status, data = 500, {"error": "%s: %s" % (type(exc).__name__, exc)}
            self._send(status, data)

    return Handler


def serve(port=DEFAULT_PORT, models_=None, ready=None):
    server = ThreadingHTTPServer(("127.0.0.1", int(port)), make_handler(models_ or Models()))
    print("pqm-lora %s caption service on 127.0.0.1:%d" % (__version__, server.server_port),
          flush=True)
    if ready is not None:
        ready(server)
    server.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    serve(args.port)


if __name__ == "__main__":
    main()
