#!/usr/bin/env python3
"""Transparent logging proxy in front of an OpenAI-compatible server (vLLM).

The agent talks to the proxy; the proxy forwards everything to vLLM unchanged and, for each
/v1/chat/completions or /v1/completions call, appends one JSON line with timestamps
(CLOCK_MONOTONIC, same clock as sampler.py) and token usage, including cached prompt tokens
(vLLM must run with --enable-prompt-tokens-details).

For streaming requests the proxy sets stream_options.include_usage=true so usage is reported.

Usage:
  python llm_proxy.py --upstream http://127.0.0.1:8000 --port 8001 --log calls.jsonl
"""
import argparse
import json
import time

from aiohttp import ClientSession, ClientTimeout, web

LOGGED_PATHS = ("/v1/chat/completions", "/v1/completions")
HOP_REQ = {"host", "content-length", "transfer-encoding", "accept-encoding", "connection"}
HOP_RESP = {"content-length", "transfer-encoding", "content-encoding", "connection"}


def sse_usage(line):
    line = line.strip()
    if not line.startswith(b"data:"):
        return None
    data = line[5:].strip()
    if data == b"[DONE]":
        return None
    try:
        return json.loads(data).get("usage")
    except (ValueError, AttributeError):
        return None


def make_app(upstream, log_path, session_header):
    app = web.Application(client_max_size=256 * 1024 * 1024)
    log = open(log_path, "a", buffering=1)

    async def on_startup(app):
        app["session"] = ClientSession(timeout=ClientTimeout(total=None))

    async def on_cleanup(app):
        await app["session"].close()
        log.close()

    async def handle(request):
        body = await request.read()
        logged = request.method == "POST" and request.path in LOGGED_PATHS
        payload, stream = None, False
        if logged:
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            if isinstance(payload, dict) and payload.get("stream"):
                stream = True
                opts = payload.get("stream_options") or {}
                opts["include_usage"] = True
                payload["stream_options"] = opts
                body = json.dumps(payload).encode()

        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_REQ}
        t0m, t0w = time.monotonic(), time.time()
        t_first = None
        usage = None
        status = "aborted"  # kept if the client disconnects or the upstream fails mid-request
        try:
            async with app["session"].request(request.method, upstream + str(request.rel_url),
                                              headers=headers, data=body) as up:
                resp_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP_RESP}
                if stream and up.status == 200:
                    resp = web.StreamResponse(status=up.status, headers=resp_headers)
                    await resp.prepare(request)
                    buf = b""
                    async for chunk in up.content.iter_any():
                        if t_first is None:
                            t_first = time.monotonic()
                        await resp.write(chunk)
                        buf += chunk
                        *lines, buf = buf.split(b"\n")
                        for ln in lines:
                            usage = sse_usage(ln) or usage
                    await resp.write_eof()
                else:
                    data = await up.read()
                    resp = web.Response(status=up.status, body=data, headers=resp_headers)
                    if logged and up.status == 200:
                        try:
                            usage = json.loads(data).get("usage")
                        except ValueError:
                            pass
                status = up.status
        finally:
            # Always log, so a request cut off by an agent timeout still marks the GPU as busy.
            t1m, t1w = time.monotonic(), time.time()
            if logged:
                usage = usage or {}
                details = usage.get("prompt_tokens_details") or {}
                p = payload if isinstance(payload, dict) else {}
                log.write(json.dumps({
                    "t_start_mono": t0m, "t_end_mono": t1m, "t_first_byte_mono": t_first,
                    "t_start_wall": t0w, "t_end_wall": t1w,
                    "session_id": request.headers.get(session_header),
                    "path": request.path, "status": status, "stream": stream,
                    "model": p.get("model"), "n_messages": len(p.get("messages") or []),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cached_tokens": details.get("cached_tokens"),
                }) + "\n")
        return resp

    app.router.add_route("*", "/{tail:.*}", handle)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--log", required=True)
    ap.add_argument("--session-header", default="X-Session-Id",
                    help="request header carrying the agent's trajectory/session id")
    args = ap.parse_args()
    # On SIGTERM, cancel in-flight requests after 2 s: they are logged as "aborted" and the upstream
    # connection is closed, so vLLM stops generating instead of running into the next trial.
    web.run_app(make_app(args.upstream.rstrip("/"), args.log, args.session_header),
                host=args.host, port=args.port, shutdown_timeout=2)


if __name__ == "__main__":
    main()
