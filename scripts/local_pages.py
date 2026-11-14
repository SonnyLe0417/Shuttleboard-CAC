"""Pages this machine serves may use the TrackNet services, on any port.

The services listen on this machine only. The app's dev server does not have a
fixed port: Vite moves to 5174, 5175, ... when 5173 is taken, so a fixed list
of allowed origins breaks every upload the moment a second dev server runs.
"""

import re

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import PlainTextResponse

LOCAL_ORIGIN = re.compile(r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?")


class AllowLocalPages:
    """Answers CORS for pages served from this machine and leaves every other origin to the app.

    For a service whose own CORS rules live elsewhere: this only ever allows more.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        origin = headers.get("origin")
        if not origin or not LOCAL_ORIGIN.fullmatch(origin):
            return await self.app(scope, receive, send)
        if scope["method"] == "OPTIONS" and "access-control-request-method" in headers:
            preflight = PlainTextResponse("OK", headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": headers["access-control-request-method"],
                "Access-Control-Allow-Headers": headers.get("access-control-request-headers", ""),
                "Access-Control-Max-Age": "600",
                "Vary": "Origin",
            })
            return await preflight(scope, receive, send)

        async def send_with_cors(message):
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers["Access-Control-Allow-Origin"] = origin
                response_headers["Access-Control-Allow-Credentials"] = "true"
                response_headers.add_vary_header("Origin")
            await send(message)

        await self.app(scope, receive, send_with_cors)
