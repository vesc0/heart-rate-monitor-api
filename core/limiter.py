from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _client_ip(request: Request) -> str:
    # Behind App Service the peer is the proxy, which appends the caller as the
    # last X-Forwarded-For hop in ip:port form.
    hop = request.headers.get("x-forwarded-for", "").split(",")[-1].strip()
    if hop.startswith("[") or hop.count(":") == 1:
        return hop.rsplit(":", 1)[0]
    return hop or get_remote_address(request)


limiter = Limiter(key_func=_client_ip)
