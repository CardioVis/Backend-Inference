import json
import os
import sys
import types

import requests
from label_studio_sdk import LabelStudio
from label_studio_sdk.core.api_error import ApiError
from label_studio_sdk.types.token_refresh_response import TokenRefreshResponse

from ls_export.env_loader import load_repo_dotenv
from ls_export.logging_support import _agent_log, _log_api_error


def _patch_token_refresh_to_include_cf_access(ls: LabelStudio, base_url: str, cf: dict) -> None:
    """label_studio_sdk TokensClientExt.refresh() POSTs via httpx with only Content-Type, dropping
    custom headers. Behind Cloudflare Access, /api/token/refresh then returns HTML → JSONDecodeError.
    """
    _tc = ls._client_wrapper._tokens_client

    def _refresh_cf(self) -> TokenRefreshResponse:
        _url = f"{base_url.rstrip('/')}/api/token/refresh/"
        _h = {"Content-Type": "application/json", **cf}
        _r = requests.post(
            _url,
            json={"refresh": self._api_key},
            headers=_h,
            timeout=120,
        )
        _agent_log(
            "token_refresh_response",
            {
                "status_code": _r.status_code,
                "content_type": _r.headers.get("Content-Type", ""),
                "body_len": len(_r.text or ""),
                "body_preview": (_r.text or "")[:400].replace("\n", "\\n"),
            },
            "H-refresh-CF",
        )
        if _r.status_code != 200:
            raise ApiError(status_code=_r.status_code, body=_r.text[:2000])
        try:
            _data = _r.json()
        except json.JSONDecodeError:
            raise ApiError(
                status_code=_r.status_code,
                body=_r.text[:2000] or "empty body (often Cloudflare HTML without CF-Access headers on refresh)",
            ) from None
        try:
            return TokenRefreshResponse.model_validate(_data)
        except AttributeError:
            return TokenRefreshResponse.parse_obj(_data)

    _tc.refresh = types.MethodType(_refresh_cf, _tc)


def _ls_bearer_headers(ls: LabelStudio, cf: dict) -> dict:
    tok = getattr(ls._client_wrapper._tokens_client, "api_key", "") or ""
    h = {"Authorization": f"Bearer {tok}"}
    h.update(cf or {})
    return h


def _label_studio_url() -> str:
    u = (os.environ.get("LABEL_STUDIO_URL") or "https://labeling.cardiovis.com/").strip()
    return u if u.endswith("/") else u + "/"


def _label_studio_api_key() -> str:
    return (
        os.environ.get("LABEL_STUDIO_API_KEY")
        or os.environ.get("LABEL_STUDIO_TOKEN")
        or ""
    ).strip()


def _cf_env_value(raw: str | None) -> str:
    """Strip whitespace; allow mistaken 'CF-Access-Client-Id: <value>' pastes."""
    s = (raw or "").strip()
    if not s:
        return ""
    low = s.lower()
    for prefix in (
        "cf-access-client-id:",
        "cf-access-client-secret:",
    ):
        if low.startswith(prefix):
            return s[len(prefix) :].strip()
    return s


def _cf_access_headers():
    """Optional machine auth when Label Studio is behind Cloudflare Access."""
    cid = _cf_env_value(
        os.environ.get("CF_ACCESS_CLIENT_ID")
        or os.environ.get("CLOUDFLARE_ACCESS_CLIENT_ID")
    )
    csec = _cf_env_value(
        os.environ.get("CF_ACCESS_CLIENT_SECRET")
        or os.environ.get("CLOUDFLARE_ACCESS_CLIENT_SECRET")
    )
    if cid and csec:
        return {"CF-Access-Client-Id": cid, "CF-Access-Client-Secret": csec}
    return {}


def _connect_label_studio() -> tuple[LabelStudio, str, dict]:
    """Validate env, probe host, build SDK client, return (client, base_url_no_trailing_slash, cf_headers)."""
    if load_repo_dotenv():
        _agent_log("dotenv_loaded", {"path": "Backend-Inference/.env"}, "H-env")
    label_studio_url = _label_studio_url()
    api_key = _label_studio_api_key()
    if not api_key:
        print(
            "Set LABEL_STUDIO_API_KEY (or LABEL_STUDIO_TOKEN) to your Label Studio "
            "access token (Account → Access Token). Do not commit it.",
            file=sys.stderr,
        )
        sys.exit(1)

    cf_headers = _cf_access_headers()
    _agent_log(
        "startup_env",
        {
            "label_studio_url": label_studio_url,
            "api_key_len": len(api_key),
            "api_key_jwt_shape": api_key.count(".") == 2,
            "cf_headers_configured": bool(cf_headers),
        },
        "H2-H5",
    )

    base_no_slash = label_studio_url.rstrip("/")
    version_url = base_no_slash + "/api/version"
    probe_headers = {"Authorization": f"Token {api_key}"}
    probe_headers.update(cf_headers)
    resp = requests.get(
        version_url,
        headers=probe_headers,
        timeout=30,
        allow_redirects=True,
    )
    text = resp.text or ""
    final = resp.url or ""
    is_cf_gate = (
        "cloudflareaccess.com" in final
        or "cdn-cgi/access/login" in final
        or "Cloudflare Access" in text
    )
    ct = (resp.headers.get("Content-Type") or "").lower()
    looks_like_json = "application/json" in ct or text.lstrip().startswith("{")
    _agent_log(
        "cf_probe_result",
        {
            "status_code": resp.status_code,
            "final_url_host": final.split("//", 1)[-1].split("/", 1)[0] if final else "",
            "is_cf_gate": is_cf_gate,
            "looks_like_json": looks_like_json,
            "content_type": resp.headers.get("Content-Type", ""),
        },
        "H1",
    )
    if is_cf_gate and not looks_like_json:
        cf_set = bool(cf_headers)
        print(
            "This host is behind Cloudflare Access: /api/version returned the Access "
            "login HTML instead of JSON.\n\n"
            f"In this process, CF_ACCESS_* service token env vars look "
            f"{'set (both ID and secret non-empty)' if cf_set else 'unset or incomplete'}.\n"
            + (
                "\n"
                "Cloudflare is still returning the IdP login page. Common causes:\n"
                "  • Policy Action must be 'Service Auth', not 'Allow'.\n"
                "  • Confirm CF_ACCESS_CLIENT_ID matches the token in your Access policy.\n"
                "  • Docs: https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/\n\n"
                if cf_set
                else ""
            )
            + "Export CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET (see script comments).\n",
            file=sys.stderr,
        )
        sys.exit(1)

    ls_headers = dict(cf_headers) if cf_headers else None
    try:
        ls = LabelStudio(base_url=base_no_slash, api_key=api_key, headers=ls_headers)
    except ApiError as e:
        _log_api_error("label_studio_client_init_api_error", e, "H2")
        print(e, file=sys.stderr)
        sys.exit(1)

    _patch_token_refresh_to_include_cf_access(ls, base_no_slash, dict(cf_headers))

    try:
        ls.users.whoami()
    except ApiError as e:
        _log_api_error("whoami_api_error", e, "H2")
        if e.status_code == 401:
            print(
                "Label Studio returned 401 on /api/current-user/whoami.\n"
                "Personal Access Tokens (JWT-shaped) must use the modern SDK path: this "
                "script now uses LabelStudio(), which calls /api/token/refresh and sends "
                "Authorization: Bearer <access>.\n"
                "If you still see 401: create a new PAT, or use a Legacy token "
                "(Account settings) if your org enables it — legacy tokens use "
                "Authorization: Token … and also work with LabelStudio().\n",
                file=sys.stderr,
            )
        else:
            print(e, file=sys.stderr)
        sys.exit(1)

    _agent_log("whoami_ok", {}, "H2")
    return ls, base_no_slash, cf_headers


def _list_projects_print(ls: LabelStudio, base_no_slash: str, cf_headers: dict) -> None:
    """Print id and title for each project (tab-separated), using the REST API."""
    hdr = _ls_bearer_headers(ls, cf_headers)
    url = f"{base_no_slash.rstrip('/')}/api/projects/"
    page = 1
    page_size = 100
    while True:
        r = requests.get(
            url,
            headers=hdr,
            params={"page": page, "page_size": page_size},
            timeout=120,
            allow_redirects=True,
        )
        if r.status_code != 200:
            print(
                f"Failed to list projects: HTTP {r.status_code}\n{(r.text or '')[:800]}",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            body = r.json()
        except json.JSONDecodeError:
            print(
                "Projects response was not JSON (check auth / Cloudflare headers).",
                file=sys.stderr,
            )
            sys.exit(1)
        if isinstance(body, list):
            items = body
        else:
            items = body.get("results") or body.get("projects") or []
        if not items:
            break
        for p in items:
            pid = p.get("id")
            title = (p.get("title") or p.get("name") or "").strip()
            print(f"{pid}\t{title}")
        if len(items) < page_size:
            break
        page += 1
        if page > 10_000:
            break
