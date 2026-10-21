"""
virustotal.py
Optional VirusTotal domain reputation enrichment.

Only called on already-shortlisted candidates.  Degrades gracefully to
"N/A" if no API key is provided, the request times out, or any exception
occurs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_VT_URL = "https://www.virustotal.com/api/v3/domains/{domain}"
_TIMEOUT = 5  # seconds


def enrich_host(dest_host: str, api_key: str | None) -> dict:
    """
    Look up *dest_host* in the VirusTotal domain reputation API.

    Parameters
    ----------
    dest_host:
        The hostname or IP address to query.
    api_key:
        A valid VirusTotal API v3 key.  If ``None`` or empty, returns "N/A"
        immediately without making a network request.

    Returns
    -------
    dict
        Keys:
            ``vt_malicious_count``  — int or "N/A"
            ``vt_total_engines``    — int or "N/A"
            ``vt_error``            — str describing any failure, else ""
    """
    na_result = {"vt_malicious_count": "N/A", "vt_total_engines": "N/A", "vt_error": ""}

    if not api_key:
        na_result["vt_error"] = "No API key provided"
        return na_result

    # Strip any port or path component
    host = dest_host.split(":")[0].strip()
    if not host:
        na_result["vt_error"] = "Invalid host"
        return na_result

    try:
        import requests  # type: ignore

        url = _VT_URL.format(domain=host)
        headers = {"x-apikey": api_key}
        resp = requests.get(url, headers=headers, timeout=_TIMEOUT)

        if resp.status_code == 200:
            data = resp.json()
            stats = (
                data.get("data", {})
                .get("attributes", {})
                .get("last_analysis_stats", {})
            )
            malicious = stats.get("malicious", 0)
            total = sum(stats.values()) if stats else 0
            return {
                "vt_malicious_count": malicious,
                "vt_total_engines": total,
                "vt_error": "",
            }
        elif resp.status_code == 401:
            na_result["vt_error"] = "Invalid API key (401)"
            return na_result
        elif resp.status_code == 404:
            # Domain simply not in VT database — that's valid information
            return {
                "vt_malicious_count": 0,
                "vt_total_engines": 0,
                "vt_error": "Domain not in VT database",
            }
        else:
            na_result["vt_error"] = f"HTTP {resp.status_code}"
            return na_result

    except Exception as exc:  # noqa: BLE001
        logger.warning("VirusTotal lookup failed for %s: %s", host, exc)
        na_result["vt_error"] = f"Request failed: {type(exc).__name__}"
        return na_result


if __name__ == "__main__":
    # Quick smoke test — no real API key needed
    print("No key test:", enrich_host("example.com", None))
    print("Invalid key test:", enrich_host("example.com", "INVALID_KEY_TEST"))
