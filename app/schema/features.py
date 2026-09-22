"""25-feature HTTP-layer schema implementation.

Feature list (25 features):
0.  method_get          - 1 if GET, 0 otherwise
1.  method_post         - 1 if POST, 0 otherwise
2.  method_other        - 1 if other method (PUT, DELETE, etc.), 0 otherwise
3.  path_depth          - Number of path segments (e.g., /a/b/c = 3)
4.  path_length         - Total path string length
5.  has_query           - 1 if query string present, 0 otherwise
6.  query_length        - Query string length
7.  param_count         - Number of query/form parameters
8.  param_name_entropy  - Shannon entropy of parameter names
9.  param_value_entropy - Shannon entropy of parameter values
10. payload_length      - Request body/payload length
11. special_char_density- Ratio of special chars to total chars in payload+query
12. sql_keyword_count   - Count of SQL keywords (select, union, insert, etc.)
13. rce_keyword_count   - Count of RCE/shell keywords (wget, curl, bash, etc.)
14. path_traversal_count- Count of path traversal sequences (../, ..\, etc.)
15. xss_keyword_count   - Count of XSS keywords (script, alert, onerror, etc.)
16. encoding_count      - Count of encoding patterns (%XX, \\xXX, \\uXXXX)
17. digit_ratio         - Ratio of digits to total chars in payload+query
18. upper_ratio         - Ratio of uppercase to total chars in payload+query
19. longest_param_len   - Length of longest parameter value
20. avg_param_len       - Average parameter value length
21. cookie_count        - Number of cookies
22. user_agent_entropy  - Shannon entropy of User-Agent string
23. header_count        - Number of HTTP headers
24. content_type_json   - 1 if Content-Type is application/json, 0 otherwise
"""

import math
import re
from dataclasses import dataclass
from urllib.parse import parse_qs

import numpy as np

FEATURE_NAMES = [
    "method_get",
    "method_post",
    "method_other",
    "path_depth",
    "path_length",
    "has_query",
    "query_length",
    "param_count",
    "param_name_entropy",
    "param_value_entropy",
    "payload_length",
    "special_char_density",
    "sql_keyword_count",
    "rce_keyword_count",
    "path_traversal_count",
    "xss_keyword_count",
    "encoding_count",
    "digit_ratio",
    "upper_ratio",
    "longest_param_len",
    "avg_param_len",
    "cookie_count",
    "user_agent_entropy",
    "header_count",
    "content_type_json",
]

SQL_KEYWORDS = {
    "select",
    "union",
    "insert",
    "update",
    "delete",
    "drop",
    "create",
    "alter",
    "exec",
    "execute",
    "declare",
    "cast",
    "convert",
    "char",
    "varchar",
    "nchar",
    "nvarchar",
    "having",
    "group by",
    "order by",
    "where",
    "or",
    "and",
    "like",
    "between",
    "in",
    "exists",
    "sleep",
    "benchmark",
    "waitfor",
    "delay",
    "information_schema",
    "sysobjects",
    "syscolumns",
    "systables",
    "version",
    "current_user",
    "user_name",
    "system_user",
    "session_user",
    "database",
    "load_file",
    "into outfile",
    "into dumpfile",
    "xp_cmdshell",
    "sp_executesql",
}

RCE_KEYWORDS = {
    "wget",
    "curl",
    "nc",
    "netcat",
    "ncat",
    "socat",
    "bash",
    "sh",
    "zsh",
    "fish",
    "python",
    "perl",
    "ruby",
    "php",
    "node",
    "java",
    "go",
    "rust",
    "chmod",
    "chown",
    "chgrp",
    "chroot",
    "su",
    "sudo",
    "whoami",
    "id",
    "groups",
    "ls",
    "cat",
    "cp",
    "mv",
    "rm",
    "mkdir",
    "rmdir",
    "touch",
    "find",
    "grep",
    "awk",
    "sed",
    "cut",
    "sort",
    "uniq",
    "head",
    "tail",
    "wc",
    "tee",
    "xargs",
    "ps",
    "top",
    "htop",
    "kill",
    "killall",
    "pkill",
    "nohup",
    "screen",
    "tmux",
    "ssh",
    "scp",
    "rsync",
    "ftp",
    "sftp",
    "tftp",
    "telnet",
    "rsh",
    "rexec",
    "cron",
    "at",
    "batch",
    "systemctl",
    "service",
    "init",
    "systemd",
    "iptables",
    "ufw",
    "firewall",
    "netstat",
    "ss",
    "lsof",
    "ifconfig",
    "ip",
    "route",
    "arp",
    "ping",
    "traceroute",
    "dig",
    "nslookup",
    "host",
    "whois",
    "base64",
    "base32",
    "base16",
    "xxd",
    "od",
    "hexdump",
    "strings",
    "eval",
    "exec",
    "system",
    "shell_exec",
    "passthru",
    "proc_open",
    "popen",
    "subprocess",
    "Runtime",
    "ProcessBuilder",
    "cmd.exe",
    "powershell",
    "cmd",
    "certutil",
    "bitsadmin",
    "mshta",
    "regsvr32",
    "rundll32",
    "wmic",
    "schtasks",
}

PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.\\",
    r"%2e%2e%2f",
    r"%2e%2e/",
    r"..%2f",
    r"..%5c",
    r"%252e%252e%252f",
    r"\.\.%00",
    r"%00\.\.",
    r"\.\.%c0%af",
]

XSS_KEYWORDS = {
    "script",
    "alert",
    "onerror",
    "onload",
    "onclick",
    "onmouseover",
    "onfocus",
    "onblur",
    "onchange",
    "onsubmit",
    "onkeydown",
    "onkeyup",
    "onkeypress",
    "onresize",
    "onscroll",
    "onselect",
    "onunload",
    "javascript:",
    "vbscript:",
    "data:",
    "expression(",
    "url(",
    "<img",
    "<svg",
    "<iframe",
    "<object",
    "<embed",
    "<applet",
    "document.cookie",
    "document.write",
    "document.location",
    "window.location",
    "eval(",
    "setTimeout(",
    "setInterval(",
    "function(",
    "constructor",
    "prototype",
    "__proto__",
}

ENCODING_PATTERNS = [
    r"%[0-9a-fA-F]{2}",  # URL encoding %XX
    r"\\x[0-9a-fA-F]{2}",  # Hex escape \xXX
    r"\\u[0-9a-fA-F]{4}",  # Unicode escape \uXXXX
    r"&#x[0-9a-fA-F]+;",  # HTML hex entity
    r"&#[0-9]+;",  # HTML decimal entity
]

SPECIAL_CHARS = set("!@#$%^&*()_+-=[]{}|;':\",./<>?`~\\")


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def _count_keywords(text: str, keywords: set) -> int:
    text_lower = text.lower()
    count = 0
    for kw in keywords:
        count += text_lower.count(kw.lower())
    return count


def _count_patterns(text: str, patterns: list[str]) -> int:
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


@dataclass
class FeatureExtractor:
    """Extracts 25 HTTP-layer features from a request."""

    def extract(
        self,
        method: str,
        path: str,
        query_string: str,
        headers: dict[str, str],
        body: str,
    ) -> np.ndarray:
        """Extract all 25 features from request components."""
        features = np.zeros(25, dtype=np.float32)

        # 0-2: HTTP method
        method_lower = method.upper()
        features[0] = 1.0 if method_lower == "GET" else 0.0
        features[1] = 1.0 if method_lower == "POST" else 0.0
        features[2] = 1.0 if method_lower not in ("GET", "POST") else 0.0

        # 3-4: Path features
        path_segments = [s for s in path.split("/") if s]
        features[3] = float(len(path_segments))
        features[4] = float(len(path))

        # 5-6: Query string
        features[5] = 1.0 if query_string else 0.0
        features[6] = float(len(query_string))

        # Parse parameters from query string and body
        params = self._parse_params(query_string, body, headers.get("content-type", ""))

        # 7: Parameter count
        features[7] = float(len(params))

        # 8-9: Parameter name/value entropy
        if params:
            param_names = [k for k, v in params]
            param_values = [v for k, v in params]
            features[8] = _shannon_entropy("".join(param_names))
            features[9] = _shannon_entropy("".join(param_values))
        else:
            features[8] = 0.0
            features[9] = 0.0

        # 10: Payload length
        features[10] = float(len(body))

        # 11: Special character density
        combined = query_string + body
        if combined:
            special_count = sum(1 for c in combined if c in SPECIAL_CHARS)
            features[11] = special_count / len(combined)
        else:
            features[11] = 0.0

        # 12: SQL keyword count
        features[12] = float(_count_keywords(combined, SQL_KEYWORDS))

        # 13: RCE keyword count
        features[13] = float(_count_keywords(combined, RCE_KEYWORDS))

        # 14: Path traversal count
        features[14] = float(_count_patterns(combined, PATH_TRAVERSAL_PATTERNS))

        # 15: XSS keyword count
        features[15] = float(_count_keywords(combined, XSS_KEYWORDS))

        # 16: Encoding count
        features[16] = float(_count_patterns(combined, ENCODING_PATTERNS))

        # 17-18: Digit/upper ratios
        if combined:
            total = len(combined)
            digit_count = sum(1 for c in combined if c.isdigit())
            upper_count = sum(1 for c in combined if c.isupper())
            features[17] = digit_count / total
            features[18] = upper_count / total
        else:
            features[17] = 0.0
            features[18] = 0.0

        # 19-20: Longest/average param value length
        if params:
            param_values = [v for k, v in params]
            features[19] = float(max(len(v) for v in param_values))
            features[20] = float(sum(len(v) for v in param_values) / len(param_values))
        else:
            features[19] = 0.0
            features[20] = 0.0

        # 21: Cookie count
        cookie_header = headers.get("cookie", "")
        if cookie_header:
            features[21] = float(len([c for c in cookie_header.split(";") if "=" in c]))
        else:
            features[21] = 0.0

        # 22: User-Agent entropy
        ua = headers.get("user-agent", "")
        features[22] = _shannon_entropy(ua)

        # 23: Header count
        features[23] = float(len(headers))

        # 24: Content-Type JSON
        content_type = headers.get("content-type", "").lower()
        features[24] = 1.0 if "application/json" in content_type else 0.0

        return features

    def _parse_params(self, query_string: str, body: str, content_type: str) -> list[tuple[str, str]]:
        """Parse parameters from query string and request body."""
        params = []

        # Query string parameters
        if query_string:
            parsed = parse_qs(query_string, keep_blank_values=True)
            for key, values in parsed.items():
                for v in values:
                    params.append((key, v))

        # Body parameters (form data or JSON)
        if body:
            ct = content_type.lower()
            if "application/x-www-form-urlencoded" in ct:
                parsed = parse_qs(body, keep_blank_values=True)
                for key, values in parsed.items():
                    for v in values:
                        params.append((key, v))
            elif "application/json" in ct:
                try:
                    import json

                    data = json.loads(body)
                    if isinstance(data, dict):
                        for key, value in data.items():
                            params.append((key, str(value)))
                except (json.JSONDecodeError, ValueError):
                    pass
            elif "multipart/form-data" in ct:
                # For multipart, we can't easily parse without the full request
                # This is a limitation - in practice, Flask parses this for us
                pass

        return params


def extract_features(
    method: str,
    path: str,
    query_string: str,
    headers: dict[str, str],
    body: str,
) -> np.ndarray:
    """Convenience function to extract features without creating an extractor instance."""
    extractor = FeatureExtractor()
    return extractor.extract(method, path, query_string, headers, body)
