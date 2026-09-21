"""RCE/upload-style honeypot entry points."""

from flask import Blueprint, jsonify, request

from . import capture_context, log_request

bp = Blueprint("rce", __name__)

# Simulated filesystem state (in-memory, resets on restart)
_fake_fs = {
    "/": ["home", "tmp", "var", "etc", "opt"],
    "/home": ["admin", "user"],
    "/home/admin": ["notes.txt", "config.yaml"],
    "/home/user": ["documents", "downloads"],
    "/tmp": [],
    "/var": ["log", "www"],
    "/var/log": ["auth.log", "syslog"],
    "/var/www": ["html"],
    "/etc": ["passwd", "shadow", "hostname"],
    "/opt": ["app"],
}
_fake_files = {
    "/home/admin/notes.txt": (
        "Internal memo: honeypot deployment scheduled for Q3.\n"
        "Decoy credentials: admin:honeypot123 (DO NOT USE IN PROD)"
    ),
    "/home/admin/config.yaml": (
        "database:\n  host: localhost\n  port: 5432\n  name: mirage_prod\n"
        "api_keys:\n  - key: fake_key_abc123\n    service: internal-monitoring\n"
    ),
    "/etc/passwd": (
        "root:x:0:0:root:/root:/bin/bash\n"
        "admin:x:1000:1000:admin:/home/admin:/bin/bash\n"
        "user:x:1001:1001:user:/home/user:/bin/bash\n"
    ),
    "/etc/shadow": (
        "root:$6$salt$hash:19000:0:99999:7:::\n"
        "admin:$6$salt$hash:19000:0:99999:7:::\n"
        "user:$6$salt$hash:19000:0:99999:7:::\n"
    ),
    "/var/log/auth.log": (
        "Sep  6 10:00:01 honeypot sshd[1234]: Accepted password for admin from "
        "192.168.1.1\n"
        "Sep  6 10:05:22 honeypot sshd[1235]: Failed password for root from 10.0.0.1\n"
    ),
    "/var/log/syslog": (
        "Sep  6 10:00:00 honeypot systemd[1]: Started honeypot service.\n"
        "Sep  6 10:00:01 honeypot kernel[1]: [    0.000000] Linux version 5.15.0\n"
    ),
}

_current_dir = "/home/admin"


def _resolve_path(path: str) -> str:
    global _current_dir
    if not path.startswith("/"):
        path = _current_dir + "/" + path
    parts = []
    for part in path.split("/"):
        if part == "" or part == ".":
            continue
        elif part == "..":
            if parts:
                parts.pop()
        else:
            parts.append(part)
    return "/" + "/".join(parts)


def _list_dir(path: str):
    resolved = _resolve_path(path)
    if resolved in _fake_fs:
        return _fake_fs[resolved]
    return []


def _read_file(path: str):
    resolved = _resolve_path(path)
    return _fake_files.get(resolved, None)


def _execute_command(cmd: str):
    global _current_dir
    parts = cmd.strip().split()
    if not parts:
        return ""
    cmd_name = parts[0]
    args = parts[1:]

    if cmd_name == "ls":
        target = args[0] if args else "."
        items = _list_dir(target)
        return "  ".join(items) if items else "(empty)"
    elif cmd_name == "cat":
        if not args:
            return "cat: missing file operand"
        content = _read_file(args[0])
        return content if content is not None else f"cat: {args[0]}: No such file or directory"
    elif cmd_name == "pwd":
        return _current_dir
    elif cmd_name == "cd":
        if not args:
            _current_dir = "/home/admin"
            return ""
        _current_dir = _resolve_path(args[0])
        if _current_dir not in _fake_fs:
            _current_dir = "/home/admin"
            return f"cd: {args[0]}: No such file or directory"
        return ""
    elif cmd_name == "whoami":
        return "admin"
    elif cmd_name == "id":
        return "uid=1000(admin) gid=1000(admin) groups=1000(admin)"
    elif cmd_name == "wget" or cmd_name == "curl":
        url = args[0] if args else ""
        return (
            f"Connecting to {url}... connected.\n"
            "HTTP request sent, awaiting response... 200 OK\n"
            "Length: unspecified [text/html]\n"
            "Saving to: 'index.html'\n"
            "\n"
            "index.html          [ <=> ] 1.23K  --.-KB/s    in 0.01s\n"
            "\n"
            "2024-09-06 10:00:00 (100 KB/s) - 'index.html' saved [1234]"
        )
    elif cmd_name == "ps":
        return (
            "  PID TTY          TIME CMD\n"
            "    1 ?        00:00:00 init\n"
            "  123 ?        00:00:00 bash\n"
            "  456 ?        00:00:00 python3\n"
            "  789 ?        00:00:00 ps"
        )
    elif cmd_name == "netstat":
        return (
            "Active Internet connections (servers and established)\n"
            "Proto Recv-Q Send-Q Local Address           Foreign Address         State\n"
            " tcp        0      0 0.0.0.0:22              0.0.0.0:*               LISTEN\n"
            " tcp        0      0 0.0.0.0:80              0.0.0.0:*               LISTEN\n"
            " tcp        0      0 127.0.0.1:5432          0.0.0.0:*               LISTEN"
        )
    else:
        return f"sh: {cmd_name}: command not found"


@bp.route("/upload", methods=["GET", "POST"])
def upload():
    """Simulated file upload endpoint — RCE/upload target."""
    source_ip = request.remote_addr or "unknown"
    ctx = capture_context()
    method = ctx["method"]

    log_request(source_ip, ctx, "rce")

    if method == "POST":
        file = request.files.get("file")
        if file:
            filename = file.filename
            content = file.read().decode("utf-8", errors="replace")
            return jsonify(
                {
                    "status": "uploaded",
                    "filename": filename,
                    "size": len(content),
                    "message": f"File '{filename}' uploaded successfully (simulated)",
                }
            ), 200
        return jsonify({"error": "No file provided"}), 400

    return jsonify(
        {
            "form": (
                "<form method='POST' action='/upload' enctype='multipart/form-data'>"
                "<input type='file' name='file'/><button type='submit'>Upload</button></form>"
            )
        }
    ), 200


@bp.route("/admin/diagnostics", methods=["GET", "POST"])
def diagnostics():
    """Simulated admin diagnostics panel — command execution style RCE target."""
    source_ip = request.remote_addr or "unknown"
    ctx = capture_context()
    method = ctx["method"]

    log_request(source_ip, ctx, "rce")

    if method == "POST":
        data = request.form
        cmd = data.get("command", "")
        output = _execute_command(cmd)
        return jsonify(
            {
                "command": cmd,
                "output": output,
                "cwd": _current_dir,
            }
        ), 200

    return jsonify(
        {
            "form": (
                "<form method='POST' action='/admin/diagnostics'>"
                "<input name='command' placeholder='Enter command (ls, cat, pwd, whoami, "
                "wget, ps, netstat...' size='60'/><button type='submit'>Execute</button></form>"
            )
        }
    ), 200
