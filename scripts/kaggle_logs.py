"""Лог идущего запуска на Kaggle.

kaggle kernels logs -f режет строки на букве "х": поток декодируется как latin-1,
а байт 0x85 из UTF-8 кода этой буквы считается переводом строки. Здесь поток
читается байтами и декодируется как UTF-8.

    .venv/bin/python scripts/kaggle_logs.py andreygetmanov/dls-hw5-xtts-smaug logs/kaggle.txt

Поток не закрывается, пока запуск идёт, поэтому скрипт останавливают снаружи.
"""
import json
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

kernel, out_path = sys.argv[1], sys.argv[2]
owner, slug = kernel.split("/")[:2]

api = KaggleApi()
api.authenticate()

with api.build_kaggle_client() as kaggle:
    http = kaggle._http_client
    http._init_session()
    url = f"{http._endpoint}/v1/kernels/logs/stream/{owner}/{slug}"
    headers = dict(http._session.headers)
    headers["Accept"] = "text/event-stream, */*"
    headers.pop("Content-Type", None)
    response = http._session.get(url, stream=True, headers=headers, auth=http._session.auth)
    response.raise_for_status()

    with open(out_path, "w") as out:
        if not (response.headers.get("Content-Type") or "").lower().startswith("text/event-stream"):
            # запуск уже закончился: приходит весь лог одним json-массивом
            for event in json.loads(response.content.decode("utf-8")):
                out.write(f"{event.get('time', 0):9.0f}  {event.get('data', '')}".rstrip("\n") + "\n")
            sys.exit(0)
        for raw in response.iter_lines(delimiter=b"\n"):
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].lstrip()
            if payload == "END_OF_LOG":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                event = {"data": payload}
            out.write(f"{event.get('time', 0):9.0f}  {event.get('data', '')}".rstrip("\n") + "\n")
            out.flush()
