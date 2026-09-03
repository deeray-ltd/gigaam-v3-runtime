# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Yuriy Krasilnikov
# Copyright (C) 2026 Deeray Ltd.
#
# This file is part of GigaAM v3 Runtime, free software distributed under
# the terms of the GNU Lesser General Public License, either version 3 of the
# License, or (at your option) any later version. See COPYING.LESSER and COPYING
# for the full terms. There is NO WARRANTY, to the extent permitted by law.

"""ASR service (asr-serve) client for HTTP /v1/transcribe and WS /v1/stream.
Uses only the standard library, with no dependencies.

Example:
    from asr_client import AsrClient
    cli = AsrClient("127.0.0.1", 8080)
    print(cli.transcribe_file("audio.wav", model="ctc", words=True)["text"])
    for ev in cli.stream(pcm_chunks(), rate=16000, fmt="pcm16", emit="words"):
        if ev["type"] == "words":
            ...
"""
import base64, hashlib, http.client, json, os, socket, struct, threading

__all__ = ["AsrClient", "AsrError", "interleave_pcm16", "apply_turns"]


class AsrError(Exception):
    pass


def interleave_pcm16(channels):
    """Build N-interleaved stereo/N-channel pcm16 from mono pcm16 byte strings.
    Pad channels with silence to the longest length."""
    n = max((len(c) for c in channels), default=0) // 2
    N = len(channels)
    out = bytearray(n * 2 * N)
    for ci, c in enumerate(channels):
        for i in range(len(c) // 2):
            out[(i * N + ci) * 2:(i * N + ci) * 2 + 2] = c[i * 2:i * 2 + 2]
    return bytes(out)


def apply_turns(dialog, event):
    """Apply a turn event to a dialogue line in place: dialog[revise_from:] = turns."""
    if event.get("type") == "turns":
        dialog[event["revise_from"]:] = event["turns"]
    return dialog


class AsrClient:
    def __init__(self, host="127.0.0.1", port=8080, timeout=120):
        self.host, self.port, self.timeout = host, int(port), timeout

    # ---------- HTTP: batch transcription ----------
    def transcribe(self, data, *, model="ctc", words=False, segments=False,
                   turns=False, channels=None, ext=None, turn_gap=None, timeout=None):
        """data is audio bytes in any supported format. Returns the parsed JSON response."""
        q = [f"model={model}"]
        if words: q.append("words=1")
        if segments: q.append("segments=1")
        if turns: q.append("turns=1")
        if channels: q.append(f"channels={channels}")
        if ext: q.append(f"ext={ext}")
        if turn_gap is not None: q.append(f"turn_gap={turn_gap}")
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)
        try:
            conn.request("POST", "/v1/transcribe?" + "&".join(q), body=data,
                         headers={"Content-Type": "application/octet-stream"})
            r = conn.getresponse()
            body = r.read()
            if r.status != 200:
                raise AsrError(f"HTTP {r.status}: {body[:200].decode('utf-8','replace')}")
            return json.loads(body)
        finally:
            conn.close()

    def transcribe_file(self, path, **kw):
        kw.setdefault("ext", os.path.splitext(path)[1].lstrip(".").lower() or None)
        with open(path, "rb") as f:
            return self.transcribe(f.read(), **kw)

    def health(self):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            conn.request("GET", "/health"); r = conn.getresponse(); return json.loads(r.read())
        finally:
            conn.close()

    # ---------- WS: streaming transcription ----------
    def stream(self, chunks, *, rate, fmt="pcm16", model="ctc", channels=1, emit=None,
               endpoint=None, dedup=None, turn_gap=None, horizon=None, lock=False):
        """chunks is an iterable of byte strings. For channels>1 it contains N-interleaved samples
        (c0 c1 … c(N-1) c0 …); use `interleave_pcm16([ch0, ch1, …])` to build them.
        Generates JSON events. `emit=turns` is the default and yields
        {type:turns, revise_from, frontier, turns:[{channel,k,start,end,text,stable,final}]};
        `emit=words` yields per-channel words {type:words|stable|final, channel?}."""
        q = [f"rate={rate}", f"fmt={fmt}", f"model={model}", f"channels={channels}"]
        if emit is not None: q.append(f"emit={emit}")
        if endpoint is not None: q.append(f"endpoint={endpoint}")
        if dedup is not None: q.append(f"dedup={1 if dedup else 0}")
        if turn_gap is not None: q.append(f"turn_gap={turn_gap}")
        if horizon is not None: q.append(f"horizon={horizon}")
        if lock: q.append("lock=1")
        ws = _WebSocket(self.host, self.port, "/v1/stream?" + "&".join(q), self.timeout)
        ws.connect()
        done = threading.Event()

        def feeder():
            try:
                for ch in chunks:
                    if done.is_set():
                        break
                    ws.send_binary(ch)
                ws.send_text(json.dumps({"type": "end"}))
            except Exception:
                pass

        t = threading.Thread(target=feeder, daemon=True); t.start()
        try:
            while True:
                msg = ws.recv()
                if msg is None:  # close
                    break
                if isinstance(msg, str):
                    ev = json.loads(msg)
                    yield ev
        finally:
            done.set()
            ws.close()


# ---------- minimal RFC 6455 WebSocket client, standard library only ----------
class _WebSocket:
    def __init__(self, host, port, path, timeout):
        self.host, self.port, self.path, self.timeout = host, port, path, timeout
        self.sock = None
        self._buf = b""
        self._sendlock = threading.Lock()  # feeder, pong, and close must not race on sendall

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {self.path} HTTP/1.1\r\nHost: {self.host}:{self.port}\r\n"
               "Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(req.encode())
        # Read the response header through \r\n\r\n.
        resp = b""
        while b"\r\n\r\n" not in resp:
            d = self.sock.recv(4096)
            if not d:
                raise AsrError("WS: connection closed during handshake")
            resp += d
        head, _, rest = resp.partition(b"\r\n\r\n")
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise AsrError(f"WS handshake failed: {head[:120]!r}")
        accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        if accept.encode() not in head:
            raise AsrError("WS: invalid Sec-WebSocket-Accept")
        self._buf = rest

    def _mask(self, payload):
        m = os.urandom(4)
        return m + bytes(b ^ m[i % 4] for i, b in enumerate(payload))

    def _send(self, opcode, payload):
        b = bytearray([0x80 | opcode])  # FIN + opcode
        n = len(payload)
        if n < 126:
            b.append(0x80 | n)
        elif n < 65536:
            b.append(0x80 | 126); b += struct.pack(">H", n)
        else:
            b.append(0x80 | 127); b += struct.pack(">Q", n)
        b += self._mask(payload)
        with self._sendlock:
            self.sock.sendall(b)

    def send_binary(self, data):
        self._send(0x2, data if isinstance(data, (bytes, bytearray)) else bytes(data))

    def send_text(self, text):
        self._send(0x1, text.encode("utf-8"))

    def _read(self, n):
        while len(self._buf) < n:
            d = self.sock.recv(65536)
            if not d:
                return None
            self._buf += d
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def recv(self):
        """Returns str (text), bytes (binary), or None (closed).
        Reassembles fragmented messages (FIN=0 followed by continuation frames 0x0)."""
        frag_op, frag_buf = None, bytearray()
        while True:
            hdr = self._read(2)
            if hdr is None:
                return None
            b0, b1 = hdr[0], hdr[1]
            fin = b0 & 0x80
            opcode = b0 & 0x0F
            masked = b1 & 0x80
            ln = b1 & 0x7F
            if ln == 126:
                ext = self._read(2)
                if ext is None:
                    return None
                ln = struct.unpack(">H", ext)[0]
            elif ln == 127:
                ext = self._read(8)
                if ext is None:
                    return None
                ln = struct.unpack(">Q", ext)[0]
            mask = None
            if masked:  # servers do not mask, but accept it defensively
                mask = self._read(4)
                if mask is None:
                    return None
            payload = self._read(ln) if ln else b""
            if payload is None:
                return None
            if masked:
                payload = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
            # Control frames are never fragmented.
            if opcode == 0x8:      # close
                return None
            elif opcode == 0x9:    # ping -> pong
                self._send(0xA, payload); continue
            elif opcode == 0xA:    # pong
                continue
            # Data frames with fragment reassembly.
            if opcode in (0x1, 0x2):       # message start
                if not fin:
                    frag_op, frag_buf = opcode, bytearray(payload); continue
                data, op = payload, opcode
            elif opcode == 0x0:            # continuation
                if frag_op is None:
                    continue               # ignore unexpected continuation
                frag_buf += payload
                if not fin:
                    continue
                data, op = bytes(frag_buf), frag_op
                frag_op, frag_buf = None, bytearray()
            else:
                continue                   # unknown opcode
            return data.decode("utf-8", "replace") if op == 0x1 else bytes(data)

    def close(self):
        try:
            self._send(0x8, b"")
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
