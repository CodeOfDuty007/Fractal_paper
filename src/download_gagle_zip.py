#!/usr/bin/env python3
"""Resumable download of gagle.zip to the external SSD.

SUPERSEDES the hf_hub_download(local_dir=...)-based version. That version
stalled repeatedly and, worse, an external watchdog killing/relaunching it
turned out to make things worse rather than better: huggingface_hub keys
its ".incomplete" partial-download path on the file's ETag
(_local_folder.py: incomplete_path(etag) -> f"{hash}.{etag}.incomplete"),
re-fetched via a fresh metadata call on every process invocation. Killing
and relaunching produced a DIFFERENT etag each time (thirteen distinct
.incomplete files accumulated, from 0B to 1.7GB, all abandoned) -- so the
resume mechanism this script's guardrails were supposed to rely on
silently was not resuming across process restarts at all.

This version does its own resume against a FIXED local file path
(gagle.zip.part), driven purely by that file's own on-disk size -- not by
any server-reported ETag. It reissues an HTTP Range request from that
exact byte offset on every retry, so a kill-and-restart (by a human, a
watchdog, or a lid-close) can never orphan progress the way the old
approach did.

It also sets an explicit read-timeout on the streamed response. A prior
stall observed in this same session (this download, and separately in
build_p0_corpus.py's dataset streaming) hung with no exception ever
raised, which is why external stall-detection was needed at all.
`requests`' timeout parameter applies to gaps between received chunks on
a streamed response, not just the initial connection, so a stall here
raises requests.exceptions.ReadTimeout/ConnectionError -- a normal,
catchable exception the retry loop below already handles -- instead of
hanging silently.
"""
import os
import sys
import time
import zipfile

import requests
from huggingface_hub import HfApi, get_token

REPO_ID = "ibomohsin/gagle"
FILENAME = "gagle.zip"
DEST_DIR = "/Volumes/NikPhotoSSD/paper"
DEST_PATH = os.path.join(DEST_DIR, FILENAME)
PART_PATH = DEST_PATH + ".part"

URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{FILENAME}"

CONNECT_TIMEOUT = 30
READ_TIMEOUT = 60          # no bytes received for this long on an open request -> retry
MAX_RETRIES = 500
BACKOFF_BASE = 3
BACKOFF_CAP = 60
CHUNK_SIZE = 1024 * 1024
PROGRESS_EVERY = 200 * 1024 * 1024


def expected_size():
    info = HfApi().repo_info(REPO_ID, repo_type="dataset", files_metadata=True)
    for s in info.siblings:
        if s.rfilename == FILENAME:
            return s.size
    raise RuntimeError(f"{FILENAME} not found in repo listing")


def auth_headers():
    token = get_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def current_part_size():
    return os.path.getsize(PART_PATH) if os.path.exists(PART_PATH) else 0


def download_once(exp_size, headers):
    """One attempt: resume from current_part_size() to exp_size, or raise."""
    local_size = current_part_size()
    if local_size >= exp_size:
        return
    req_headers = dict(headers)
    req_headers["Range"] = f"bytes={local_size}-"
    mode = "ab" if local_size > 0 else "wb"

    with requests.get(URL, headers=req_headers, stream=True,
                       timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as resp:
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"unexpected status {resp.status_code}: {resp.text[:200]}")
        if local_size > 0 and resp.status_code == 200:
            # We asked for bytes=<local_size>- but got a full 200 instead of a
            # 206 partial response: the server ignored our Range header and is
            # about to send the WHOLE file from byte zero. Appending that to
            # an existing partial would silently corrupt it -- refuse instead.
            raise RuntimeError(
                f"server ignored Range request (status 200, expected 206) with "
                f"{local_size} bytes already on disk -- refusing to append and corrupt {PART_PATH}"
            )
        last_progress_print = local_size
        with open(PART_PATH, mode) as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                local_size += len(chunk)
                if local_size - last_progress_print >= PROGRESS_EVERY:
                    f.flush()
                    os.fsync(f.fileno())
                    print(f"[progress] {local_size}/{exp_size} bytes "
                          f"({100 * local_size / exp_size:.1f}%)")
                    last_progress_print = local_size
            f.flush()
            os.fsync(f.fileno())


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    exp_size = expected_size()
    print(f"[info] expected size: {exp_size} bytes")
    headers = auth_headers()
    if not headers:
        print("[warn] no HF token found (run `hf auth login`) -- proceeding unauthenticated, "
              "subject to lower rate limits.")

    if os.path.exists(DEST_PATH) and os.path.getsize(DEST_PATH) == exp_size:
        print(f"[skip] {DEST_PATH} already exists at the expected size; verifying only.")
    else:
        attempt = 0
        while current_part_size() < exp_size:
            attempt += 1
            if attempt > MAX_RETRIES:
                print("[abort] exceeded max retries")
                return 1
            try:
                print(f"[attempt {attempt}] resuming from {current_part_size()} bytes")
                download_once(exp_size, headers)
            except Exception as e:
                wait = min(BACKOFF_BASE * attempt, BACKOFF_CAP)
                print(f"[retry] attempt={attempt} {type(e).__name__}: {e} -- "
                      f"retrying in {wait}s (resume point: {current_part_size()} bytes, safe)")
                time.sleep(wait)

        os.rename(PART_PATH, DEST_PATH)
        print(f"[transferred] {DEST_PATH}")

    # Verify before trusting it.
    actual_size = os.path.getsize(DEST_PATH)
    if actual_size != exp_size:
        print(f"[FAIL] size mismatch: expected {exp_size}, got {actual_size}. Deleting and re-download needed.")
        os.remove(DEST_PATH)
        return 1

    print("[verify] size OK, running zipfile.testzip() (checks CRCs of every member, takes a while)...")
    with zipfile.ZipFile(DEST_PATH) as zf:
        bad_file = zf.testzip()
    if bad_file is not None:
        print(f"[FAIL] corrupt member in zip: {bad_file}. Deleting and re-download needed.")
        os.remove(DEST_PATH)
        return 1

    print(f"[done] downloaded and verified: {DEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
