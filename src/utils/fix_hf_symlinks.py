#!/usr/bin/env python3
"""
Repair a HuggingFace cache that was transported across an exFAT filesystem.

WHY THIS EXISTS
  HF stores real file content in  hub/<model>/blobs/<sha>
  and puts symlinks in           hub/<model>/snapshots/<rev>/<filename>

  exFAT cannot store symlinks. Crossing it converts each symlink into a
  1067-byte plain text "XSym" stub:

      XSym
      0052
      ef2d91555bf44dec41c0ba5e8979c22d
      ../../blobs/583327915e2ce21dbdd7c44a90a9dcb2b02559a1

  transformers then reads config.json, gets that text instead of JSON, and
  raises JSONDecodeError. The stub still names its target, so the mapping is
  fully recoverable locally -- no re-copy of the 20GB cache required.

  Also removes macOS AppleDouble "._filename" junk left by the exFAT round trip.

BYTE LAYOUT (verified)
  offset  0..4    b"XSym\n"
  offset  5..9    4-digit target length + b"\n"
  offset 10..42   32-char md5 + b"\n"
  offset 43..     target path, NUL-padded to 1067 bytes total

USAGE
  python3 fix_hf_symlinks.py                  # dry run, changes nothing
  python3 fix_hf_symlinks.py --apply          # repair
  python3 fix_hf_symlinks.py --cache /some/other/hf_cache

GATE
  "targets resolvable" must EQUAL "XSym stubs found" and errors must be 0.
  A TARGET MISSING means that blob never transferred -- the script refuses to
  apply rather than leave a half-repaired cache.
"""

import argparse
import os
import sys

DEFAULT_CACHE = "/data/user/paper/hf_cache"
XSYM_SIZE = 1067
TARGET_OFFSET = 43


def scan(cache):
    """Walk the cache; classify every file. Mutates nothing."""
    xsym, apple, errors = [], [], []

    for root, _dirs, files in os.walk(cache):
        for name in files:
            path = os.path.join(root, name)

            if name.startswith("._"):
                apple.append(path)
                continue

            try:
                if os.path.islink(path):
                    continue  # already a real symlink; nothing to do
                if os.path.getsize(path) != XSYM_SIZE:
                    continue
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError as exc:
                errors.append((path, f"read failed: {exc}"))
                continue

            if data[:4] != b"XSym":
                continue  # coincidentally 1067 bytes, but genuine content

            try:
                length = int(data[5:9])
                target = data[TARGET_OFFSET:TARGET_OFFSET + length]
                target = target.decode().rstrip("\x00").strip()
            except (ValueError, UnicodeDecodeError) as exc:
                errors.append((path, f"parse failed: {exc}"))
                continue

            if not target:
                errors.append((path, "empty target"))
                continue

            resolved = os.path.normpath(os.path.join(root, target))
            if not os.path.exists(resolved):
                errors.append((path, f"TARGET MISSING: {target}"))
                continue

            xsym.append((path, target))

    return xsym, apple, errors


def main():
    parser = argparse.ArgumentParser(
        description="Repair XSym symlink stubs in a HuggingFace cache."
    )
    parser.add_argument("--cache", default=DEFAULT_CACHE,
                        help=f"cache root (default: {DEFAULT_CACHE})")
    parser.add_argument("--apply", action="store_true",
                        help="actually perform the repair (default: dry run)")
    args = parser.parse_args()

    if not os.path.isdir(args.cache):
        print(f"ERROR: cache directory not found: {args.cache}")
        return 2

    print(f"Scanning {args.cache} ...\n")
    xsym, apple, errors = scan(args.cache)

    print(f"XSym stubs found:      {len(xsym) + sum(1 for _, e in errors if 'TARGET MISSING' in e)}")
    print(f"  targets resolvable:  {len(xsym)}")
    print(f"AppleDouble (._) junk: {len(apple)}")
    print(f"errors:                {len(errors)}")

    for path, err in errors[:20]:
        print(f"  ERR  {path}\n       {err}")
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more")

    if xsym:
        print("\nsample of what will be relinked:")
        for path, target in xsym[:5]:
            print(f"  {os.path.basename(path):<32} -> {target}")

    if not args.apply:
        print("\nDRY RUN - nothing changed. Re-run with --apply to fix.")
        return 0

    if errors:
        print("\nREFUSING TO APPLY: resolve the errors above first.")
        print("A TARGET MISSING means that blob never transferred; that model")
        print("needs re-copying from the source cache.")
        return 1

    if not xsym and not apple:
        print("\nNothing to do - cache is already clean.")
        return 0

    for path in apple:
        os.remove(path)

    relinked = 0
    for path, target in xsym:
        try:
            os.remove(path)
            os.symlink(target, path)
            relinked += 1
        except OSError as exc:
            print(f"  FAILED on {path}: {exc}")
            print(f"  stopped after {relinked} repairs - cache is partially fixed.")
            return 1

    print(f"\nAPPLIED: {relinked} symlinks restored, "
          f"{len(apple)} AppleDouble files removed.")
    print("\nVerify with:")
    print(f"  head -c 200 {args.cache}/hub/models--google--gemma-2b/snapshots/*/config.json")
    print("  (should print real JSON starting with '{')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
