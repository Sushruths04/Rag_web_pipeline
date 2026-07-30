"""Dev/production server entrypoint. Run from production/backend:
    python serve.py [--port 8017] [--mode process|thread]
"""
import argparse

import uvicorn

from app.main import create_app

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--mode", choices=("thread", "process"), default="process")
    args = ap.parse_args()
    uvicorn.run(create_app(mode=args.mode), host="127.0.0.1", port=args.port)
