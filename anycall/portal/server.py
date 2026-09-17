"""AnyCall Portal Server CLI Entrypoint."""
from __future__ import annotations

import argparse
import sys
import uvicorn

from anycall.portal.api import create_app


def main():
    parser = argparse.ArgumentParser(description="AnyCall Field Control Station Portal")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--db", type=str, default="anycall.db", help="SQLite database path (default: anycall.db)")
    parser.add_argument("--backbone", type=str, default="birdnet", help="Active backbone (birdnet, perch, panns, mock)")
    parser.add_argument("--threshold", type=float, default=0.60, help="Rejection threshold theta (default: 0.60)")
    parser.add_argument("--listen-mic", action="store_true", help="Start continuous live microphone listening on launch")
    parser.add_argument("--mic-device", type=int, default=None, help="Microphone device index (default: system default)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for local development")

    args = parser.parse_args()

    print("=" * 72)
    print("           AnyCall Wildlife Field Station & Control Portal           ")
    print("=" * 72)
    print(f"  • Host / Port       : http://{args.host}:{args.port}")
    print(f"  • Database          : {args.db}")
    print(f"  • Active Backbone   : {args.backbone}")
    print(f"  • Default Threshold : {args.threshold:.2f}")
    print(f"  • Continuous Mic    : {'ENABLED (Listening)' if args.listen_mic else 'OFF (Manual)'}")
    if args.mic_device is not None:
        print(f"  • Mic Device Index  : {args.mic_device}")
    print("=" * 72)

    app = create_app(
        db_path=args.db,
        backbone_name=args.backbone,
        default_threshold=args.threshold,
    )

    if args.listen_mic:
        try:
            app.state.mic_listener.start(device=args.mic_device)
            print("[Portal] Continuous microphone listener actively monitoring audio input.")
        except Exception as e:
            print(f"[Portal] Warning: Failed to auto-start microphone listener: {e}")

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
