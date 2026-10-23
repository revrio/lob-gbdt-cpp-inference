import asyncio
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import websockets

STREAM_URL = "wss://stream.binance.com:9443/ws/btcusdt@depth10@100ms"
OUTPUT_DIR = Path("data/raw")
CHUNK_SIZE = 10_000  # ~16.6 minutes of 100ms ticks per Parquet file

buffer = []
session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
chunk_idx = 0
keep_running = True

def flush_buffer():
    """Writes buffered rows to a compressed Parquet file and clears RAM."""
    global buffer, chunk_idx
    if not buffer:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(buffer)
    filename = OUTPUT_DIR / f"btcusdt_lob_{session_id}_part{chunk_idx:04d}.parquet"
    df.to_parquet(filename, engine="pyarrow", compression="snappy", index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved {len(df)} rows -> {filename}")
    buffer.clear()
    chunk_idx += 1

def handle_exit(sig, frame):
    """Catches Ctrl+C to flush remaining rows before exiting."""
    global keep_running
    print("\nShutting down gracefully... flushing remaining buffer to disk.")
    keep_running = False

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

async def record_lob():
    global keep_running
    reconnect_delay = 1

    while keep_running:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to {STREAM_URL}...")
            # ping_interval=None disables client-initiated ping timeouts while still
            # automatically answering Binance's server-initiated pings!
            async with websockets.connect(STREAM_URL, ping_interval=None) as ws:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected! Recording 100ms LOB snapshots (Press Ctrl+C to stop)...")
                reconnect_delay = 1

                while keep_running:
                    # Since data arrives every 100ms, 10s of silence means the connection died
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    ts_ns = time.time_ns()
                    data = json.loads(msg)

                    bids = data.get("bids", [])
                    asks = data.get("asks", [])
                    if len(bids) < 10 or len(asks) < 10:
                        continue

                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])

                    if best_bid >= best_ask or best_bid <= 0:
                        continue

                    row = {
                        "timestamp_ms": ts_ns // 1_000_000,
                        "last_update_id": data.get("lastUpdateId", 0),
                    }

                    for i in range(10):
                        row[f"bid_px_{i+1}"] = float(bids[i][0])
                        row[f"bid_sz_{i+1}"] = float(bids[i][1])
                        row[f"ask_px_{i+1}"] = float(asks[i][0])
                        row[f"ask_sz_{i+1}"] = float(asks[i][1])

                    buffer.append(row)

                    if len(buffer) >= CHUNK_SIZE:
                        flush_buffer()

        except Exception as e:
            if not keep_running:
                break
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Disconnected ({e}). Reconnecting in {reconnect_delay}s...")
            flush_buffer()
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)

    flush_buffer()
    print("Recording complete.")

if __name__ == "__main__":
    asyncio.run(record_lob())
