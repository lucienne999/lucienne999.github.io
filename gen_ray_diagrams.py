import base64
import zlib
import urllib.request
import urllib.error


def gen_kroki_png(graph, output_path, scale=2):
    compressed = zlib.compress(graph.encode('utf-8'), 9)
    b64 = base64.urlsafe_b64encode(compressed).decode('ascii')
    url = f"https://kroki.io/mermaid/png/{b64}?scale={scale}"
    print(f"[kroki png] {output_path} ...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        with open(output_path, "wb") as f:
            f.write(r.read())
    print("  ok")


def gen_mermaid_ink_png(graph, output_path):
    b64 = base64.urlsafe_b64encode(graph.encode('utf-8')).decode('ascii')
    url = f"https://mermaid.ink/img/{b64}?type=png"
    print(f"[mermaid.ink png] {output_path} ...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        with open(output_path, "wb") as f:
            f.write(r.read())
    print("  ok")


graph1 = """sequenceDiagram
    participant WA as WorkerA
    participant AP as Actor
    WA->>AP: MyActor.remote() (random ID, non-detached)
    Note over AP: Lifecycle bound to WorkerA
    WA->>AP: call methods
    WA-->>AP: WorkerA exits, actor killed
    Note right of AP: WorkerB cannot access this actor
"""

graph2 = """sequenceDiagram
    participant WA as WorkerA
    participant WB as WorkerB
    participant WC as WorkerC
    participant GCS as GCS
    participant AP as Actor
    WA->>GCS: lookup writer_0
    GCS-->>WA: not found, create new
    WA->>AP: start detached actor
    WA->>GCS: register writer_0 to AP
    GCS-->>WA: return handle_A
    WB->>GCS: lookup writer_0 (get_if_exists)
    GCS-->>WB: exists, return handle_B
    Note over WA,AP: handle_A and handle_B point to same Actor
    WA-->>WA: WorkerA exits
    Note over AP: Actor survives (detached)
    WB->>AP: upsert.remote() ok
    WC->>GCS: lookup writer_0 (get_if_exists)
    GCS-->>WC: exists, return handle_C
    WC->>AP: upsert.remote() ok
    Note over AP: cleaned up when Ray job ends
"""


if __name__ == "__main__":
    base = "/Users/bytedance/codebase/lucienne999.github.io/public/images"
    for name, g in [("ray-actor-default.png", graph1), ("ray-actor-detached.png", graph2)]:
        out = f"{base}/{name}"
        try:
            gen_kroki_png(g, out)
        except Exception as e1:
            print(f"  kroki failed: {e1}, trying mermaid.ink...")
            try:
                gen_mermaid_ink_png(g, out)
            except Exception as e2:
                print(f"  mermaid.ink also failed: {e2}")
                raise
    print("Done!")
