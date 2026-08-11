import asyncio
import threading
import time
import ray


ray.init()


# 1. 默认串行 Actor：不需要锁，天然安全
@ray.remote
class SafeSerialCounter:
    def __init__(self):
        self.count = 0

    def incr(self, n: int = 1) -> int:
        current = self.count
        time.sleep(0.01)
        self.count = current + n
        return self.count

    def get(self) -> int:
        return self.count


# 2. 开启 max_concurrency 的普通 Actor：如果不加锁，会有竞态问题
@ray.remote(max_concurrency=8)
class UnsafeConcurrentCounter:
    def __init__(self):
        self.count = 0

    def incr(self, n: int = 1) -> int:
        current = self.count
        time.sleep(0.01)
        self.count = current + n
        return self.count

    def get(self) -> int:
        return self.count


# 3. 开启 max_concurrency 并加锁的普通 Actor：线程安全
@ray.remote(max_concurrency=8)
class LockedConcurrentCounter:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()

    def incr(self, n: int = 1) -> int:
        with self.lock:
            current = self.count
            time.sleep(0.01)
            self.count = current + n
            return self.count

    def get(self) -> int:
        with self.lock:
            return self.count


# 4. Async Actor：适合 I/O 密集，用 asyncio.Lock 保护共享状态
@ray.remote
class AsyncIOCounter:
    def __init__(self):
        self.count = 0
        self.lock = asyncio.Lock()

    async def fetch_and_incr(self, url_id: int, n: int = 1) -> dict:
        await asyncio.sleep(0.1)
        async with self.lock:
            self.count += n
            current = self.count
        return {"url_id": url_id, "processed": current}

    async def get(self) -> int:
        async with self.lock:
            return self.count


if __name__ == "__main__":
    N = 50

    print("=== 1. 默认串行 Actor ===")
    c1 = SafeSerialCounter.remote()
    start = time.time()
    refs = [c1.incr.remote() for _ in range(N)]
    results = ray.get(refs)
    print(f"最终 count: {ray.get(c1.get.remote())} (期望 {N})")
    print(f"耗时: {time.time() - start:.2f}s (串行执行，所以慢但正确)\n")

    print("=== 2. 开启 max_concurrency 但不加锁 ===")
    c2 = UnsafeConcurrentCounter.remote()
    start = time.time()
    refs = [c2.incr.remote() for _ in range(N)]
    results = ray.get(refs)
    print(f"最终 count: {ray.get(c2.get.remote())} (期望 {N})")
    print(f"耗时: {time.time() - start:.2f}s (并发快，但 count 不正确！)\n")

    print("=== 3. 开启 max_concurrency + threading.Lock ===")
    c3 = LockedConcurrentCounter.remote()
    start = time.time()
    refs = [c3.incr.remote() for _ in range(N)]
    results = ray.get(refs)
    print(f"最终 count: {ray.get(c3.get.remote())} (期望 {N})")
    print(f"耗时: {time.time() - start:.2f}s (并发执行，加锁保证正确)\n")

    print("=== 4. Async Actor: I/O 密集场景 ===")
    c4 = AsyncIOCounter.remote()
    start = time.time()
    refs = [c4.fetch_and_incr.remote(i) for i in range(N)]
    results = ray.get(refs)
    print(f"最终 count: {ray.get(c4.get.remote())} (期望 {N})")
    print(f"耗时: {time.time() - start:.2f}s (I/O 等待时让出执行权，并发高效且正确)")

    ray.shutdown()
