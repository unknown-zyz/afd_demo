# D2 cross-host control-plane smoke (PASS)

- Date: 2026-05-20 03:31-03:33
- Coord: Host1 (afd-npu-test, 192.168.0.125:50071)
- Attn workers: Host1 ranks 0,1 (CPU fallback, --no-init-dist)
- FFN workers: Host2 (afd-npu-test-h2, liteserver-910c-2-00001) ranks 2,3
- Bundle commit: 4a0a7e0 (with FFN role= fix)
- Result: all 4 registered + survived > 30s stale threshold
