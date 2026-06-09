# EP16 OOM boundary

No OOM occurred in the requested `1A16F/EP16` matrix.

Tested range:

- `batch`: `2,4,8,16,32,64,128,256`
- `seq`: `128,256,512,1024`
- `mode`: `serial`, `decode-dbo`, `decode-dbo-crosslayer`
- `tokens`: `20`
- `micro-batches`: `2`

Largest verified non-OOM point:

| Mode | Batch | Seq | Status | Host1 HBM peak | Host2 HBM peak |
|---|---:|---:|---|---:|---:|
| serial | 256 | 1024 | OK | 88% | 72% |
| decode-dbo | 256 | 1024 | OK | 88% | 72% |
| decode-dbo-crosslayer | 256 | 1024 | OK | 88% | 72% |

Since no OOM was observed, the true boundary and first-OOM side are not determined. Host1 Attention reached higher HBM pressure than Host2 FFN EP ranks in the largest tested rows, so Attention is closer to the tested memory ceiling, but larger configs are required to prove whether A or F OOMs first.
