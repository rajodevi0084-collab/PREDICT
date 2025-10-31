# Labels Pipeline Data Contract (v1.0.0)

## Input Minute Bars
- **Format**: Parquet files with schema `[symbol:str, ts:datetime64[ns, Asia/Kolkata], open:float64, high:float64, low:float64, close:float64, volume:float64]`.
- **Constraints**:
  - `ts` must be timezone-aware (`Asia/Kolkata`).
  - `(symbol, ts)` pairs must be unique.
  - No gaps larger than twice the bar interval (2 minutes) unless an explicit gap flag column is present.

## Events Parquet Schema
Columns and types produced by `make_events`:

| Column    | Type                                      | Description                                     |
|-----------|-------------------------------------------|-------------------------------------------------|
| event_id  | int64                                     | Sequential event identifier.                    |
| symbol    | string                                    | Equity ticker.                                  |
| t_event   | datetime64[ns, Asia/Kolkata]              | Event start timestamp.                          |
| t_end     | datetime64[ns, Asia/Kolkata]              | Time-barrier horizon.                           |
| pt_px     | float64                                   | Profit-take price level.                        |
| sl_px     | float64                                   | Stop-loss price level.                          |
| t_touch   | datetime64[ns, Asia/Kolkata] or NaT       | Timestamp of first barrier touch or horizon.    |
| label     | int8 (values −1, 0, 1)                    | Barrier outcome.                                |
| ret       | float64                                   | Realised log-return at touch time.              |
| ambiguous | bool                                      | True if min-move filter marked event ambiguous. |

## Labels Parquet Schema
Columns and types produced by `labels_from_events`:

| Column    | Type                                      | Description                                     |
|-----------|-------------------------------------------|-------------------------------------------------|
| event_id  | int64                                     | Carries over from events parquet.               |
| symbol    | string                                    | Equity ticker.                                  |
| t_event   | datetime64[ns, Asia/Kolkata]              | Event timestamp.                                |
| t_end     | datetime64[ns, Asia/Kolkata]              | Time-barrier horizon.                           |
| pt_px     | float64                                   | Profit-take price level.                        |
| sl_px     | float64                                   | Stop-loss price level.                          |
| t_touch   | datetime64[ns, Asia/Kolkata] or NaT       | Touch/horizon time.                             |
| label     | int8 (values −1, 0, 1)                    | Final classification label.                     |
| ret       | float64                                   | Realised log-return.                            |
| ambiguous | bool                                      | Ambiguity flag retained for auditing.           |

Any consumer must tolerate the absence of ambiguous rows when `drop_ambiguous=true` (rows removed entirely).

## Splits JSON Structure
A JSON list where each element is:
```
{
  "fold": int,                   # fold number starting at 0
  "train_idx": [int, ...],       # indices referencing the labels parquet rows
  "val_idx": [int, ...],         # validation indices
  "val_window": ["ISO8601 start", "ISO8601 end"]
}
```

All indices refer to the zero-based ordering of the labels parquet. No train/validation overlap is permitted.

---
Any breaking change to these schemas requires bumping the version string above and communicating to downstream consumers.
