# Observed tide API context

Source: [browser request/response capture](api_call.txt), captured 2026-09-27. This records one observed session; it is not an API specification and does not prove the service is currently available.

## Services observed

- Tide backend: `http://117.250.29.124:5008`
- Sunrise and sunset: `https://api.sunrise-sunset.org/json`
- All captured requests returned HTTP 200.

## Tide backend routes

| Route | Method | Observed request | Observed response |
| --- | --- | --- | --- |
| `/api/port/port-list` | GET | None | `{status, data}`; 74 port records with `port_id`, `port_name`, `port_country`, `created_at`, `latitude`, and `longitude`. |
| `/api/port/port-details` | POST | `{port_id}` | `{status, data}` with the selected port's identity and coordinates. |
| `/api/tide/predicted-tide-data` | POST | `{port_id, start_date, days}` | `{status, count, data}`; event rows have `port_id`, `date_time`, `tide_height_m`, `tide_type`, and `created_at`. The capture returned 4 events for 1 day and 116 for 30 days. |
| `/api/tide/predicted-one-minute-data` | POST | `{port_id, start_date, days}` | `{status, count, data}`; minute samples have `port_id`, `date_time`, `tide_height_m`, and `created_at`. The 1-day request returned 1,440 rows. |
| `/api/tide/moon-data` | POST | `{date}` | `{data}` with `id`, `date_time`, `moon_phase`, and `created_at`; the captured response contained a New Moon on 2026-10-10 and a Full Moon on 2026-10-26. |

The observed tide queries selected Diamond Harbour (`DIAMOND HARBOUR (Hugli R.)`, India), port ID `6fdb84c4-1459-4f5a-960e-a5c50acfdeba`, at latitude `22.12`, longitude `88.10`. The port list contained 74 records in total.

## Sunrise and sunset lookup

The browser also called the public sunrise-sunset endpoint with `lat=22.12`, `lng=88.1`, `date=2026-09-27`, and `formatted=0`. Its response gave sunrise `2026-09-27T05:26:36+05:30` and sunset `2026-09-27T17:30:43+05:30`. The same lookup occurred twice in the capture.

## Timestamp and implementation boundaries

- Tide response timestamps are serialized with a trailing `Z`; the sunrise service returned explicit `+05:30` timestamps.
- The one-minute request used `start_date=2026-09-27` and `days=1`, but returned timestamps from `2026-09-26T13:00:00Z` through `2026-09-27T12:59:00Z`. Confirm the backend's date-boundary and timezone rules before treating a requested date as a UTC or local calendar day.
- This capture demonstrates observed browser traffic only. The current repository's Python entry point is a placeholder, and the Python sources do not implement or call these HTTP routes.
