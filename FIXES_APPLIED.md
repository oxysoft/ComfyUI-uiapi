# Critical Fixes Applied to ComfyUI-uiapi

## Summary
Fixed 6 critical issues in the server-side Python code for production stability and resource management.

---

## 1. Race Condition on Disconnect (Lines 249-288)

**Problem:** Manager was deleted immediately on disconnect, potentially causing errors for in-flight requests.

**Fix:**
- Added `_disconnected` flag to track disconnection state
- Cancel all pending requests gracefully before cleanup
- Set error responses for cancelled requests: `{"status": "error", "error": "Client disconnected"}`
- Schedule cleanup asynchronously with `_cleanup_manager()` method (0.1s delay)
- Clear both `_pending_requests` and `_buffered_requests`

**Behavior:** Existing requests complete gracefully with error status; manager cleanup is deferred.

---

## 2. Unbounded Dict Growth (Lines 454-471, 624-637)

**Problem:** `download_tasks` and `pending_downloads` dicts grew without cleanup, causing memory leaks.

**Fix:**
- Added `TASK_CLEANUP_AGE = 3600` (1 hour)
- Created `cleanup_old_tasks()` - removes completed tasks older than 1 hour
- Created `cleanup_old_pending_downloads()` - removes pending entries older than 1 hour
- Both called periodically in `uiapi_download_status()` endpoint (line 703-704)

**Behavior:** Old completed tasks auto-cleaned on status checks; memory usage bounded.

---

## 3. Unhandled Task Exception (Lines 640-653, 682-684)

**Problem:** `asyncio.create_task()` for downloads had no exception handling; silent failures.

**Fix:**
- Added `_handle_task_exception(task, task_id)` callback function
- Logs unhandled exceptions with full traceback
- Updates task status: `{"status": "error", "error": str(e), "completed": True}`
- Attached via `task.add_done_callback()` on line 684

**Behavior:** Download task failures are logged and status updated correctly.

---

## 4. Infinite Timeout (Lines 87, 990, 1218, 1237, 1310)

**Problem:** `INF_TIMEOUT = 99999999999` caused requests to hang indefinitely.

**Fix:**
- Replaced with `DEFAULT_TIMEOUT = 300.0` (5 minutes)
- Applied to all user interaction points:
  - Model URL lookup from webui
  - Workflow dialog responses
  - Model URL batch requests
  - Single model URL requests

**Behavior:** Requests timeout after 5 minutes with proper error responses.

---

## 5. Blocking File I/O (Lines 89-113)

**Problem:** Synchronous `open()` calls blocked the event loop.

**Fix:**
- Added `import aiofiles` (line 16)
- Converted `load_stored_urlmap()` to async with `aiofiles.open()`
- Converted `save_model_urls()` to async with `aiofiles.open()`
- Updated all call sites (11 locations) to use `await`

**Behavior:** File I/O no longer blocks async event loop; improved concurrency.

---

## 6. File Handle Leak (model_defs.py Line 320)

**Problem:** `open(path).read()` leaked file handles.

**Fix:**
```python
# Before:
return open(path).read().strip()

# After:
with open(path) as f:
    return f.read().strip()
```

**Behavior:** File handles properly closed after reading CivitAI tokens.

---

## Dependencies Added
- `aiofiles` - for async file I/O operations

Install with:
```bash
pip install aiofiles
```

---

## Testing Recommendations

1. **Disconnect handling:** Connect/disconnect clients rapidly; verify no crashes
2. **Memory cleanup:** Monitor memory usage over 2+ hours with active downloads
3. **Task exceptions:** Trigger download failures; verify error status updates
4. **Timeout handling:** Test with slow/unresponsive webui; verify 5min timeout
5. **File I/O:** Monitor event loop responsiveness under heavy model URL saves
6. **File handles:** Check `lsof` output for leaked handles after token reads

---

## Minimal Changes Principle

All fixes preserve existing behavior:
- No API changes
- No data format changes
- Error responses added where none existed
- Timeouts shortened from infinite to reasonable (5min)
- Async conversions maintain same logic flow
