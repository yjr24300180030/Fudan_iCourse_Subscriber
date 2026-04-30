"""iCourse Subscriber — main orchestration.

Runs a single check: login → detect new lectures → stream audio → transcribe
→ summarize → email. Designed to be triggered by GitHub Actions cron.
"""

import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from src import bucketer, config
from src.database import Database
from src.emailer import Emailer
from src.icourse import ICourseClient
from src.ocr import ocr_image_text
from src.ppt_dedup import compute_dhash, dedup_dhash, is_invalid_page
from src.ppt_fetcher import fetch_ppt_image
from src.summarizer import Summarizer
from src.transcriber import IncompleteAudioError, NoAudioStreamError, Transcriber
from src.webvpn import WebVPNSession


# ── Concurrency primitives ─────────────────────────────────────────────────
# 20 image-download workers (IO-bound, GitHub runner has bandwidth headroom)
# and 3 OCR workers (CPU-bound; RapidOCR's ONNX runtime releases the GIL on
# native calls so Python threads get real parallelism). Sized for the
# ubuntu-latest 4-vCPU runner shape: ASR uses ~2 cores during transcribe,
# 3 OCR workers fills the rest plus a touch of oversubscription which the
# scheduler hides behind IO waits — matches the "比完美适配再高压一点点"
# tuning request.
_IMAGE_WORKERS = 20
_OCR_WORKERS = 3


class PrefetchCache:
    """Per-lecture image pre-fetcher driven by the global image pool.

    schedule()  — fire all downloads for a sub_id, return immediately.
    wait()      — block until every download for sub_id resolves.
    discard()   — drop the cached entry (release the image bytes).

    schedule is idempotent so callers can blindly schedule before wait when
    they're not sure who scheduled the entry first.
    """

    def __init__(self, image_pool: ThreadPoolExecutor):
        self._image_pool = image_pool
        self._lock = threading.Lock()
        # sub_id -> {"items": list[dict] | None, "futures": dict[int, Future]}
        self._cache: dict[str, dict] = {}

    def schedule(self, client: ICourseClient, course_id: str, sub_id: str):
        sub_id = str(sub_id)
        with self._lock:
            if sub_id in self._cache:
                return
            # Reserve the slot so concurrent schedule() calls don't double-fetch
            self._cache[sub_id] = {"items": None, "futures": {}}

        try:
            ppt_items = client.get_ppt_list(course_id, sub_id)
        except Exception as e:
            print(
                f"    [Prefetch {sub_id}] PPT list fetch failed: "
                f"{type(e).__name__}: {e}"
            )
            ppt_items = []
        for idx, item in enumerate(ppt_items, start=1):
            item["page_num"] = idx

        futures: dict[int, "Future"] = {}
        for item in ppt_items:
            futures[item["page_num"]] = self._image_pool.submit(
                fetch_ppt_image, client, item,
            )

        with self._lock:
            self._cache[sub_id]["items"] = ppt_items
            self._cache[sub_id]["futures"] = futures

    def wait(self, sub_id: str) -> tuple[list[dict], dict[int, bytes]]:
        sub_id = str(sub_id)
        with self._lock:
            entry = self._cache.get(sub_id)
        if entry is None:
            return [], {}
        items = entry.get("items") or []
        images: dict[int, bytes] = {}
        for page_num, fut in entry.get("futures", {}).items():
            try:
                img = fut.result()
            except Exception as e:
                print(
                    f"    [Prefetch {sub_id}] page {page_num} download failed: "
                    f"{type(e).__name__}: {e}"
                )
                img = None
            if img is not None:
                images[page_num] = img
        return items, images

    def discard(self, sub_id: str) -> None:
        sub_id = str(sub_id)
        with self._lock:
            self._cache.pop(sub_id, None)


def _ocr_worker(
    db: Database, sub_id: str, page_num: int, image_bytes: bytes,
) -> tuple[int, str]:
    """OCR a single page; classify; persist; return (page_num, status).

    Runs in the OCR pool so multiple instances may overlap with the main
    thread's ASR call.  All DB writes here go through Database._lock so
    concurrent workers don't trip over each other.
    """
    try:
        text = ocr_image_text(image_bytes)
    except Exception as e:
        print(f"      page {page_num}: OCR error {type(e).__name__}: {e}")
        db.update_ppt_page(sub_id, page_num, None, "failed")
        return page_num, "failed"
    status = "invalid" if is_invalid_page(text) else "done"
    db.update_ppt_page(sub_id, page_num, text, status)
    return page_num, status


def _fetch_and_ocr_ppts(
    client: ICourseClient, db: Database, course_id: str, sub_id: str,
) -> None:
    """Drive every PPT page for this lecture to a final ocr_status.

    Four stages, each idempotent so an interrupted run resumes cleanly:
      1. Fetch the latest PPT page list from iCourse and INSERT OR IGNORE
         each into ppt_pages with status='pending'. Repeated calls are safe.
      2. For every still-pending page: download image bytes + compute
         perceptual dhash and store it. Status stays 'pending' so the same
         row can be re-attempted on the next run if we crash here.
      3. Sliding-window dHash dedup over the chronologically-ordered
         pending pages. Losers are stamped 'dedup_dropped' and skipped
         from the OCR loop below.
      4. OCR each survivor; if the recovered text matches one of the
         INVALID_PAGE_PATTERNS (classroom desktop, e-learning portal),
         mark 'invalid'; otherwise 'done' with the text.

    'done' is the only status get_done_ppt_pages returns, so dropped /
    invalid / failed pages naturally vanish from the LLM prompt.
    """
    try:
        ppt_items = client.get_ppt_list(course_id, sub_id)
    except Exception as e:
        print(f"    [WARN] PPT list fetch failed: {type(e).__name__}: {e}")
        ppt_items = []

    if ppt_items:
        for idx, item in enumerate(ppt_items, start=1):
            item["page_num"] = idx
        inserted = db.insert_ppt_pages_pending(sub_id, ppt_items)
        total = db.count_total_ppt_pages(sub_id)
        print(f"    PPT pages: {total} total ({inserted} newly registered)")

    pending = db.get_pending_ppt_pages(sub_id)
    if not pending:
        return

    print(f"    PPT pipeline: {len(pending)} pending page(s)...")

    # Stage 2: download + dHash. We hold images in memory between the dedup
    # and OCR stages so we don't re-download survivors. Failed downloads
    # are immediately stamped 'failed' and never reach OCR.
    image_cache: dict[int, bytes] = {}
    failed_count = 0
    for p in pending:
        page_num = p["page_num"]
        img = fetch_ppt_image(client, p)
        if img is None:
            db.update_ppt_page(sub_id, page_num, None, "failed")
            failed_count += 1
            continue
        image_cache[page_num] = img
        db.update_ppt_page_dhash(sub_id, page_num, compute_dhash(img))

    # Stage 3: dedup. Re-read so each row has its freshly-written dhash.
    pending_with_dhash = [
        p for p in db.get_pending_ppt_pages(sub_id)
        if p["page_num"] in image_cache
    ]
    dhashes = [p.get("dhash") for p in pending_with_dhash]
    dropped_idx = dedup_dhash(dhashes)
    dropped_pages = {pending_with_dhash[i]["page_num"] for i in dropped_idx}
    for page_num in dropped_pages:
        db.update_ppt_page(sub_id, page_num, None, "dedup_dropped")
        image_cache.pop(page_num, None)

    # Stage 4: OCR survivors + classify invalid screens.
    done_count = 0
    invalid_count = 0
    for p in pending_with_dhash:
        page_num = p["page_num"]
        if page_num not in image_cache:
            continue
        try:
            text = ocr_image_text(image_cache[page_num])
        except Exception as e:
            print(f"      page {page_num}: OCR error {type(e).__name__}: {e}")
            db.update_ppt_page(sub_id, page_num, None, "failed")
            failed_count += 1
            continue
        if is_invalid_page(text):
            db.update_ppt_page(sub_id, page_num, text, "invalid")
            invalid_count += 1
        else:
            db.update_ppt_page(sub_id, page_num, text, "done")
            done_count += 1

    print(
        f"    PPT pipeline: {done_count} done, {len(dropped_pages)} dedup'd, "
        f"{invalid_count} invalid, {failed_count} failed"
    )


def process_lecture(
    client: ICourseClient,
    db: Database,
    transcriber: Transcriber,
    summarizer: Summarizer,
    course_id: str,
    course_title: str,
    lecture: dict,
    *,
    prefetch: PrefetchCache,
    ocr_pool: ThreadPoolExecutor,
    next_info: tuple[str, str] | None = None,
) -> str | None:
    """Download, transcribe, OCR PPTs, and summarize a single lecture.

    Concurrency layout:
      Phase A  collect prefetched images (or schedule + wait if not pre-scheduled)
      Phase B  INSERT OR IGNORE pending PPT rows for any newly-listed pages
      Phase C  dHash + dedup + submit OCR jobs to the pool — fully main-thread
      Phase D  ASR transcribe (main thread) — OCR workers churn in parallel
      Phase E  drain OCR futures for accounting
      Phase F  schedule next lecture's prefetch (so its image downloads
               overlap with this lecture's LLM call), then summarize

    Stage-skipping: a previously-saved transcript is reused; PPT OCR
    resumes page-by-page; a v2 (PPT-aware) summary short-circuits the
    whole call. transcript_segments returned by the transcriber are
    kept ONLY in memory and discarded after bucketer assembles the
    LLM prompt.

    Returns the summary string, or None if no summary was produced.
    """
    sub_id = str(lecture["sub_id"])
    sub_title = lecture.get("sub_title", sub_id)
    date = lecture.get("date", "")

    print(f"\n  -- Processing: {sub_title} ({date})")
    print(f"    [Time] Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t_start = time.time()

    existing = db.get_lecture(sub_id)
    has_transcript = existing and existing.get("transcript")
    has_v2_summary = (
        existing
        and existing.get("summary")
        and (existing.get("summary_format_version") or 0) >= 1
    )

    if has_v2_summary:
        # Already produced a PPT-aware summary; nothing left to do but
        # ensure marks are consistent.  Skip prefetch entirely so we don't
        # download images for nothing.
        print(f"    v2 summary exists ({len(existing['summary'])} chars), skipping.")
        if next_info is not None:
            prefetch.schedule(client, next_info[0], next_info[1])
        db.mark_processed(sub_id)
        db.clear_error(sub_id)
        return existing["summary"]

    # ── Phase A — wait for prefetched images ───────────────────────────────
    prefetch.schedule(client, course_id, sub_id)  # idempotent
    ppt_items, images = prefetch.wait(sub_id)

    # ── Phase B — register PPT pages as pending ────────────────────────────
    if ppt_items:
        inserted = db.insert_ppt_pages_pending(sub_id, ppt_items)
        total = db.count_total_ppt_pages(sub_id)
        print(f"    PPT pages: {total} total ({inserted} newly registered)")

    pending = db.get_pending_ppt_pages(sub_id)

    # Sync fallback for any pending row missing from the prefetched images
    # (stale row from a prior interrupted run, or download failure).
    for p in pending:
        page_num = p["page_num"]
        if page_num in images:
            continue
        img = fetch_ppt_image(client, p)
        if img is None:
            db.update_ppt_page(sub_id, page_num, None, "failed")
        else:
            images[page_num] = img

    # ── Phase C — dhash + dedup + submit OCR ───────────────────────────────
    ocr_futures = []
    dropped_pages: set[int] = set()
    if pending:
        for p in pending:
            page_num = p["page_num"]
            img = images.get(page_num)
            if img is None:
                continue
            db.update_ppt_page_dhash(sub_id, page_num, compute_dhash(img))

        pending_with_img = [
            r for r in db.get_pending_ppt_pages(sub_id)
            if r["page_num"] in images
        ]
        dhashes = [r.get("dhash") for r in pending_with_img]
        dropped_idx = dedup_dhash(dhashes)
        dropped_pages = {pending_with_img[i]["page_num"] for i in dropped_idx}
        for page_num in dropped_pages:
            db.update_ppt_page(sub_id, page_num, None, "dedup_dropped")
            images.pop(page_num, None)

        for p in pending_with_img:
            page_num = p["page_num"]
            if page_num in dropped_pages:
                continue
            img = images.get(page_num)
            if img is None:
                continue
            ocr_futures.append(
                ocr_pool.submit(_ocr_worker, db, sub_id, page_num, img)
            )

    # ── Phase D — transcribe (blocks main thread; OCR pool runs in parallel)
    transcript_segments: list[dict] | None = None
    if has_transcript:
        print(f"    Transcript exists ({len(existing['transcript'])} chars), skipping transcription.")
        transcript = existing["transcript"]
    else:
        print(f"    [Time] Fetching video URL at {time.strftime('%H:%M:%S')}")
        video_url = client.get_video_url(course_id, sub_id)
        if not video_url:
            print(f"    No video URL for {sub_id}, skipping.")
            return None

        vpn_url, http_headers = client.get_stream_params(video_url)
        print(f"    [Time] Streaming audio at {time.strftime('%H:%M:%S')}")
        print(f"    [URL] {vpn_url[:100]}...")

        max_attempts = 3
        best_duration = -1.0
        best_transcript = ""
        best_segments: list[dict] = []
        for attempt in range(1, max_attempts + 1):
            try:
                transcript, transcript_segments = transcriber.transcribe_url(
                    vpn_url, http_headers=http_headers,
                )
                db.update_transcript(sub_id, transcript)
                break
            except IncompleteAudioError as e:
                if transcriber._last_duration > best_duration:
                    best_duration = transcriber._last_duration
                    best_transcript = transcriber._last_transcript
                    best_segments = transcriber._last_segments
                print(f"    [WARN] Attempt {attempt}/{max_attempts}: {e}")
                if attempt < max_attempts:
                    _check_session(client)
                    video_url = client.get_video_url(course_id, sub_id)
                    vpn_url, http_headers = client.get_stream_params(video_url)
                    print(f"    Retrying with fresh connection...")
                else:
                    print(
                        f"    [FAIL] All {max_attempts} attempts got incomplete audio, "
                        f"using longest ({best_duration:.0f}s)."
                    )
                    transcript = best_transcript
                    transcript_segments = best_segments
                    db.update_transcript(sub_id, transcript)
            except NoAudioStreamError as e:
                print(f"    [SKIP] Video-only (no audio stream): {e}")
                db.update_error(sub_id, "transcribe", str(e))
                db.mark_processed(sub_id)
                return None
            except Exception as e:
                print(f"    [FAIL] Transcription error: {type(e).__name__}: {e}")
                db.update_error(sub_id, "transcribe", str(e))
                raise

    # ── Phase E — drain OCR pool ───────────────────────────────────────────
    done_count = invalid_count = failed_count = 0
    for fut in as_completed(ocr_futures):
        try:
            _page_num, status = fut.result()
            if status == "done":
                done_count += 1
            elif status == "invalid":
                invalid_count += 1
            elif status == "failed":
                failed_count += 1
        except Exception as e:
            print(f"      OCR worker exception: {type(e).__name__}: {e}")
            failed_count += 1
    if pending:
        print(
            f"    PPT pipeline: {done_count} done, {len(dropped_pages)} dedup'd, "
            f"{invalid_count} invalid, {failed_count} failed"
        )

    # Free this lecture's downloaded image bytes before the LLM call.
    images.clear()
    prefetch.discard(sub_id)

    # ── Phase F — schedule next prefetch, then summarize ──────────────────
    if next_info is not None:
        prefetch.schedule(client, next_info[0], next_info[1])

    if not transcript.strip():
        print(f"    Empty transcript, skipping summary.")
        db.mark_processed(sub_id)
        db.clear_error(sub_id)
        return None

    try:
        kept_pages = db.get_done_ppt_pages(sub_id)
        prompt_text, mode = bucketer.assemble(
            transcript, transcript_segments, kept_pages,
        )
        print(
            f"    [Time] Generating summary at {time.strftime('%H:%M:%S')}"
            f" — mode={mode}, prompt={len(prompt_text)} chars"
        )
        summary, model_used = summarizer.summarize(course_title, prompt_text)
        print(f"    [OK] Summary by {model_used}: {len(summary)} chars")
        db.update_summary_v2(sub_id, summary, model_used)
    except Exception as e:
        print(f"    [FAIL] Summarization error: {type(e).__name__}: {e}")
        db.update_error(sub_id, "summarize", str(e))
        raise

    db.mark_processed(sub_id)
    db.clear_error(sub_id)
    elapsed = time.time() - t_start
    print(f"    [Time] Done at {time.strftime('%H:%M:%S')}: {sub_title} (total {elapsed:.0f}s)")
    return summary


def _resummarize_old_lectures(
    client: ICourseClient,
    db: Database,
    summarizer: Summarizer,
    email_items: list,
    course_ids: list[str],
):
    """Upgrade pre-v2 summaries to PPT-aware v2 format (flat-mode prompt).

    Old lectures (summary_format_version=0) keep their original transcript
    but never had PPT OCR. We:
      1. Fetch + OCR PPT pages on demand.
      2. Re-assemble in flat mode (no segment timestamps available).
      3. Re-summarize, save as v2, reset emailed_at so the user gets the
         updated summary on the next email send.

    Each upgraded lecture is appended to email_items with is_update=True so
    Emailer adds the （含 PPT 识别·更新）subject suffix and a 更新 badge.

    Scoped to ``course_ids``: only lectures whose course_id appears in the
    current run's COURSE_IDS list are upgraded, so we don't pay re-OCR cost
    for courses the user isn't actively monitoring this run.
    """
    targets = db.get_lectures_to_resummarize_for_courses(course_ids)
    if not targets:
        return
    print(f"\n[Resummarize] {len(targets)} lecture(s) eligible for v2 upgrade.")

    seen_sub_ids = {item["sub_id"] for item in email_items}
    for row in targets:
        sub_id = str(row["sub_id"])
        course_id = row["course_id"]
        sub_title = row.get("sub_title", sub_id)
        course_title = row.get("course_title", "Unknown")
        date = row.get("date", "")
        try:
            print(f"  -- Resummarize: {course_title} / {sub_title}")
            _check_session(client)
            try:
                _fetch_and_ocr_ppts(client, db, course_id, sub_id)
            except Exception as e:
                print(f"    [WARN] PPT OCR phase failed: {type(e).__name__}: {e}")

            transcript = row.get("transcript") or ""
            if not transcript.strip():
                print(f"    Empty transcript, cannot resummarize.")
                continue

            kept_pages = db.get_done_ppt_pages(sub_id)
            prompt_text, mode = bucketer.assemble(transcript, None, kept_pages)
            print(
                f"    Prompt: mode={mode}, {len(prompt_text)} chars,"
                f" {len(kept_pages)} kept PPT page(s)"
            )

            summary, model_used = summarizer.summarize(course_title, prompt_text)
            db.update_summary_v2(sub_id, summary, model_used)
            db.reset_emailed(sub_id)
            print(f"    [OK] v2 summary by {model_used}: {len(summary)} chars")

            if sub_id in seen_sub_ids:
                continue
            email_items.append({
                "sub_id": sub_id,
                "course_title": course_title,
                "sub_title": sub_title,
                "date": date,
                "summary": summary,
                "is_update": True,
            })
            seen_sub_ids.add(sub_id)
        except Exception:
            print(f"    [FAIL] Resummarize {sub_id}:")
            traceback.print_exc()


def login_with_retry(max_attempts: int = 5) -> WebVPNSession:
    """Login to WebVPN + iCourse CAS with retry (new session each attempt)."""
    for attempt in range(max_attempts):
        try:
            vpn = WebVPNSession()
            print(f"\n[Login] WebVPN (attempt {attempt + 1}/{max_attempts})...")
            vpn.login()
            print("[Login] iCourse CAS...")
            vpn.authenticate_icourse()
            return vpn
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"  Failed: {type(e).__name__}, retrying...")
                time.sleep(3)
            else:
                raise


def _check_session(client: ICourseClient) -> None:
    """Verify WebVPN session; re-login in place if expired.

    Mutates ``client`` rather than returning a new one — every helper that
    holds a reference (background OCR / image-prefetch threads, retry
    closures) keeps using the same instance and automatically picks up the
    refreshed cookies.
    """
    if client.check_alive():
        return
    print("[Session] WebVPN session expired, re-logging in...")
    client.vpn = login_with_retry()
    client._userinfo = None  # cached for the dead session, must re-fetch


def run():
    """Single execution of the full pipeline."""
    print("=" * 60)
    print("iCourse Subscriber — starting run")
    print("=" * 60)

    if not config.COURSE_IDS:
        print("No COURSE_IDS configured. Set the COURSE_IDS env var.")
        return

    db = Database()
    transcriber = Transcriber()
    summarizer = Summarizer()
    emailer = Emailer() if config.SMTP_EMAIL and config.SMTP_PASSWORD else None

    vpn = login_with_retry()
    client = ICourseClient(vpn)
    email_items = []

    image_pool = ThreadPoolExecutor(
        max_workers=_IMAGE_WORKERS, thread_name_prefix="img",
    )
    ocr_pool = ThreadPoolExecutor(
        max_workers=_OCR_WORKERS, thread_name_prefix="ocr",
    )
    prefetch = PrefetchCache(image_pool)

    try:
        # ── Phase 1 — sync, fast: enumerate every (course, lecture) we'll
        # process this run. Done up-front so the prefetch loop has visibility
        # of "next lecture" across course boundaries.
        all_lectures: list[tuple[str, str, dict]] = []
        for course_id in config.COURSE_IDS:
            try:
                print(f"\n{'─' * 50}")
                print(f"[Course] {course_id}")

                _check_session(client)
                detail = client.get_course_detail(course_id)
                course_title = detail["title"]
                teacher = detail["teacher"]
                lectures = detail["lectures"]
                playback_count = sum(1 for l in lectures if l.get("has_playback"))
                print(f"  Title: {course_title} (Teacher: {teacher})")
                print(f"  Total lectures: {len(lectures)} ({playback_count} with playback)")

                db.upsert_course(course_id, course_title, teacher)

                # Deduplicate by sub_title on the full lecture list first
                # (school system sometimes lists duplicates; doing this on the raw list
                # ensures consistent dedup across runs, not just for pending lectures)
                seen_sub_titles: set[str] = set()
                deduped_lectures = []
                for lec in lectures:
                    title = lec.get("sub_title", "")
                    if title and title in seen_sub_titles:
                        print(f"  [Dedup] Skipping duplicate: {title}"
                              f" (sub_id={lec['sub_id']})")
                        continue
                    if title:
                        seen_sub_titles.add(title)
                    deduped_lectures.append(lec)
                lectures = deduped_lectures

                known_processed = db.get_processed_sub_ids(course_id)
                new_lectures = [
                    lec for lec in lectures
                    if lec.get("has_playback")
                    and str(lec["sub_id"]) not in known_processed
                ]
                unprocessed = db.get_unprocessed_lectures(course_id)
                new_ids = {str(lec["sub_id"]) for lec in new_lectures}
                retry_only = [
                    {"sub_id": u["sub_id"], "sub_title": u["sub_title"], "date": u["date"]}
                    for u in unprocessed if u["sub_id"] not in new_ids
                ]
                new_lectures.extend(retry_only)

                print(f"  New/retry lectures: {len(new_lectures)}")

                if not new_lectures:
                    print("  No new lectures, skipping.")
                    continue

                for lecture in new_lectures:
                    sub_id = str(lecture["sub_id"])
                    db.insert_lecture(
                        sub_id, course_id,
                        lecture.get("sub_title", ""),
                        lecture.get("date", ""),
                    )
                    all_lectures.append((course_id, course_title, lecture))

            except Exception:
                print(f"  ERROR enumerating course {course_id}:")
                traceback.print_exc()

        # ── Phase 2 — concurrent: pre-schedule first lecture's images, then
        # process each lecture in order.  Each call hands off the *next*
        # lecture's identity so process_lecture can schedule its prefetch
        # right before the LLM call (so the image downloads overlap with
        # the LLM round-trip).
        if all_lectures:
            first_course, _, first_lec = all_lectures[0]
            prefetch.schedule(client, first_course, str(first_lec["sub_id"]))

        for i, (course_id, course_title, lecture) in enumerate(all_lectures):
            sub_id = str(lecture["sub_id"])
            next_info: tuple[str, str] | None = None
            if i + 1 < len(all_lectures):
                next_course, _, next_lec = all_lectures[i + 1]
                next_info = (next_course, str(next_lec["sub_id"]))

            _check_session(client)
            try:
                summary = process_lecture(
                    client, db, transcriber, summarizer,
                    course_id, course_title, lecture,
                    prefetch=prefetch, ocr_pool=ocr_pool,
                    next_info=next_info,
                )
                if summary:
                    email_items.append({
                        "sub_id": sub_id,
                        "course_title": course_title,
                        "sub_title": lecture.get("sub_title", sub_id),
                        "date": lecture.get("date", ""),
                        "summary": summary,
                    })
            except Exception:
                print(f"    ERROR processing {sub_id}:")
                traceback.print_exc()
            finally:
                # Belt-and-braces: if process_lecture errored before discarding
                # the cache entry, drop it now so we don't leak image bytes.
                prefetch.discard(sub_id)

        # ── Phase 3 — resummarize old lectures (sequential; rare upgrade path)
        try:
            _check_session(client)
            _resummarize_old_lectures(
                client, db, summarizer, email_items, config.COURSE_IDS,
            )
        except Exception:
            print("[Resummarize] phase errored:")
            traceback.print_exc()

    finally:
        # Drain the pools regardless of how Phase 2 ended.  shutdown(wait=True)
        # blocks until in-flight downloads / OCR jobs finish; without it a
        # crashing main thread would leave them running and racing on DB.
        image_pool.shutdown(wait=True)
        ocr_pool.shutdown(wait=True)

    # Recover any previously processed-but-unsent lectures
    unsent = db.get_unsent_lectures()
    if unsent:
        seen_sub_ids = {item["sub_id"] for item in email_items}
        for row in unsent:
            if row["sub_id"] not in seen_sub_ids:
                email_items.append({
                    "sub_id": row["sub_id"],
                    "course_title": row["course_title"],
                    "sub_title": row["sub_title"],
                    "date": row["date"],
                    "summary": row["summary"],
                })
        print(f"[Email] Including {len(unsent)} previously unsent lecture(s).")

    # Send one email with all summaries
    if emailer and email_items:
        try:
            print(f"\n[Email] Sending summary for {len(email_items)} lecture(s)...")
            if emailer.send(email_items):
                db.mark_emailed_batch([item["sub_id"] for item in email_items])
            else:
                print("[Email] Send failed, lectures will be retried next run.")
        except Exception:
            print("[Email] Failed to send:")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("Run complete.")


if __name__ == "__main__":
    run()
