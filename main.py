"""iCourse Subscriber — main orchestration.

Runs a single check: login → detect new lectures → stream audio → transcribe
→ summarize → email. Designed to be triggered by GitHub Actions cron.
"""

import time
import traceback

from src import bucketer, config
from src.database import Database
from src.emailer import Emailer
from src.icourse import ICourseClient
from src.ocr import ocr_image_text
from src.ppt_dedup import filter_pages
from src.ppt_fetcher import fetch_ppt_image
from src.summarizer import Summarizer
from src.transcriber import IncompleteAudioError, NoAudioStreamError, Transcriber
from src.webvpn import WebVPNSession


def _fetch_and_ocr_ppts(
    client: ICourseClient, db: Database, course_id: str, sub_id: str,
) -> None:
    """Make sure every PPT page for this lecture has reached a final OCR status.

    Steps:
      1. Fetch the latest PPT page list from iCourse and INSERT OR IGNORE
         each into ppt_pages with status='pending'. Repeated calls are safe;
         only previously-unknown pages are added.
      2. For every still-pending row, download the image and OCR it. Each
         page commits to DB independently so the work is resumable across
         interrupted runs and concurrent workers (SQLite WAL row-level locks).

    Failures (network, OCR engine) flip the row to 'failed' and continue;
    bucketer queries only 'done' rows so 'failed' pages simply drop out.
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

    print(f"    OCR: {len(pending)} pending page(s)...")
    done_count = 0
    failed_count = 0
    for p in pending:
        page_num = p["page_num"]
        img = fetch_ppt_image(client, p)
        if img is None:
            db.update_ppt_page(sub_id, page_num, None, "failed")
            failed_count += 1
            continue
        try:
            text = ocr_image_text(img)
        except Exception as e:
            print(f"      page {page_num}: OCR error {type(e).__name__}: {e}")
            db.update_ppt_page(sub_id, page_num, None, "failed")
            failed_count += 1
            continue
        db.update_ppt_page(sub_id, page_num, text, "done")
        done_count += 1
    print(f"    OCR done: {done_count} ok, {failed_count} failed")


def _build_filtered_pages(db: Database, sub_id: str) -> list[dict]:
    """Read all done OCR pages for sub_id, drop classroom-desktop / dup pages.

    Filtering is done in-memory every prompt build so the DB stays a faithful
    record of what the OCR step produced; removed pages can resurface if the
    rules change later.
    """
    all_done = db.get_done_ppt_pages(sub_id)
    if not all_done:
        return []
    kept, stats = filter_pages(all_done)
    if stats["desktop_dropped"] or stats["jaccard_dropped"]:
        print(
            f"    Filter: -{stats['desktop_dropped']} desktop,"
            f" -{stats['jaccard_dropped']} dup → {len(kept)} pages kept"
        )
    return kept


def process_lecture(
    client: ICourseClient,
    db: Database,
    transcriber: Transcriber,
    summarizer: Summarizer,
    course_id: str,
    course_title: str,
    lecture: dict,
) -> str | None:
    """Download, transcribe, OCR PPTs, and summarize a single lecture.

    Stage-skipping: a previously-saved transcript is reused; PPT OCR is
    resumed page-by-page; a v2 (PPT-aware) summary is reused as-is. The
    transcript_segments returned by the transcriber are kept ONLY in
    memory and discarded after the bucketer assembles the LLM prompt.

    Returns the summary string, or None if no summary was produced.
    """
    sub_id = str(lecture["sub_id"])
    sub_title = lecture.get("sub_title", sub_id)
    date = lecture.get("date", "")

    print(f"\n  -- Processing: {sub_title} ({date})")
    print(f"    [Time] Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t_start = time.time()

    # Check existing progress for stage-skipping
    existing = db.get_lecture(sub_id)
    has_transcript = existing and existing.get("transcript")
    has_v2_summary = (
        existing
        and existing.get("summary")
        and (existing.get("summary_format_version") or 0) >= 1
    )

    transcript_segments: list[dict] | None = None

    # 1) Transcribe (stream audio directly from CDN — no video download)
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
                # transcriber stashes the partial result on _last_*; remember
                # the *longest* attempt so a fluky shorter retry doesn't
                # overwrite an earlier mostly-complete one.
                if transcriber._last_duration > best_duration:
                    best_duration = transcriber._last_duration
                    best_transcript = transcriber._last_transcript
                    best_segments = transcriber._last_segments
                print(f"    [WARN] Attempt {attempt}/{max_attempts}: {e}")
                if attempt < max_attempts:
                    # Re-login and get fresh URL for retry
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

    # 2) PPT list + OCR (resumable page-by-page)
    try:
        _fetch_and_ocr_ppts(client, db, course_id, sub_id)
    except Exception as e:
        # OCR is best-effort; failures shouldn't kill the lecture.
        print(f"    [WARN] PPT OCR phase failed: {type(e).__name__}: {e}")

    # 3) Summarize
    if not transcript.strip():
        print(f"    Empty transcript, skipping summary.")
        db.mark_processed(sub_id)
        db.clear_error(sub_id)
        return None

    if has_v2_summary:
        print(f"    v2 summary exists ({len(existing['summary'])} chars), skipping.")
        summary = existing["summary"]
    else:
        try:
            kept_pages = _build_filtered_pages(db, sub_id)
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
    """
    targets = db.get_lectures_to_resummarize()
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

            kept_pages = _build_filtered_pages(db, sub_id)
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

            # Find new lectures with playback + previously failed (unprocessed) ones
            known_processed = db.get_processed_sub_ids(course_id)
            new_lectures = [
                lec for lec in lectures
                if lec.get("has_playback")
                and str(lec["sub_id"]) not in known_processed
            ]
            # Also retry any previously inserted but unprocessed
            unprocessed = db.get_unprocessed_lectures(course_id)
            new_ids = {str(lec["sub_id"]) for lec in new_lectures}
            # Merge: new from API + retries from DB
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
                _check_session(client)
                try:
                    summary = process_lecture(
                        client, db, transcriber, summarizer,
                        course_id, course_title, lecture,
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

        except Exception:
            print(f"  ERROR processing course {course_id}:")
            traceback.print_exc()

    # Upgrade pre-v2 (no PPT OCR) summaries before the email step so the user
    # sees the new content in the same digest.
    try:
        _check_session(client)
        _resummarize_old_lectures(client, db, summarizer, email_items)
    except Exception:
        print("[Resummarize] phase errored:")
        traceback.print_exc()

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
