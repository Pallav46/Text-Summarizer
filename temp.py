#!/usr/bin/env python3
"""
Fetch YouTube transcript and output in a requested language (default: English).

Usage:
  python transcript.py <VIDEO_ID> [--lang en] [--out file.txt] [--timestamps]

This script uses `youtube-transcript-api` to fetch captions. If no transcript
is available in the requested language, it will fetch any available transcript
and auto-translate it to the target language using `googletrans`.
"""
import argparse
import sys
import html
from typing import List

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
except Exception:
    YouTubeTranscriptApi = None

# googletrans (unofficial) will be used for translation fallback
try:
    from googletrans import Translator
except Exception:
    Translator = None


def fetch_transcript(video_id: str, lang: str = 'en') -> List[dict]:
    """Try to fetch transcript in requested language; fallback to any language.

    Returns a list of segments with keys: text, start, duration.
    """
    if YouTubeTranscriptApi is None:
        raise RuntimeError('youtube_transcript_api not installed. See requirements.txt')

    ytt = YouTubeTranscriptApi()
    # The package exposes a richer API: .fetch(...) and .list(...). We'll try to fetch
    # a transcript for the requested language first, otherwise pick the first available
    # transcript and return its segments.
    try:
        # fetch will try the requested languages in order
        fetched = ytt.fetch(video_id, languages=(lang,))
        # fetched behaves like a list of segments
        # normalize fetched snippet objects to dicts
        out = []
        for s in fetched:
            out.append({
                'text': getattr(s, 'text', '') or '',
                'start': getattr(s, 'start', 0.0),
                'duration': getattr(s, 'duration', 0.0),
                'language': getattr(s, 'language', None),
                'language_code': getattr(s, 'language_code', None),
            })
        return out
    except Exception:
        # fallback: list all transcripts and pick a transcript we can fetch
        transcript_list = ytt.list(video_id)
        # try to find transcript matching the requested language (may raise)
        try:
            t = transcript_list.find_transcript([lang])
        except Exception:
            # pick the first available transcript
            try:
                t = next(iter(transcript_list))
            except StopIteration:
                raise RuntimeError('No transcripts available for this video')

        # t is a Transcript object; fetch segments
        segments = t.fetch()
        # convert snippet objects to plain dicts for compatibility
        out = []
        for s in segments:
            out.append({
                'text': getattr(s, 'text', '') or '',
                'start': getattr(s, 'start', 0.0),
                'duration': getattr(s, 'duration', 0.0),
                'language': getattr(s, 'language', None),
                'language_code': getattr(s, 'language_code', None),
            })
        return out


def segments_to_text(segments: List[dict], include_timestamps: bool = False) -> str:
    if include_timestamps:
        lines = [f"[{s.get('start'):.2f}] {s.get('text','').strip()}" for s in segments]
        return "\n".join(lines)
    return " ".join(s.get('text', '').strip() for s in segments)


def translate_text(text: str, dest: str = 'en') -> str:
    if Translator is None:
        raise RuntimeError('googletrans not installed. See requirements.txt')
    t = Translator()
    # googletrans may choke on extremely long strings; translate in chunks if needed
    # split into ~2000 char chunks
    chunks = [text[i:i+1800] for i in range(0, len(text), 1800)]
    translated = []
    for c in chunks:
        res = t.translate(c, dest=dest)
        translated.append(res.text)
    return ' '.join(translated)


def main():
    parser = argparse.ArgumentParser(description='Fetch YouTube transcript (default English output).')
    parser.add_argument('video_id', help='YouTube video id or URL')
    parser.add_argument('--lang', '-l', default='en', help='Target language code (default: en)')
    parser.add_argument('--out', '-o', help='Write transcript to file')
    parser.add_argument('--timestamps', action='store_true', help='Include timestamps in output')
    args = parser.parse_args()

    if YouTubeTranscriptApi is None:
        print('Dependency missing: youtube_transcript_api. Install with: pip install -r requirements.txt', file=sys.stderr)
        sys.exit(2)

    try:
        segments = fetch_transcript(args.video_id, lang=args.lang)
        # If the transcript returned is not in the requested lang, youtube_transcript_api
        # will have returned whatever it could; detect language by checking the first segment
        text = segments_to_text(segments, include_timestamps=args.timestamps)

        # If requested lang is not the same as segment 'language' keys or we explicitly want
        # to ensure English output, run through translator when necessary.
        # youtube_transcript_api may return translated transcript if possible; there's no
        # universal flag, so we do a safe fallback: if lang != 'en' and user asked 'en', we
        # still accept the returned text; but if original language is different and we need
        # to force translation, translate.

        # Heuristic: if requested lang is 'en' but segments contain non-ascii letters, translate.
        need_translate = False
        if args.lang == 'en':
            if any(any(ord(ch) > 127 for ch in seg.get('text', '')) for seg in segments):
                need_translate = True

        if need_translate:
            if Translator is None:
                print('Transcript appears to be non-English and googletrans is not installed.\nInstall with: pip install -r requirements.txt', file=sys.stderr)
                # still print raw transcript
            else:
                # translate the joined text (no timestamps) and then either print with timestamps or not
                joined = " ".join(s.get('text','') for s in segments)
                translated = translate_text(joined, dest=args.lang)
                translated = html.unescape(translated)
                if args.out:
                    with open(args.out, 'w', encoding='utf-8') as f:
                        f.write(translated)
                print('\n--- Transcript (translated) ---\n')
                print(translated)
                return

        # decode HTML entities (some transcripts contain &amp;#39; etc.)
        text = html.unescape(text)

        if args.out:
            with open(args.out, 'w', encoding='utf-8') as f:
                f.write(text)

        print('\n--- Transcript ---\n')
        print(text)

    except TranscriptsDisabled:
        print('Transcripts are disabled for this video.', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print('Error fetching transcript:', str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
