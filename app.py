# app.py
from flask import Flask, render_template, request, jsonify
import traceback
import os
import html
import sys
import subprocess
import warnings
import re
import string
from heapq import nlargest

# Abstractive summarization depends on heavy libraries (torch, transformers).
# Import them lazily and fail gracefully so the server can still run for testing the
# transcript-fetching and extractive fallback.
abstractive_summarizer = None
abstractive_tokenizer = None
abstractive_model = None
abstractive_pipeline = None
try:
    import torch
    # Prefer the high-level transformers pipeline for abstractive summarization when available
    from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
    have_transformers = True
except Exception:
    # Missing heavy dependencies; keep names defined as None so later code can
    # safely detect availability and continue running the Flask app.
    torch = None
    pipeline = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
    have_transformers = False

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import warnings
warnings.filterwarnings("ignore")

# Optional transcript fetching (primary functionality delegated to temp.py)
try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
except Exception:
    YouTubeTranscriptApi = None
    TranscriptsDisabled = Exception
    NoTranscriptFound = Exception

# Import the user's transcript helper (temp.py) and use it as the single source of truth
try:
    from temp import fetch_transcript, segments_to_text, translate_text
except Exception:
    fetch_transcript = None
    segments_to_text = None
    translate_text = None


def extract_transcript_from_url(youtube_video_url: str) -> str:
    """Use the helper from `temp.py` to fetch and format the transcript.

    Expects `fetch_transcript(video_id, lang)` and `segments_to_text(segments, include_timestamps)`
    to be available in `temp.py`. Raises on failure with informative messages.
    """
    # Extract video id
    vid_match = re.search(r"v=([a-zA-Z0-9_-]{11})", youtube_video_url)
    if not vid_match:
        vid_match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", youtube_video_url)
    if not vid_match:
        raise ValueError("Invalid YouTube URL or video ID not found.")

    video_id = vid_match.group(1)

    if fetch_transcript is None or segments_to_text is None:
        raise RuntimeError("Transcript helper not available. Ensure `temp.py` exists and exports fetch_transcript and segments_to_text.")

    # Use the provided helper which uses youtube_transcript_api internally
    segments = fetch_transcript(video_id, lang='en')
    text = segments_to_text(segments, include_timestamps=False)
    return text

app = Flask(__name__)

# Initialize models
print("Loading models... This may take a few minutes.")

# Load abstractive model (PEGASUS)
# Use a standard transformers seq2seq model for abstractive summarization.
# We pick 't5-base' by default (works well with the transformers pipeline). Change
# this to a lighter model like 't5-small' if you need a smaller footprint.
abstractive_model_name = "t5-base"
if have_transformers:
    try:
        # Try to instantiate a high-level pipeline which handles tokenization and generation
        abstractive_pipeline = pipeline("summarization", model=abstractive_model_name, tokenizer=abstractive_model_name)
        print("✓ Abstractive pipeline (transformers) loaded successfully.")
    except Exception as e:
        print(f"✗ Failed to load abstractive pipeline: {e}")
        abstractive_pipeline = None
else:
    abstractive_pipeline = None

# Load extractive model (spaCy)
try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ Extractive model (spaCy) loaded successfully.")
except Exception as e:
    print(f"✗ Failed to load extractive model: {e}")
    # Try to download the model programmatically (best-effort). If this fails, we keep a fallback.
    try:
        print("Attempting to download 'en_core_web_sm' automatically...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        print("✓ Extractive model (spaCy) downloaded and loaded successfully.")
    except Exception as e2:

            # Use the provided helper in temp.py for transcript fetching and formatting
        try:
            from temp import fetch_transcript, segments_to_text
        except Exception:
            fetch_transcript = None
            segments_to_text = None
        print(f"✗ Automatic download failed: {e2}")
        print("Extractive summarization will fall back to a lightweight local method.")
        nlp = None

def abstractive_summarize(text, max_length=150, min_length=30):
    """
    Generate abstractive summary using PEGASUS model.

    This version chunks long inputs into manageable pieces, summarizes each chunk,
    then (if multiple chunks) summarizes the concatenated chunk-summaries to
    produce a final concise summary. This avoids tokenizer/model indexing errors
    on very long transcripts.
    """
    # Prefer using the high-level transformers pipeline if available
    if abstractive_pipeline is not None:
        # chunk input similarly to avoid extremely long inputs
        chunk_char_limit = 40000
        paras = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
        if not paras:
            paras = [text]

        chunks = []
        current = []
        current_len = 0
        for p in paras:
            if current_len + len(p) + 1 <= chunk_char_limit:
                current.append(p)
                current_len += len(p) + 1
            else:
                if current:
                    chunks.append(" ".join(current))
                current = [p]
                current_len = len(p) + 1
        if current:
            chunks.append(" ".join(current))

        summaries = []
        for c in chunks:
            try:
                # Use `max_new_tokens` to control generation length (preferred over max_length)
                out = abstractive_pipeline(c, max_new_tokens=max_length, min_length=min_length, do_sample=False)
                if out and isinstance(out, list):
                    summaries.append(out[0].get('summary_text', '').strip())
            except Exception as e:
                # propagate to caller to allow fallback behavior
                raise

        if not summaries:
            raise RuntimeError("Abstractive summarizer returned no output for any chunk.")

        if len(summaries) == 1:
            return summaries[0]

        combined = " ".join(summaries)
        out = abstractive_pipeline(combined, max_new_tokens=max_length, min_length=min_length, do_sample=False)
        if out and isinstance(out, list):
            return out[0].get('summary_text', '').strip()
        raise RuntimeError("Final abstractive summarization returned no output.")
    # If transformers pipeline not available, fall back to previously loaded model/tokenizer
    if abstractive_model is None or abstractive_tokenizer is None:
        return "Abstractive summarization model is not available."

    # Fallback: use the direct model/tokenizer approach (kept for compatibility)
    # Chunk by paragraphs until a character threshold to avoid extremely long inputs.
    chunk_char_limit = 4000
    paras = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
    if not paras:
        paras = [text]

    chunks = []
    current = []
    current_len = 0
    for p in paras:
        if current_len + len(p) + 1 <= chunk_char_limit:
            current.append(p)
            current_len += len(p) + 1
        else:
            if current:
                chunks.append(" ".join(current))
            current = [p]
            current_len = len(p) + 1
    if current:
        chunks.append(" ".join(current))

    summaries = []
    for c in chunks:
        inputs = abstractive_tokenizer(c, truncation=True, max_length=1024, return_tensors="pt")
        try:
            summary_ids = abstractive_model.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            if summary_ids is None or len(summary_ids) == 0:
                continue
            summaries.append(abstractive_tokenizer.decode(summary_ids[0], skip_special_tokens=True))
        except IndexError as ie:
            raise RuntimeError(f"Abstractive summarization failed (index error): {ie}")
        except Exception:
            raise

    if not summaries:
        raise RuntimeError("Abstractive summarizer returned no output for any chunk.")

    if len(summaries) == 1:
        return summaries[0]

    combined = " ".join(summaries)
    inputs = abstractive_tokenizer(combined, truncation=True, max_length=1024, return_tensors="pt")
    final_ids = abstractive_model.generate(
        inputs.input_ids,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    if final_ids is None or len(final_ids) == 0:
        raise RuntimeError("Final abstractive summarization returned no output.")
    return abstractive_tokenizer.decode(final_ids[0], skip_special_tokens=True)


def extractive_summarize(text, num_sentences=None, compression_ratio=0.3, return_list=False):
    """
    Generate extractive summary using spaCy and frequency-based scoring
    """
    # If spaCy model is available, use it. Otherwise, use a lightweight regex-based fallback.
    if nlp is not None:
        # Process text with spaCy
        doc = nlp(text)

        # Calculate word frequencies (excluding stop words and punctuation)
        word_frequencies = {}
        for word in doc:
            w = word.text.lower()
            if w not in STOP_WORDS and w not in string.punctuation:
                word_frequencies[w] = word_frequencies.get(w, 0) + 1

        # Normalize frequencies
        max_frequency = max(word_frequencies.values()) if word_frequencies else 1
        for w in list(word_frequencies.keys()):
            word_frequencies[w] = word_frequencies[w] / max_frequency

        # Score sentences based on word frequencies
        sentence_scores = {}
        sentences = list(doc.sents)
        for sent in sentences:
            for word in sent:
                w = word.text.lower()
                if w in word_frequencies:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[w]

        # Determine how many sentences to select
        if num_sentences is None:
            select_length = int(len(sentences) * compression_ratio)
        else:
            try:
                select_length = int(num_sentences)
            except Exception:
                select_length = int(len(sentences) * compression_ratio)

        # Bound the selection to a reasonable range
        select_length = max(1, min(select_length, 10))  # Between 1 and 10 sentences

        if not sentence_scores:
            return ''

        if select_length >= len(sentence_scores):
            best = nlargest(len(sentence_scores), sentence_scores, key=sentence_scores.get)
        else:
            best = nlargest(select_length, sentence_scores, key=sentence_scores.get)

        # Preserve original order of sentences
        ordered = sorted(best, key=lambda s: sentences.index(s))
        final_summary = [sentence.text for sentence in ordered]
        if return_list:
            return final_summary
        return ' '.join(final_summary)

    # Lightweight fallback (no spaCy model available)
    # - Split into sentences with regex
    # - Compute simple word-frequency scores (ignores punctuation and a basic stopword set)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return ''

    # Build frequency table
    words = re.findall(r"\w+", text.lower())
    word_frequencies = {}
    for w in words:
        if w in STOP_WORDS or w in string.punctuation:
            continue
        word_frequencies[w] = word_frequencies.get(w, 0) + 1

    if not word_frequencies:
        # If nothing to score, return first sentence as fallback
        return sentences[0]

    max_freq = max(word_frequencies.values())
    for w in word_frequencies:
        word_frequencies[w] = word_frequencies[w] / max_freq

    sentence_scores = {}
    for sent in sentences:
        for w in re.findall(r"\w+", sent.lower()):
            if w in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[w]

    if num_sentences is None:
        select_length = int(len(sentences) * compression_ratio)
    else:
        try:
            select_length = int(num_sentences)
        except Exception:
            select_length = int(len(sentences) * compression_ratio)

    select_length = max(1, min(select_length, 10))

    if not sentence_scores:
        # Fallback to first sentence
        if return_list:
            return [sentences[0]]
        return sentences[0]

    if select_length >= len(sentence_scores):
        best = nlargest(len(sentence_scores), sentence_scores, key=sentence_scores.get)
    else:
        best = nlargest(select_length, sentence_scores, key=sentence_scores.get)

    # Preserve the original order
    ordered = sorted(best, key=lambda s: sentences.index(s))
    if return_list:
        return ordered
    return ' '.join(ordered)

@app.route('/')
def home():
    # expose whether abstractive model is available so the UI can disable that option
    # the runtime may use `abstractive_pipeline` (transformers pipeline) as the
    # indicator that abstractive summarization is available
    abstractive_available = (abstractive_pipeline is not None)
    return render_template('index.html', abstractive_available=abstractive_available)

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        # Support providing a YouTube URL instead of raw text
        youtube_url = data.get('youtube_url', '').strip()
        summary_type = data.get('type', 'abstractive')
        
        # If a YouTube URL is provided, attempt to fetch its transcript
        if not text and youtube_url:
            try:
                text = extract_transcript_from_url(youtube_url) or ''
            except Exception as e:
                return jsonify({'error': f'Could not fetch transcript: {str(e)}'}), 400

        if not text:
            return jsonify({'error': 'Please enter some text to summarize or provide a valid youtube_url.'}), 400
        
        if len(text.split()) < 10:
            return jsonify({'error': 'Please enter longer text for summarization (at least 10 words).'}), 400
        
        # Allow user to request a specific number of sentences for extractive mode
        num_sentences = data.get('num_sentences', None)

        # Choose summarization method based on user selection
        # Attempt abstractive when requested; on failure, fall back to extractive
        fallback_used = False
        fallback_reason = None
        if summary_type == 'abstractive':
            try:
                summary = abstractive_summarize(text)
                summary_points = None
            except Exception as e:
                # Log and fall back to extractive
                tb = traceback.format_exc()
                print("Abstractive summarization failed, falling back to extractive. Traceback:\n", tb)
                fallback_used = True
                fallback_reason = str(e)
                summary_points = extractive_summarize(text, num_sentences=num_sentences, return_list=True)
                summary = ' '.join(summary_points) if isinstance(summary_points, (list, tuple)) else str(summary_points)

        elif summary_type == 'extractive':
            summary_points = extractive_summarize(text, num_sentences=num_sentences, return_list=True)
            summary = ' '.join(summary_points) if isinstance(summary_points, (list, tuple)) else str(summary_points)
        else:
            return jsonify({'error': 'Invalid summary type. Choose either "abstractive" or "extractive".'}), 400

        # Calculate compression statistics
        original_words = len(text.split())
        summary_words = len(summary.split())
        compression_percentage = ((original_words - summary_words) / original_words) * 100

        response_payload = {
            'summary': summary,
            'original_length': original_words,
            'summary_length': summary_words,
            'compression_rate': f"{compression_percentage:.1f}%"
        }

        # If abstractive failed and we fell back, include that info in the response
        if 'fallback_used' in locals() and fallback_used:
            response_payload['fallback'] = True
            response_payload['fallback_reason'] = fallback_reason

        # Include sentence points when available (extractive mode)
        if summary_points is not None:
            response_payload['summary_points'] = summary_points

        return jsonify(response_payload)
    
    except Exception as e:
        return jsonify({'error': f'An error occurred during summarization: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)