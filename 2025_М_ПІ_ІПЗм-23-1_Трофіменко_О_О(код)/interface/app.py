import json
import os
import subprocess
import tempfile
import threading
from collections import defaultdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder="templates")

OUTPUTS_BASE = Path("../outputs")
NOVELS_BASE = Path("../novels")

processing_states = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_novel():
    data = request.json
    book_title = data.get("title", "").strip()
    book_text = data.get("text", "").strip()

    if not book_title or not book_text:
        return jsonify({"error": "Please provide both title and text."}), 400

    safe_id = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in book_title)
    book_id = safe_id or "untitled"
    state_id = book_id

    processing_states[state_id] = {
        "steps": [
            {"name": "Saving text", "done": False},
            {"name": "Preparing conda environment", "done": False},
            {"name": "Starting NLP process", "done": False},
            {"name": "Processing NLP", "done": False},
            {"name": "Collecting statistics", "done": False},
        ],
        "done": False,
        "error": None,
        "output_html": None,
        "stats": None,
    }

    def process_task():
        try:
            steps = processing_states[state_id]["steps"]
            novel_path = NOVELS_BASE / f"{book_id}.txt"
            os.makedirs(NOVELS_BASE, exist_ok=True)
            with open(novel_path, "w", encoding="utf-8") as f:
                f.write(book_text)
            steps[0]["done"] = True

            import time
            time.sleep(0.5)
            steps[1]["done"] = True

            output_dir = OUTPUTS_BASE / f"output-{book_id}"
            os.makedirs(output_dir, exist_ok=True)
            steps[2]["done"] = True

            result = subprocess.run(
                [
                    "conda", "run", "-p", "../envs/booknlp_env",
                    "python", "book_nlp_wrap.py",
                    str(novel_path),
                    str(output_dir)
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            steps[3]["done"] = True

            stats = result.stdout
            result_file = output_dir / "novel.book.html"
            output_html = None
            if result_file.exists():
                with open(result_file, encoding="utf-8") as f:
                    output_html = f.read()
            else:
                processing_states[state_id]["error"] = "Result HTML not found."
            steps[4]["done"] = True

            processing_states[state_id]["output_html"] = output_html
            processing_states[state_id]["stats"] = stats
            processing_states[state_id]["done"] = True

        except subprocess.CalledProcessError as e:
            processing_states[state_id][
                "error"] = f"BookNLP processing failed. {e.stderr or ''}"
            processing_states[state_id]["done"] = True
        except Exception as ex:
            processing_states[state_id]["error"] = str(ex)
            processing_states[state_id]["done"] = True

    threading.Thread(target=process_task, daemon=True).start()

    return jsonify({"id": state_id})


@app.route("/progress/<state_id>")
def progress(state_id):
    state = processing_states.get(state_id)
    if not state:
        return jsonify({"error": "Unknown process"}), 404
    resp = {
        "steps": state["steps"],
        "done": state["done"],
        "error": state["error"],
        "output_html": state["output_html"] if state["done"] else None,
        "stats": state["stats"] if state["done"] else None,
    }
    return jsonify(resp)


def create_entity_glossary(entities_path):
    df = pd.read_csv(entities_path, sep='\t')

    clusters = defaultdict(list)
    for _, row in df.iterrows():
        clusters[row['COREF']].append(row)
    glossary = []
    for coref, mentions in clusters.items():
        sorted_mentions = sorted(
            mentions,
            key=lambda m: {'PROP': 0, 'NOM': 1, 'PRON': 2}.get(m['prop'], 99)
        )
        main = sorted_mentions[0]
        entry = {
            'entity': str(main['text']),
            'type': str(main['cat']),
            'translation': '',
            'extra': {
                'coref': int(coref),
                'mention_types': list({m['prop'] for m in mentions}),
                'all_mentions': list({m['text'] for m in mentions})
            }
        }
        glossary.append(entry)
    return glossary


@app.route("/create_glossary", methods=["POST"])
def create_glossary():
    data = request.json
    book_id = data.get("id")
    if not book_id:
        return jsonify({"error": "No id provided"}), 400

    entities_path = OUTPUTS_BASE / f"output-{book_id}" / "novel.entities"
    if not entities_path.exists():
        return jsonify({"error": "Entities file not found."}), 404

    glossary = create_entity_glossary(str(entities_path))
    glossary_path = OUTPUTS_BASE / f"output-{book_id}" / "entity_glossary.json"
    with open(glossary_path, "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)

    return jsonify({"glossary": glossary})


@app.route("/delete_entity", methods=["POST"])
def delete_entity():
    data = request.json
    book_id = data.get("id")
    entity = data.get("entity")
    if not book_id or entity is None:
        return jsonify({"error": "No id or entity"}), 400

    glossary_path = OUTPUTS_BASE / f"output-{book_id}" / "entity_glossary.json"
    if not glossary_path.exists():
        return jsonify({"error": "Glossary file not found."}), 404

    with open(glossary_path, encoding="utf-8") as f:
        glossary = json.load(f)

    glossary = [entry for entry in glossary if entry["entity"] != entity]
    with open(glossary_path, "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)

    return jsonify({"glossary": glossary})


@app.route("/download_glossary/<book_id>")
def download_glossary(book_id):
    glossary_path = OUTPUTS_BASE / f"output-{book_id}" / "entity_glossary.json"
    if not glossary_path.exists():
        return "Glossary not found.", 404
    return send_file(glossary_path, as_attachment=True, download_name="entity_glossary.json")


# TRANSLATION

@app.route("/translation")
def translation():
    return render_template("translation.html")


translation_states = {}


@app.route("/start_translation", methods=["POST"])
def start_translation():
    data = request.json
    text = data.get("text", "")
    target_lang = data.get("target_lang", "")
    model = data.get("model", "")
    glossary = data.get("glossary", None)
    progressive = data.get("progressive", False)

    if not text or not target_lang or not model:
        return jsonify({"error": "Please provide all fields"}), 400

    tid = str(abs(hash(text + target_lang + model)))
    temp_dir = tempfile.mkdtemp(prefix="translation-")
    input_path = os.path.join(temp_dir, "input.txt")

    glossary_path = None
    if glossary:
        glossary_path = os.path.join(temp_dir, "glossary.json")
        with open(glossary_path, "w", encoding="utf-8") as g:
            json.dump(glossary, g, ensure_ascii=False, indent=2)
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(text)
    output_path = os.path.join(temp_dir, "output.txt")
    metadata_path = os.path.join(temp_dir, "output_metadata.json")

    translation_states[tid] = {
        "progress": "Queued",
        "done": False,
        "error": None,
        "output": "",
        "chunks_info": None,
        "tid": tid
    }

    def translation_task():
        args = [
            "conda", "run", "-p", "../envs/llm_env",
            "python", "llm_translation_wrap.py",
            input_path, output_path, target_lang, model
        ]
        if glossary_path:
            args.append("--glossary")
            args.append(glossary_path)
        if progressive:
            args.append("--progressive")

        try:
            translation_states[tid]["progress"] = "Translating"
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=True,
            )
            translation_states[tid]["progress"] = "Completed"

            with open(output_path, encoding="utf-8") as out_f:
                translation_states[tid]["output"] = out_f.read()
            if os.path.exists(metadata_path):
                with open(metadata_path, encoding="utf-8") as meta_f:
                    translation_states[tid]["chunks_info"] = json.load(meta_f)
            translation_states[tid]["done"] = True
        except subprocess.CalledProcessError as e:
            translation_states[tid]["error"] = e.stderr or str(e)
            translation_states[tid]["done"] = True

    threading.Thread(target=translation_task, daemon=True).start()
    return jsonify({"tid": tid})


@app.route("/translation_progress/<tid>")
def translation_progress(tid):
    state = translation_states.get(tid)
    if not state:
        return jsonify({"error": "Unknown process"}), 404
    return jsonify(state)


@app.route("/analyze_glossary", methods=["POST"])
def analyze_glossary():
    data = request.json
    text = data.get("text", "")
    glossary = data.get("glossary")
    model = data.get("model", "")
    if not text or not glossary or not model:
        return jsonify({"error": "Missing data"}), 400
    tid = str(abs(hash(text + model + json.dumps(glossary))))
    temp_dir = tempfile.mkdtemp(prefix="glossary-analyze-")
    input_path = os.path.join(temp_dir, "input.txt")
    glossary_path = os.path.join(temp_dir, "glossary.json")
    output_path = os.path.join(temp_dir, "entity_suggestions.json")
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(glossary_path, "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)

    result = subprocess.run([
        "conda", "run", "-p", "../envs/llm_env",
        "python", "glossary_entity_analyze.py",
        input_path, glossary_path, model, output_path
    ], capture_output=True, text=True)
    if not os.path.exists(output_path):
        return jsonify({"error": result.stderr or "Analysis failed"}), 500
    with open(output_path, encoding="utf-8") as f:
        entity_list = json.load(f)
    return jsonify({"entities": entity_list})

# Comparison


@app.route("/compare")
def compare():
    return render_template("compare.html")


nltk.download('punkt', quiet=True)


@app.route("/compare_texts", methods=["POST"])
def compare_texts():
    data = request.json
    text1 = data.get("text1", "")
    text2 = data.get("text2", "")

    if not text1 or not text2:
        return jsonify({"error": "Please provide both texts"}), 400

    model = SentenceTransformer(
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    embeddings_refs = model.encode([text1], convert_to_tensor=False)
    embeddings_preds = model.encode([text2], convert_to_tensor=False)
    cos_sim = cosine_similarity(embeddings_refs, embeddings_preds)
    similarity = float(np.mean(np.diagonal(cos_sim))) * 100  # Процент

    return jsonify({"similarity": round(similarity, 2)})


if __name__ == "__main__":
    app.run(debug=True)
