import argparse
import json
import os
import re
import sys

import nltk
from huggingface_hub import InferenceClient

nltk.download('punkt_tab', quiet=True)
inference = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct", provider="together")


def chunk_text_semantically(text, max_tokens=300):
    paragraphs = text.strip().split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        tokens = para.split()
        token_count = len(tokens)

        if current_length + token_count > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = token_count
        else:
            current_chunk.append(para)
            current_length += token_count

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


def load_glossary(glossary_json_path):
    with open(glossary_json_path, 'r', encoding='utf-8') as f:
        glossary = json.load(f)

    formatted_glossary = ""

    for category, entries in glossary.items():
        formatted_glossary += f"{category.capitalize()}:\n"
        for term, details in entries.items():
            if category.lower() in ["characters", "locations"] and isinstance(details, dict):
                translation = details.get("translation", "")
                extra = details.get("extra", "")
                age = details.get("age", "")
                gender = details.get("gender", "")

                details_list = []
                if translation:
                    details_list.append(translation)
                if gender:
                    details_list.append(gender)
                if age:
                    details_list.append(age)
                if extra:
                    details_list.append(extra)

                details_str = ", ".join(details_list)
                formatted_glossary += f"  - {term}: {details_str}\n"
            else:
                formatted_glossary += f"  - {term}: {details}\n"

    return formatted_glossary.strip()


def load_context(context_json_path):
    with open(context_json_path, 'r', encoding='utf-8') as f:
        context = json.load(f)

    location = context.get("location", "")
    situation = context.get("situation", "")

    formatted_context = f"{location}, {situation}"

    return formatted_context.strip()


def analyze_and_update_context(chunk, previous_context=None):
    if previous_context is None:
        context = {"location": "", "situation": ""}
    else:
        if isinstance(previous_context, str):
            location_match = re.search(
                r"location: ([^,]+)", previous_context, re.IGNORECASE)
            situation_match = re.search(
                r"situation: ([^,]+)", previous_context, re.IGNORECASE)

            context = {
                "location": location_match.group(1).strip() if location_match else "",
                "situation": situation_match.group(1).strip() if situation_match else ""
            }
        else:
            context = previous_context

    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes narrative text and extracts setting and situation."},
        {"role": "user", "content": f"""
Analyze this text and update the narrative context.
If this information contradicts previous context, prioritize the new information.

Previous context:
Location: {context.get('location', '')}
Situation: {context.get('situation', '')}

Text chunk to analyze:
{chunk}

Return ONLY a JSON object with these keys:
1. \"location\" (current setting)
2. \"situation\" (what's happening)
"""}
    ]

    completion = inference.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=messages,
        max_tokens=500,
    )

    response = completion.choices[0].message.content or ""

    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            updated_context = json.loads(json_match.group(0))
            return updated_context
        else:
            context_dict = {}
            location_match = re.search(r'"location":\s*"([^"]+)"', response)
            situation_match = re.search(r'"situation":\s*"([^"]+)"', response)

            if location_match:
                context_dict["location"] = location_match.group(1)
            if situation_match:
                context_dict["situation"] = situation_match.group(1)

            return context_dict
    except json.JSONDecodeError:
        location_match = re.search(r'"location":\s*"([^"]+)"', response)
        situation_match = re.search(r'"situation":\s*"([^"]+)"', response)

        updated_context = {}
        if location_match:
            updated_context["location"] = location_match.group(1)
        else:
            updated_context["location"] = context.get("location", "")

        if situation_match:
            updated_context["situation"] = situation_match.group(1)
        else:
            updated_context["situation"] = context.get("situation", "")

        return updated_context


def format_context_for_translation(context_dict):
    formatted = []

    if context_dict.get("location"):
        formatted.append(f"Location: {context_dict['location']}")

    if context_dict.get("situation"):
        formatted.append(f"Situation: {context_dict['situation']}")

    return "\n".join(formatted)


def progressive_context_translation(input_file, output_file, target_language, glossary_path=None):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    if glossary_path:
        glossary = load_glossary(glossary_path)
    else:
        glossary = "None"

    chunks = chunk_text_semantically(text, max_tokens=200)
    translated_chunks = []
    current_context = None

    translation_metadata = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        updated_context = analyze_and_update_context(chunk, current_context)
        print("CONTEXT")
        print(updated_context)
        formatted_context = format_context_for_translation(updated_context)

        messages = [
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": f"""
Context:
{formatted_context}

Glossary:
{glossary}

Translate the following text to {target_language}. Preserve the original formatting:

{chunk}
"""}
        ]

        completion = inference.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=messages,
            max_tokens=512,
        )

        response = completion.choices[0].message.content or ""

        translated_chunks.append(response)

        translation_metadata.append({
            "chunk_index": i,
            "original_text": chunk,
            "translated_text": response.strip(),
            "context_used": updated_context
        })

        current_context = updated_context

    full_translation = "\n".join(translated_chunks)

    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out:
        out.write(full_translation)

    metadata_file = output_file.replace(".txt", "_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as meta_out:
        json.dump(translation_metadata, meta_out, indent=2, ensure_ascii=False)

    return {
        "translation_path": output_file,
        "metadata_path": metadata_file,
        "chunks_processed": len(chunks),
        "final_context": current_context
    }


def regular_translation(input_file, output_file, target_language, glossary=None):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text_semantically(text, max_tokens=200)
    translated_chunks = []

    glossary_strings = ""
    if glossary:
        for entity in glossary:
            glossary_strings += f"{entity["name"]}[{entity["type"]}, {entity["gender"]}] => {entity["translation"]}\n"

    for chunk in chunks:
        messages = [
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": f"""
        {f'Glossary:\n{glossary_strings}\n' if glossary else ''}
        The glossary contains the word and its correct translation. If the text contains this word, then translate according to the glossary.
Translate the following text to {target_language}. Preserve the original formatting. The answer should only contain a translation:

        {chunk}
        """}
        ]

        completion = inference.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=messages,
            max_tokens=1024,
        )

        response = completion.choices[0].message.content or ""

        translated_chunks.append(response)

    full_translation = "\n".join(translated_chunks)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        out.write(full_translation)


parser = argparse.ArgumentParser()
parser.add_argument("input_path")
parser.add_argument("output_path")
parser.add_argument("target_lang")
parser.add_argument("model")
parser.add_argument("--glossary", default=None)
parser.add_argument("--progressive", action="store_true")
args = parser.parse_args()
print(args)
try:
    if args.progressive:
        result = progressive_context_translation(
            args.input_path,
            args.output_path,
            args.target_lang,
            args.glossary,
        )
        if result and "metadata_path" in result and os.path.exists(result["metadata_path"]):
            print("Progressive context distillation metadata saved.",
                  file=sys.stderr)
        else:
            print("No metadata produced!", file=sys.stderr)
    else:
        glossary = None
        if args.glossary:
            with open(args.glossary, encoding="utf-8") as f:
                glossary = json.load(f)
        regular_translation(
            args.input_path,
            args.output_path,
            args.target_lang,
            glossary,
        )
    print("DONE", file=sys.stderr)
except Exception as ex:
    with open("llm_translation_error.log", "w", encoding="utf-8") as errlog:
        errlog.write(str(ex))
    print(f"ERROR: {ex}", file=sys.stderr)
    sys.exit(1)
