import json
import sys

from huggingface_hub import InferenceClient

input_path, glossary_path, model, output_path = sys.argv[1:]

with open(input_path, encoding="utf-8") as f:
    text = f.read()
with open(glossary_path, encoding="utf-8") as f:
    glossary = json.load(f)


inference = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct", provider="together")


messages = [
    {"role": "system", "content": "You are a translation assistant. Given a text and a glossary, return ONLY a JSON array listing the most important translation entities: names, locations, items. Each should be a dict with fields: name, type, translation, gender."},
    {"role": "user", "content": f"""
Text:
{text}
Glossary:
{json.dumps(glossary, ensure_ascii=False, indent=2)}
Return a JSON array of entities.
"""}
]
completion = inference.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=800
)
response = completion.choices[0].message.content or ""
try:
    json_start = response.find('[')
    json_end = response.rfind(']')
    if json_start != -1 and json_end != -1:
        response_json = response[json_start:json_end + 1]
        entity_list = json.loads(response_json)
    else:
        entity_list = json.loads(response)
except Exception:
    entity_list = []

with open(output_path, "w", encoding="utf-8") as out:
    json.dump(entity_list, out, ensure_ascii=False, indent=2)
