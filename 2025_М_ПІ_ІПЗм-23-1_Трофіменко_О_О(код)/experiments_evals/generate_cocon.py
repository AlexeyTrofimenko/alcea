# generate-cocon.py

import json
import random


def generate_cocon_data(n=5, out_file="synthetic_cocon_data.json"):
    characters = [
        ("Nero", "Неро", "чоловік воїн"),
        ("Cora", "Кора", "жінка чародійка"),
        ("Arthus", "Артус", "чоловік лицар"),
        ("Lyanna", "Ліана", "жінка вчена")
    ]

    locs = [
        ("Ivory Tower", ["Башня Іворі", "Башня Слонової Кістки"]),
        ("Summer Vale", ["Літня долина", "Долина Літа"])
    ]

    actions = [
        "draws a sword",
        "lights a fire",
        "searches for supplies",
        "sets up a small camp",
        "scouts the perimeter",
        "examines the surroundings",
        "collects medicinal herbs"
    ]

    data = []

    for i in range(n):
        num_chars = random.choice([2, 3])
        chosen_chars = random.sample(characters, num_chars)

        loc_choice = random.choice(locs)
        loc_translation = random.choice(loc_choice[1])

        char_names_english = [c[0] for c in chosen_chars]
        text_intro = f"{' and '.join(char_names_english)} arrived at the {loc_choice[0]}."

        actions_lines = []
        for c in chosen_chars:
            action = random.choice(actions)
            actions_lines.append(f"{c[0]} {action}.")

        text_body = " ".join(actions_lines)
        text_full = text_intro + " " + text_body

        char_contexts = []
        for c in chosen_chars:
            char_contexts.append(
                f"Character: {c[0]} => {c[1]}, Extra: {c[2]}"
            )

        loc_context = f"Location: {loc_choice[0]} => {loc_translation}"

        data.append({
            "id": i + 1,
            "text": text_full,
            "ref": "",
            "context": {
                "characters": " | ".join(char_contexts),
                "locations": loc_context
            }
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    generate_cocon_data()
