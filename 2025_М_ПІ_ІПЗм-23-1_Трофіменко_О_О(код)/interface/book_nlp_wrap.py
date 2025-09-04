import os
import sys
import time

from booknlp.booknlp import BookNLP

model_params = {
    "pipeline": "entity,quote,supersense,event,coref",
    "model": "small"
}


def main():
    if len(sys.argv) != 3:
        print("Usage: python book_nlp_wrap.py <input_file> <output_dir>",
              file=sys.stderr)
        sys.exit(1)
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    book_id = "novel"
    os.makedirs(output_dir, exist_ok=True)

    timings = {}
    start = time.time()
    timings['startup'] = 0

    t0 = time.time()
    booknlp = BookNLP("en", model_params)
    timings['startup'] = time.time() - t0

    t0 = time.time()
    booknlp.process(input_file, output_dir, book_id)
    timings['process'] = time.time() - t0
    timings['total'] = time.time() - start

    with open(input_file, encoding="utf-8") as f:
        text = f.read()
    word_count = len(text.split())

    print(f"--- startup: {timings['startup']:.3f} seconds ---")
    print(f"--- process: {timings['process']:.3f} seconds ---")
    print(f"--- TOTAL: {timings['total']:.3f} seconds ---")
    print(f"--- words: {word_count}")


if __name__ == "__main__":
    main()
