import difflib
import json
import os
import re
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sacrebleu
import seaborn as sns
from matplotlib.colors import Normalize
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(lines):
    processed = []
    for line in lines:
        line = line.strip()
        line = re.sub(r'[\"“”«»‘’„”\(\)\[\]\{\}<>«»—–\-…]', '', line)
        line = re.sub(r'\s+', ' ', line)
        line = re.sub(r'([,.!?;:])', r' \1 ', line)
        line = re.sub(r'\s{2,}', ' ', line)
        line = line.lower().strip()
        if line:
            processed.append(line)
    return processed


def calculate_rouge(refs, preds):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(
        r, p)['rougeL'].fmeasure for r, p in zip(refs, preds)]
    return round(sum(rouge_scores) / len(rouge_scores), 3) * 100


def read_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read().splitlines()


model = SentenceTransformer(
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def semantic_similarity(refs, preds):
    embeddings_refs = model.encode(refs, convert_to_tensor=True)
    embeddings_preds = model.encode(preds, convert_to_tensor=True)
    cos_sim = cosine_similarity(embeddings_refs, embeddings_preds)
    return round(np.mean(np.diagonal(cos_sim)), 3) * 100


def calculate_chrf(refs, preds):
    return round(sacrebleu.corpus_chrf(preds, [refs]).score, 2)


languages = {'en_fr': 'FR', 'en_ge': 'DE', 'en_ua': 'UKR'}
methods = ['google', 'human', 'default',
           'glossary', 'context', 'context+glossary']
experiment_types = {'1': 'narrative', '2': 'dialogue'}

data_path = Path('Experiments')
results = []

for lang_folder, lang_code in languages.items():
    for exp_num, exp_type in experiment_types.items():
        base_path = data_path / lang_folder / exp_num
        references = preprocess(read_file(base_path / 'human.txt'))

        for method in methods:
            translation = preprocess(read_file(base_path / f'{method}.txt'))

            bleu = sacrebleu.corpus_bleu(translation, [references]).score
            rouge = calculate_rouge(references, translation)
            semantic_sim = semantic_similarity(references, translation)
            chrf = calculate_chrf(references, translation)

            results.append({
                'Language': lang_code,
                'Experiment': exp_type,
                'Method': method,
                'BLEU': round(bleu, 2),
                'ROUGE-L': rouge,
                'Semantic_Similarity': semantic_sim,
                'CHRF++': chrf
            })

df = pd.DataFrame(results)
print(df)
df.to_csv('Experiments/results/all_metrics_detailed.csv', index=False)


df = pd.read_csv('Experiments/results/all_metrics_detailed.csv')


metrics = ['BLEU', 'ROUGE-L', 'Semantic_Similarity', 'CHRF++']
methods_to_compare = ['glossary', 'context', 'context+glossary']

filtered_df = df[
    (df['Experiment'] == 'narrative') &
    (df['Method'].isin(['default'] + methods_to_compare))
]

heatmap_data = {}

for lang in filtered_df['Language'].unique():
    lang_df = filtered_df[filtered_df['Language'] == lang]
    default_row = lang_df[lang_df['Method'] == 'default'].iloc[0]

    for method in methods_to_compare:
        method_row = lang_df[lang_df['Method'] == method].iloc[0]
        improvements = []
        for metric in metrics:
            base = default_row[metric]
            val = method_row[metric]
            perc_improvement = ((val - base) / base) * 100 if base != 0 else 0
            improvements.append(round(perc_improvement, 2))

        heatmap_data[(method, lang)] = improvements

improvement_df = pd.DataFrame.from_dict(
    heatmap_data, orient='index', columns=metrics
)

improvement_df.index = pd.MultiIndex.from_tuples(
    improvement_df.index, names=['Method', 'Language'])

avg_improvement = improvement_df.groupby('Method').mean()

plt.figure(figsize=(8, 5))
sns.heatmap(avg_improvement, annot=True, fmt=".2f",
            cmap="coolwarm", cbar_kws={'label': 'Avg % Improvement'})
plt.title('Average % Improvement over Default (Narrative)')
plt.ylabel('Method')
plt.xlabel('Metric')
plt.tight_layout()
plt.show()


def normalize(text):
    return re.sub(r'[^\w\s]', '', text.lower())


def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def load_glossary(glossary_path):
    with open(glossary_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def extract_variants_from_text(text, term, threshold=0.8):
    tokens = re.findall(r'\b\w+\b', text.lower())
    base_tokens = normalize(term).split()
    base_length = len(base_tokens)
    variants = []
    count = 0
    i = 0
    while i <= len(tokens) - base_length:
        candidate = " ".join(tokens[i:i+base_length])
        ratio = difflib.SequenceMatcher(
            None, candidate, normalize(term)).ratio()
        if ratio >= threshold:
            variants.append(candidate)
            count += 1
            i += base_length
        else:
            i += 1
    return set(variants), count, base_length


def count_matches_against_variants(text, variants, base_length, threshold=0.8):
    tokens = re.findall(r'\b\w+\b', text.lower())
    count = 0
    i = 0
    while i <= len(tokens) - base_length:
        candidate = " ".join(tokens[i:i+base_length])
        for variant in variants:
            ratio = difflib.SequenceMatcher(None, candidate, variant).ratio()
            if ratio >= threshold:
                count += 1
                i += base_length
                break
        else:
            i += 1
    return count


def compute_f1_metric(human_text, machine_text, glossary, threshold=0.8):
    """
    For each glossary term, computes fuzzy-matching based precision, recall, and F1.
    Fuzzy matching is used to identify possible matches (accounting for morphological variations).
    The overall CoCon metric is the average F1 across all terms.
    Returns overall metrics and detailed term-level results.
    """
    term_pairs = {}
    for category in glossary:
        if isinstance(glossary[category], dict):
            for term, value in glossary[category].items():
                if isinstance(value, dict):
                    term_pairs[term] = value.get("translation", "")
                else:
                    term_pairs[term] = value

    total_f1 = 0
    total_precision = 0
    total_recall = 0
    total_delta = 0
    term_count = 0
    term_details = {}

    for source_term, target_translation in term_pairs.items():
        human_variants, human_count, base_length = extract_variants_from_text(
            human_text, target_translation, threshold)
        if human_count == 0:
            continue
        machine_count = count_matches_against_variants(
            machine_text, human_variants, base_length, threshold)
        true_positive = min(human_count, machine_count)
        precision = true_positive / machine_count if machine_count > 0 else 0
        recall = true_positive / human_count if human_count > 0 else 0
        f1 = (2 * precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0
        delta = (machine_count - human_count) / human_count

        term_details[source_term] = {
            'Glossary Translation': target_translation,
            'Human Count': human_count,
            'Machine Count': machine_count,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Delta': delta,
            'Human Variants': list(human_variants)
        }
        total_f1 += f1
        total_precision += precision
        total_recall += recall
        total_delta += delta
        term_count += 1

    if term_count > 0:
        overall_metrics = {
            'Num Terms': term_count,
            'Avg Precision': total_precision / term_count,
            'Avg Recall': total_recall / term_count,
            'Avg F1': total_f1 / term_count,
            'Avg Delta': total_delta / term_count
        }
    else:
        overall_metrics = {'Num Terms': 0, 'Avg Precision': 0,
                           'Avg Recall': 0, 'Avg F1': 0, 'Avg Delta': 0}
    return overall_metrics, term_details


def main():
    # Contains sample1_en.txt and sample2_en.txt
    source_text_dir = "data/texts"
    # Structure: /Experiments/en_{lang}/{text_part}/{method}.txt
    experiments_dir = "Experiments"
    # Structure: fantasy_terms_{lang}_{text_part}.json
    glossary_dir = "data/glossary"

    languages = ['fr', 'ge', 'ua']           # French, German, Ukrainian
    methods = ['default', 'glossary',
               'context+glossary', 'context', 'google', 'human']
    text_parts = {'1': 'Narrative', '2': 'Dialogue'}
    detailed_method = 'glossary'

    results = {part: {} for part in text_parts}
    table1_data = []
    table2_data = []

    for part, part_label in text_parts.items():
        source_text_path = os.path.join(
            source_text_dir, f"sample{part}_en.txt")
        if not os.path.exists(source_text_path):
            print(
                f"Source text file {source_text_path} not found. Skipping {part_label}.")
            continue

        results[part] = {}
        for lang in languages:
            results[part][lang] = {}
            glossary_path = os.path.join(
                glossary_dir, f"fantasy_terms_{lang}_{part}.json")
            if not os.path.exists(glossary_path):
                print(
                    f"Glossary file {glossary_path} not found. Skipping language {lang.upper()} for {part_label}.")
                continue
            glossary = load_glossary(glossary_path)

            human_path = os.path.join(
                experiments_dir, f"en_{lang}", part, "human.txt")
            if not os.path.exists(human_path):
                print(
                    f"Human translation file {human_path} not found. Skipping language {lang.upper()} for {part_label}.")
                continue
            human_text = load_text(human_path)

            for method in methods:
                method_path = os.path.join(
                    experiments_dir, f"en_{lang}", part, f"{method}.txt")
                if not os.path.exists(method_path):
                    print(
                        f"Translation file {method_path} not found. Skipping method {method} for {lang.upper()} ({part_label}).")
                    continue
                machine_text = load_text(method_path)
                overall_metrics, term_details = compute_f1_metric(
                    human_text, machine_text, glossary, threshold=0.8)
                results[part][lang][method] = overall_metrics

                table1_data.append({
                    'Language': lang.upper(),
                    'Text Part': part_label,
                    'Method': method,
                    'Num Terms': overall_metrics['Num Terms'],
                    'Avg Precision': round(overall_metrics['Avg Precision'], 3),
                    'Avg Recall': round(overall_metrics['Avg Recall'], 3),
                    'Avg F1': round(overall_metrics['Avg F1'], 3)
                })

                if lang == 'ua' and part == '1' and method == detailed_method:
                    for term, detail in term_details.items():
                        table2_data.append({
                            'Term': term,
                            'Glossary Translation': detail['Glossary Translation'],
                            'Human Count': detail['Human Count'],
                            'Machine Count': detail['Machine Count'],
                            'Precision': round(detail['Precision'], 3),
                            'Recall': round(detail['Recall'], 3),
                            'F1': round(detail['F1'], 3)
                        })

    for part, part_label in text_parts.items():
        available_langs = []
        method_scores_by_lang = {method: [] for method in methods}
        for lang in languages:
            if lang not in results[part]:
                continue
            available_langs.append(lang.upper())
            for method in methods:
                score = results[part][lang].get(method, {}).get('Avg F1', 0)
                method_scores_by_lang[method].append(score)
        if not available_langs:
            continue

        x = np.arange(len(available_langs))
        width = 0.12
        offsets = np.linspace(-width*(len(methods)-1)/2,
                              width*(len(methods)-1)/2, len(methods))

        plt.figure(figsize=(10, 6))
        for i, method in enumerate(methods):
            scores = method_scores_by_lang[method]
            plt.bar(x + offsets[i], scores, width, label=method)
        plt.xlabel("Language")
        plt.ylabel("Average F1 Score")
        plt.title(f"{part_label} Translation F1 Metrics by Language and Method")
        plt.xticks(x, available_langs)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.tight_layout()
        plt.show()

    scatter_part = '1'  # Narrative
    scatter_lang = 'fr'
    scatter_method = 'default'
    glossary_path = os.path.join(
        glossary_dir, f"fantasy_terms_{scatter_lang}_{scatter_part}.json")
    human_path = os.path.join(
        experiments_dir, f"en_{scatter_lang}", scatter_part, "human.txt")
    machine_path = os.path.join(
        experiments_dir, f"en_{scatter_lang}", scatter_part, f"{scatter_method}.txt")
    if os.path.exists(glossary_path) and os.path.exists(human_path) and os.path.exists(machine_path):
        glossary = load_glossary(glossary_path)
        human_text = load_text(human_path)
        machine_text = load_text(machine_path)

        x_vals, y_vals, labels = [], [], []
        for category in glossary:
            if isinstance(glossary[category], dict):
                for term, value in glossary[category].items():
                    target_translation = value.get(
                        "translation", "") if isinstance(value, dict) else value
                    human_variants, human_count, base_length = extract_variants_from_text(
                        human_text, target_translation, threshold=0.8)
                    machine_count = count_matches_against_variants(
                        machine_text, human_variants, base_length, threshold=0.8)
                    if human_count > 0:
                        x_vals.append(human_count)
                        y_vals.append(machine_count)
                        labels.append(term)

        plt.figure(figsize=(6, 6))
        plt.scatter(x_vals, y_vals, alpha=0.7)
        max_val = max(x_vals + y_vals) if (x_vals + y_vals) else 1
        plt.plot([0, max_val], [0, max_val], 'r--', label="Ideal (y=x)")
        plt.xlabel("Human Count")
        plt.ylabel(f"{scatter_method.capitalize()} Machine Count")
        plt.title(
            f"Scatter Plot for {scatter_lang.upper()} ({text_parts[scatter_part]}) - {scatter_method} method")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Required files for scatter plot are missing.")

    heatmap_methods = ['glossary', 'context+glossary', 'human']
    languages_heat = []
    improvement_data = []
    part_key = '1'
    for lang in languages:
        if lang in results[part_key] and 'default' in results[part_key][lang]:
            default_f1 = results[part_key][lang]['default'].get('Avg F1', None)
            if default_f1 is None or default_f1 == 0:
                continue
            row = []
            for m in heatmap_methods:
                if m in results[part_key][lang]:
                    m_f1 = results[part_key][lang][m].get('Avg F1', 0)
                    improvement = (m_f1 - default_f1) / default_f1 * 100
                    row.append(improvement)
                else:
                    row.append(np.nan)
            improvement_data.append(row)
            languages_heat.append(lang.upper())
    improvement_data = np.array(improvement_data)
    improvement_data = improvement_data.T

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(improvement_data, cmap='viridis')
    ax.set_xticks(np.arange(len(languages_heat)))
    ax.set_xticklabels(languages_heat)
    ax.set_yticks(np.arange(len(heatmap_methods)))
    ax.set_yticklabels([m.capitalize() for m in heatmap_methods])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(improvement_data.shape[0]):
        for j in range(improvement_data.shape[1]):
            ax.text(
                j, i, f"{improvement_data[i, j]:.1f}%", ha="center", va="center", color="w")
    ax.set_title("Improvement over Default (%) on CoCon Metric (Narrative)")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    heatmap_methods2 = ['google', 'glossary',
                        'context+glossary', 'human', 'default']
    languages_heat2 = []
    values_data = []
    for lang in languages:
        if lang in results[part_key]:
            row = []
            for m in heatmap_methods2:
                if m in results[part_key][lang]:
                    row.append(results[part_key][lang]
                               [m].get('Avg F1', np.nan))
                else:
                    row.append(np.nan)
            values_data.append(row)
            languages_heat2.append(lang.upper())
    values_data = np.array(values_data)  # shape: (#languages, #methods)
    values_data = values_data.T  # now shape: (#methods, #languages)

    fig, ax = plt.subplots(figsize=(8, 4))
    im2 = ax.imshow(values_data, cmap='plasma', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(languages_heat2)))
    ax.set_xticklabels(languages_heat2)
    ax.set_yticks(np.arange(len(heatmap_methods2)))
    ax.set_yticklabels([m.capitalize() for m in heatmap_methods2])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(values_data.shape[0]):
        for j in range(values_data.shape[1]):
            ax.text(j, i, f"{values_data[i, j]:.2f}",
                    ha="center", va="center", color="w")
    ax.set_title("CoCon Metric Values (Avg F1) for Narrative Translations")
    fig.colorbar(im2, ax=ax)
    plt.tight_layout()
    plt.show()

    table1_df = pd.DataFrame(table1_data)

    filtered_methods = ['default', 'glossary', 'context+glossary', 'context']
    table1_df = table1_df[table1_df['Method'].isin(filtered_methods)]

    table1_df = table1_df.drop(columns=['Text Part', 'Num Terms'])

    summary_df = table1_df.groupby(
        ['Language', 'Method'], as_index=False).mean()

    summary_df = summary_df[['Language', 'Method',
                            'Avg Precision', 'Avg Recall', 'Avg F1']]

    output_path = "Experiments/results/summary_avg_metrics.csv"
    summary_df.to_csv(output_path, index=False)

    print("Table 1: Summary of Metrics for All Languages and Methods")
    print(table1_df.to_string(index=False))

    table2_df = pd.DataFrame(table2_data)
    print("\nTable 2: Detailed Term-Level Metrics for Ukrainian Narrative (Glossary Method)")
    print(table2_df.to_string(index=False))


if __name__ == "__main__":
    main()
