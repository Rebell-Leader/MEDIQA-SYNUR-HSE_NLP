import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import csv
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Repository root (one level above paper_artifacts/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "paper_artifacts")

def generate_figure_2():
    print("Generating Figure 2: Precision-Recall Grouped Bar Chart...")

    # Data from Table 2 in results_sections.md
    data = [
        {"Model": "GPT-4o", "Metric": "Precision", "Score": 0.8131},
        {"Model": "GPT-4o", "Metric": "Recall", "Score": 0.6685},
        {"Model": "GPT-4o", "Metric": "F1", "Score": 0.7338},

        {"Model": "Qwen-235B (0-shot)", "Metric": "Precision", "Score": 0.7513},
        {"Model": "Qwen-235B (0-shot)", "Metric": "Recall", "Score": 0.7538},
        {"Model": "Qwen-235B (0-shot)", "Metric": "F1", "Score": 0.7525},

        {"Model": "GPT-5 (0-shot)", "Metric": "Precision", "Score": 0.7695},
        {"Model": "GPT-5 (0-shot)", "Metric": "Recall", "Score": 0.7743},
        {"Model": "GPT-5 (0-shot)", "Metric": "F1", "Score": 0.7719},

        {"Model": "Qwen-235B (RAG)", "Metric": "Precision", "Score": 0.7897},
        {"Model": "Qwen-235B (RAG)", "Metric": "Recall", "Score": 0.7975},
        {"Model": "Qwen-235B (RAG)", "Metric": "F1", "Score": 0.7936},

        {"Model": "ACE (Ours)", "Metric": "Precision", "Score": 0.7812},
        {"Model": "ACE (Ours)", "Metric": "Recall", "Score": 0.8188},
        {"Model": "ACE (Ours)", "Metric": "F1", "Score": 0.7996},
    ]

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))

    # Custom color palette
    # distinct colors for P, R, F1
    colors = ["#3498db", "#e74c3c", "#2ecc71"] # Blue, Red, Green

    g = sns.barplot(
        data=df,
        x="Model",
        y="Score",
        hue="Metric",
        palette=colors,
        edgecolor="black",
        linewidth=0.5
    )

    # Add value labels
    for container in g.containers:
        g.bar_label(container, fmt="%.3f", padding=3, rotation=90, fontsize=10)

    plt.ylim(0.5, 0.9) # Zoom in on the relevant range
    plt.title("Model Performance Comparison (Test Set)", fontsize=14, fontweight='bold')
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    output_path = os.path.join(ARTIFACTS_DIR, "figure_2_precision_recall.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")
    plt.close()

def generate_figure_3():
    print("Generating Figure 3: Impact of RAG (F1 Scores)...")

    # Data from analysis
    models = ['Qwen-235B', 'GPT-5-nano']
    baseline_scores = [0.7525, 0.5953] # 0-shot
    rag_scores = [0.7936, 0.4267] # RAG

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, baseline_scores, width, label='0-shot Baseline', color='#3498db')
    rects2 = ax.bar(x + width/2, rag_scores, width, label='RAG Enhanced', color='#e74c3c')

    ax.set_ylabel('F1 Score')
    ax.set_title('Impact of RAG on Different Model Sizes')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)

    # Add labels
    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')

    # Add delta annotation
    for i in range(len(models)):
        delta = rag_scores[i] - baseline_scores[i]
        label = f"{'+' if delta > 0 else ''}{delta*100:.1f}%"
        color = 'green' if delta > 0 else 'red'
        # Position above the higher bar
        height = max(baseline_scores[i], rag_scores[i])
        ax.annotate(label,
                    xy=(x[i], height),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color=color, fontweight='bold', fontsize=12)

    plt.tight_layout()
    output_path = os.path.join(ARTIFACTS_DIR, "figure_3_f1_scores.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")
    plt.close()

def load_schema_map():
    schema_path = os.path.join(BASE_DIR, "synur_schema.json")
    id_map = {}
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
            for item in schema:
                id_map[str(item['id'])] = item['name']
    except Exception as e:
        print(f"Error loading schema: {e}")
    return id_map

def generate_figure_4():
    print("Generating Figure 4: Top-5 Error Heatmap...")

    csv_path = os.path.join(ARTIFACTS_DIR, "error_analysis_granular.csv")
    if not os.path.exists(csv_path):
        print(f"Error file not found: {csv_path}")
        return

    id_map = load_schema_map()

    # Aggregation
    error_counts = {}

    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            clean_lines = (line.replace('\0', '') for line in f)
            reader = csv.DictReader(clean_lines)

            for row in reader:
                if not row: continue
                oid = row.get('obs_id', 'Unknown')
                etype = row.get('type', 'Unknown')

                # Simplify types
                if 'Value' in etype: etype = 'Value Error'
                if etype not in ['FP', 'FN', 'Value Error']: continue

                if oid not in error_counts:
                    error_counts[oid] = {'FP': 0, 'FN': 0, 'Value Error': 0, 'Total': 0}

                error_counts[oid][etype] += 1
                error_counts[oid]['Total'] += 1

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Sort and take Top 5
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1]['Total'], reverse=True)[:5]

    # Prepare DataFrame
    plot_data = []

    # Name shortener helper
    def shorten_name(name):
        name = name.lower()
        if "gastrointestinal" in name: return "GI Symptoms"
        if "urinary" in name and "symptom" in name: return "Urinary Sx"
        if "respiratory" in name and "intervention" in name: return "Resp. Interv."
        if "appearance" in name and "urine" in name: return "Urine Appr."
        if "cognitive" in name or "disturbances" in name: return "Cognitive Dist."
        return name.title()[:20]

    for oid, stats in sorted_errors:
        # Try both '3' and '03'
        clean_id = str(int(oid)) if oid.isdigit() else oid
        raw_name = id_map.get(clean_id, id_map.get(oid, f"ID {oid}"))
        short_name = shorten_name(raw_name)

        plot_data.append({
            "Entity": f"{short_name} (#{clean_id})",
            "False Positives": stats['FP'],
            "False Negatives": stats['FN'],
            "Value Errors": stats['Value Error']
        })

    df = pd.DataFrame(plot_data).set_index("Entity")

    # Plot Heatmap
    plt.figure(figsize=(8, 5))

    sns.heatmap(
        df,
        annot=True,
        fmt="d",
        cmap="Reds",
        linewidths=.5,
        cbar_kws={'label': 'Error Count'}
    )

    plt.title("Top 5 Most Problematic Entities by Error Type", fontsize=14, fontweight='bold')
    plt.ylabel("")
    plt.xlabel("Error Type")
    plt.tight_layout()

    output_path = os.path.join(ARTIFACTS_DIR, "figure_4_error_heatmap.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")
    plt.close()

if __name__ == "__main__":
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
