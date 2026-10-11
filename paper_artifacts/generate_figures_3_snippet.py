
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
    ax.legend()
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
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color=color, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(ARTIFACTS_DIR, "figure_3_f1_scores.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")
    plt.close()
