import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
#--------MADE USING AI SOLELY-------
def generate_paper_visuals(results_path):
    print("Loading data into Pandas DataFrame...")
    data = []
    # Read the JSONL file line by line
    with open(results_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Convert to DataFrame for seaborn plotting
    df = pd.DataFrame([
        {
            'H': d['metrics']['Hurst_Exp'], 
            'S': d['metrics']['Holder_Exp'], 
            'Label': d.get('Label', 'LLM'),
            'temp': d.get('temp', None)
        } for d in data
    ])
    
    print(f"Successfully loaded {len(df)} rows. Generating plots...")

    # --- FIGURE 10 REPLICATION: Distribution ---
    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")

    # Hurst Exponent (LRD) Plot
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=df, x='H', hue='Label', fill=True, common_norm=False, palette="crest")
    # Human range reference from the paper
    plt.axvspan(0.6, 0.7, color='gray', alpha=0.3, label='Human Range')
    plt.title("Long-Range Dependence (Hurst) Distribution", fontsize=14)
    plt.xlabel("Hurst Exponent (H)")

    # Hölder Exponent (S) Plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df, x='S', hue='Label', fill=True, common_norm=False, palette="flare")
    # Human range reference from the paper
    plt.axvspan(0.4, 0.5, color='gray', alpha=0.3, label='Human Range')
    plt.title("Self-Similarity (S) Distribution", fontsize=14)
    plt.xlabel("Hölder Exponent (S)")

    plt.tight_layout()
    plt.savefig("distribution_comparison.png", dpi=300)
    print("🎨 Distribution plot saved as 'distribution_comparison.png'")

    # --- JOINT PLOT: The "Fingerprint" ---
    # This shows how H and S correlate, creating a cluster for Human vs LLM
    g = sns.jointplot(data=df, x='H', y='S', hue='Label', kind="kde", palette="viridis")
    g.figure.suptitle("Fractal Parameter Clustering", y=1.02)
    plt.savefig("fractal_cluster.png", dpi=300)
    print("🎨 Cluster plot saved as 'fractal_cluster.png'")
def plot_temperature_sensitivity(df):
    """Replicates the paper's analysis on how Temperature affects Fractal metrics."""
    plt.figure(figsize=(10, 6))
    
    # Filter for LLM only since Humans don't have a 'temperature'
    llm_df = df[df['Label'] == 'LLM']
    
    sns.lineplot(data=llm_df, x='temp', y='H', marker='o', label='LLM Hurst (H)')
    
    # Add the 'Human Golden Range' as a reference band
    plt.axhspan(0.62, 0.68, color='green', alpha=0.1, label='Human Range (Typical)')
    
    plt.title("Impact of Temperature on Long-Range Dependence")
    plt.xlabel("Decoding Temperature")
    plt.ylabel("Hurst Exponent")
    plt.legend()
    plt.savefig("temp_sensitivity.png")

def plot_semantic_fractal_correlation(df):
    """Plots the relationship between meaning (Semantic) and structure (Fractal)."""
    if 'Semantic_Consistency' not in df.columns:
        print("Skipping: Semantic metrics not found in data.")
        return

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x='H', y='Semantic_Consistency', hue='Label', style='Label', alpha=0.6)
    plt.title("Semantic Consistency vs. Structural Complexity")
    plt.savefig("semantic_fractal_corr.png")
if __name__ == "__main__":
    generate_paper_visuals('analysis_results.jsonl')
