import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import os

# Load the CSV
df = pd.read_csv("AIC_base_model_base_all_V2.csv")

# Define pretty color mapping
colours_dict = {
    "N-Gram Word": "#348ABD",
    "N-Gram POS": "#188487",
    "N-Gram Word/POS": "#FDBF11",
    "PCFG Total": "#E24A33",
    "PCFG Syntactic": "#FC7E5E",
    "PCFG Lexical": "#7E2F8E",
    "PCFG POS": "#AF7F3D",
    "RNNG Word": "#33BBCC",
    "Transformer Word": "#E586B6",
    "BOW N-Gram Word": "#66A61E",  # dark red
    "BOW N-Gram POS"       : "#5D8AA8",  # steel blue
    "BOW N-Gram Word/POS"  : "#A60628",  # olive green
    "RNNG POS"       : "#D95F02",  # burnt orange
    "Transformer POS"        : "#7570B3",  # lavender blue
    "Transformer Word/POS"   : "#1B9E77",  # green-teal
    "Transformer Grammar"  : "#545454"  # bright yellow
}


lang_names = {
    "E": "English",
    "C": "Chinese",
    "K": "Korean",
    "S": "Spanish"
}

# Ensure output directory exists
output_dir = "plots/bar_graphs/V2/All"
os.makedirs(output_dir, exist_ok=True)

# Plot per language
languages = ["E", "C", "K", "S"]
# languages = ["K"]

for lang in languages:
    sub = df[df["language"] == lang].copy()
    sub = sub.sort_values("delta_AIC", ascending=False).reset_index(drop=True)
    sub = sub.head(7)

    # print(sub.iloc[:,1])
    # sub = sub[sub.iloc[:, 1] != "Transformer Grammar Surprisal"]
    # fig, ax = plt.subplots(figsize=(6, 4))
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
  # Adjust height if needed
    x_positions = [i * 1.1 for i in range(len(sub))]

    ax.bar(
        x=x_positions,
        height=sub["delta_AIC"],
        color=[colours_dict.get(surp, "#999999") for surp in sub["surprisal"]],
        width=1
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [k.replace(" Surprisal", "") for k in sub["surprisal"]],
        rotation=315,
        ha="left",
        fontsize=12
    )

  

    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_title(lang_names.get(lang, lang), fontsize=20, y=0.90, weight='medium')
    ax.set_ylabel("ΔAIC", fontsize=16, weight='medium')
    ax.tick_params(axis='x', pad=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    for label in ax.get_xticklabels():
        # get current position in display coords
        label_pos = label.get_position()
        # create an offset transform that moves the label to the left by a few pixels
        # negative x-offset shifts label left so first letter touches tick
        offset = mtransforms.ScaledTranslation(-8/72, 0, fig.dpi_scale_trans)  # 5 points left
        label.set_transform(label.get_transform() + offset)
    # Shared xlabel per plot
    fig.text(0.5, -0.01, 'Surprisal Type', ha='center', va='center', fontsize=16, weight='medium')

    # Save plot
    filename = os.path.join(output_dir, f"AIC_barplot_{lang_names.get(lang, lang)}_V2_top7.svg")
    # plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', format='svg', transparent=False)
    plt.close()
