# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Update this to the filename you want to load
# csv_file = "AIC_base_model_baseX.csv"

# # Read the CSV file
# df = pd.read_csv(csv_file)

# # Ensure numeric values are actually numeric
# df['delta_AIC'] = pd.to_numeric(df['delta_AIC'], errors='coerce')

# # Make an output directory for the plots
# output_dir = "plots/original"
# os.makedirs(output_dir, exist_ok=True)

# # Group by language and plot each one
# for lang, group in df.groupby("language"):
#     # Sort values from greatest to least
#     group_sorted = group.sort_values("delta_AIC", ascending=False)

#     # Plot
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(
#         group_sorted["surprisal"],
#         group_sorted["delta_AIC"],
#         color=group_sorted["colour"]
#     )

#     # Add title and labels
#     plt.title(f"Delta AIC for Language {lang}")
#     plt.ylabel("Delta AIC (Base - Model)")
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()

#     # Save plot
#     plot_path = os.path.join(output_dir, f"{lang}_delta_AIC_barplot_X.png")
#     plt.savefig(plot_path)
#     plt.close()

#     print(f"Saved plot for {lang} to {plot_path}")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


# Load the CSV
df = pd.read_csv("AIC_base_model_base_V2.csv")

# Define pretty color mapping
color_dict = {
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
# Plot per language
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(6, 15), sharex=False)
# fig.text(0.5, 0.04, 'Surprisal Type', ha='center', va='center', fontsize=12)

# Ensure consistent order of languages
languages = ["E", "C", "K", "S"]

for i, lang in enumerate(languages):
    sub = df[df["language"] == lang].copy()
    sub = sub.sort_values("delta_AIC", ascending=False)

    ax = axs[i]
    sub = sub.sort_values(by="delta_AIC", ascending=False).reset_index(drop=True)
    x_positions = [i * 1.1 for i in range(len(sub))]

    # Plot the bars with spaced positions
    ax.bar(
        x=x_positions,
        height=sub["delta_AIC"],
        color=[color_dict.get(surp, "#999999") for surp in sub["surprisal"]],
        width=1  # Or adjust as needed
    )

    # Set x-tick positions and labels to match
    # a_pos =  [i * 2.95 for i in range(len(sub))]

    ax.set_xticks(x_positions)
    ax.set_xticklabels([k.replace(" Surprisal", "") for k in sub["surprisal"] ], rotation=315, ha="left", fontsize=12)

    # Styling
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_title(lang_names.get(lang, lang), fontsize=20,y = 0.90,weight='medium')
    ax.set_ylabel("ΔAIC", fontsize=16,weight='medium')
    ax.tick_params(axis='x', pad=1)  # increase padding to 10 points (default is usually 3-5)

    # ax.set_xticks(range(len(sub)))
    # ax.set_xticklabels(sub["surprisal"], rotation=45, ha="right", fontsize=9)
    for label in ax.get_xticklabels():
        # get current position in display coords
        label_pos = label.get_position()
        # create an offset transform that moves the label to the left by a few pixels
        # negative x-offset shifts label left so first letter touches tick
        offset = mtransforms.ScaledTranslation(-8/72, 0, fig.dpi_scale_trans)  # 5 points left
        label.set_transform(label.get_transform() + offset)
    # ax.tick_params(axis='x', labelbottom=True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

# fig.subplots_adjust(bottom=0.25)  # tweak as needed   
fig.text(0.5,-0.01, 'Surprisal Type', ha='center', va='center', fontsize=16,weight='medium')
plt.tight_layout()
plt.savefig(f"plots/bar_graphs/V2/OG/AIC_barplot_stacked.svg",bbox_inches='tight',format="svg")
    # plt.show()