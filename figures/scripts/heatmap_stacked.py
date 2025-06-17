import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.transforms as mtransforms
import os

d = {"E":"English","C":"Chinese","S":"Spanish","K":"Korean"}

languages = ["E", "S", "C", "K"]
order = ["base","ngram_word",'ngram_pos',"ngram_word_pos","pcfg_total","pcfg_lex","pcfg_syn","pcfg_pos","rnng_word","trans_word"]
names = ["None (Base Model)", "N-Gram Word", "N-Gram POS", "N-Gram Word/POS","PCFG Total", "PCFG Lexical","PCFG Syntactic","PCFG POS","RNNG Word","Transformer Word"]
order2 = ["s(Participant)",
        "s(Trial)", 
        "s(Word)",
        "s(VocabAdjPerf)",
        "s(ReadingCompAdjPerf)",
        "SentPos",
        "s(WordPos)",
        "s(WordLogFreq)",
        "s(WordLength)",
        "ti(WordLogFreq,WordLength)",
        "s(Surprisal)",
        "SentPos1:Surprisal",
        "SentPos2:Surprisal",
        "ti(Surprisal,WordLength)",
        "ti(Surprisal,WordLogFreq)",
        "ti(Surprisal,WordPos)",
        "ti(Surprisal,Prev Surprisal)",
        "ti(Surprisal,Prev WordLogFreq)",
        ]

fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(5, 10), constrained_layout=True)

cmap = ListedColormap(["lavenderblush","#e5ecf3", "#1c2436"])  # N/A (-1), Not Significant (0), Significant (1)

for i, language in enumerate(languages):
    language_files = {}
    directory = f'/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/model_summaries/{language}'
    files = os.listdir(directory)

    for file in files:
        language_files[file[:-4]] = {k: -1 for k in order2}
        at_point = False
        with open(f"{directory}/{file}", 'r') as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line == "Parametric coefficients:\n":
                    at_point = True
                    line = f.readline()
                    line = f.readline()
                    line = f.readline()
                elif line == "Approximate significance of smooth terms:\n":
                    at_point = True
                    line = f.readline()
                    line = f.readline()

                if '---' in line and at_point:
                    at_point = False
                if at_point:
                    line = line.split()
                    print(line)
                    key = ""
                    if "SentPos1" in line[0]:
                        key = "SentPos1:Surprisal"
                    elif "SentPos2:" in line[0]:
                        key = "SentPos2:Surprisal"
                    elif "SentPos2" in line[0]:
                        key = "SentPos"
                    elif "surp)" in line[0]:
                        key = "s(Surprisal)"
                    elif "p,WordLength)" in line[0]:
                        key = "ti(Surprisal,WordLength)"
                    elif ",LogFreq)" in line[0]:
                        key = "ti(Surprisal,WordLogFreq)"
                    elif ",WordPos)" in line[0]:
                        key = "ti(Surprisal,WordPos)"
                    elif line[0] == "s(Subject)":
                        key = "s(Participant)"
                    elif line[0] == "s(procWordID)":
                        key = "s(Word)"
                    elif line[0] == "s(Vocab_Competence.Acc)":
                        key = "s(VocabAdjPerf)"
                    elif line[0] == "s(Comp_Competence.Acc)":
                        key = "s(ReadingCompAdjPerf)"
                    elif line[0] ==  "s(LogFreq)":
                        key = "s(WordLogFreq)"
                    elif line[0] == "ti(LogFreq,WordLength)":
                        key = "ti(WordLogFreq,WordLength)"
                    elif "pPrev1)" in line[0]:
                        key ="ti(Surprisal,Prev Surprisal)"
                    elif "qPrev1)" in line[0]:
                        key = "ti(Surprisal,Prev WordLogFreq)"
                    else:
                        key = line[0]

                    if "*" in line[-1]:
                        language_files[file[:-4]][key] = 1
                    else:
                        language_files[file[:-4]][key] = 0

    data = []
    for o in order:
        l = []
        for o2 in order2:
            l.append(language_files[o][o2])
        data.append(l)
    data = np.array(data)

    ax = axs[i]

    im = ax.imshow(data, cmap=cmap, vmin=-1.5, vmax=1.5)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    if i == 3:  # Only bottom subplot shows x labels to avoid clutter
        ax.set_xticklabels(
        order2,
        rotation=300,
        ha="left",
        fontsize=10
        )  
        for label in ax.get_xticklabels():
            offset = mtransforms.ScaledTranslation(-8/72, 3/74, fig.dpi_scale_trans)
            label.set_transform(label.get_transform() + offset) 
        ax.set_xlabel("Terms", fontsize=12, fontweight='medium')
 
    else:
        ax.set_xticklabels([])  # Hide xticklabels for upper plots

    ax.set_ylabel("Surprisal Type", fontsize=12, fontweight='medium')
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title(d[language], fontsize=14)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('medium')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('medium')

    ax.tick_params(top=False, bottom=True, left=True, right=False)

    # Optional: Adjust x tick label positions if needed




# Create one legend for the whole figure
legend_labels = {
    "Significant": "#1c2436",
    "Not Significant": "#e5ecf3",
    "N/A": "lavenderblush"

}
legend_patches = [
    mpatches.Patch(facecolor=color, edgecolor='black', linewidth=1.0, label=label)
    for label, color in legend_labels.items()
]
fig.legend(handles=legend_patches, 
           loc='lower center', 
           ncol=3, 
           frameon=False, 
           fontsize=8,    
           handlelength=1.5,
            handleheight=2,
            bbox_to_anchor=(0.6, -0.03))

# plt.suptitle("Significance Heatmaps by Language", fontsize=18)
plt.savefig("plots/heatmaps/V2/Partial/stacked_heatmaps.svg", bbox_inches='tight')
# plt.show()
