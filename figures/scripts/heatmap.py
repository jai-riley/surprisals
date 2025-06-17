import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.transforms as mtransforms
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
import os
for language in ["E", "S", "C", "K"]:
    language_files = {}
    directory = f'/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/model_summaries/{language}'
    files = os.listdir(directory)
    print(files)
    for file in files:
        print(file)
        language_files[file[:-4]] = {k: -1 for k in order2}
        at_point = False
        with open(f"{directory}/{file}", 'r') as f:
            
            line = f.readline()
            # line = repr(line.rstrip("\n"))
            while line:
                line = f.readline()
                # line = line.rstrip("\n")

                # print(line)
                if line == "Parametric coefficients:\n": 
                    at_point = True
                    line = f.readline()
                    line = f.readline()
                    line = f.readline()
                if line == "Approximate significance of smooth terms:\n":
                    at_point = True
                    line = f.readline()
                    line = f.readline()

                if '---' in line and at_point:
                    at_point = False

                if at_point:
                    line = line.split()
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

        f.close()
        print(language_files)
                

                    # line = repr(line.rstrip("\n"))
                # print(line)
                # print("hi inside loop")  # You'll now see this multiple times if the file has content

                    
            # print('yes')
    # Example 5x5 heatmap values
    data = []
    
    for o in order:
        l = []
        for o2 in order2:
            print(o2)
            l.append(language_files[o][o2])
        data.append(l)


    data = np.array(data)
    print(data)
    # Define colors for each value: N/A (-1), Not Significant (0), Significant (1)
    cmap = ListedColormap(["lavenderblush","#e5ecf3", "#1c2436"])  # Order must match [-1, 0, 1]

    # Set boundaries between values
    bounds = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot using imshow with discrete colors
    cbar = ax.imshow(data, cmap=cmap, vmin=-1.5, vmax=1.5)

    # Remove cell labels
    ax.set_xticks(np.arange(data.shape[1]))
    print(data.shape[1],data.shape[0])
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(names)


    ax.set_xticklabels(
        order2,
        rotation=315,
        ha="left",
        fontsize=12
    )

    for label in ax.get_xticklabels():
        offset = mtransforms.ScaledTranslation(-8/72, 0, fig.dpi_scale_trans)
        label.set_transform(label.get_transform() + offset)
    # Add custom legend
    legend_labels = {
        "N/A": "lavenderblush",
        "Not Significant": "#e5ecf3",
        "Significant": "#1c2436"
    }

    legend_patches = [
        mpatches.Patch(
            facecolor=color,
            edgecolor='black',  # Add border
            linewidth=1.0,
            label=label
        ) for label, color in legend_labels.items()
    ]
    # Place legend below the heatmap in 3 columns
    legend = ax.legend(
        handles=legend_patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -1.1),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        handleheight=1.5
    )

    plt.title(d[language], fontsize=14)
    plt.tight_layout()
    plt.savefig(f"plots/heatmaps/{language}.svg",bbox_inches='tight',format="svg")
