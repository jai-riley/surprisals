import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.transforms as mtransforms

all = pd.read_csv("/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/all_data_cleaned.csv")
ngram_word = all['n_gram_word_surp']
ngram_word_pos = all['n_gram_word_POS_surp']
ngram_pos = all['n_gram_POS_surp']
bow_ngram_word = all['n_gram_BOW_word_surp']
bow_ngram_word_pos = all['n_gram_BOW_word_POS_surp']
bow_ngram_pos = all['n_gram_BOW_POS_surp']
PCFG_total = all['PCFG_total_surp']
PCFG_lex = all['PCFG_lex_surp']
PCFG_syn = all["PCFG_syn_surp"]
PCFG_pos = all["PCFG_pos_surp"]
rnng_word = all['RNNG_word_surp']
trans_word = all['transformer_word_surp']
rnng_pos = all['RNNG_pos_surp']
trans_pos = all['transformer_pos_surp']
trans_gram = all['TG_word_surp']

# First data group
# data1 = pd.DataFrame({
#     "value": pd.concat([ngram_word, ngram_word_pos,bow_ngram_word, bow_ngram_word_pos, PCFG_total, PCFG_lex, rnng_word, trans_word,trans_gram ], ignore_index=True),
#     "group": ["N-Gram Word"] * len(ngram_word) +
#              ["N-Gram Word/POS"] * len(ngram_word_pos) +
#              ["BOW N-Gram Word"] * len(bow_ngram_word) +
#              ["BOW N-Gram Word/POS"] * len(bow_ngram_word_pos) +
#              ["PCFG Total"] * len(PCFG_total) +
#              ["PCFG Lexical"] * len(PCFG_lex) +
#              ["RNNG Word"] * len(rnng_word) +
#              ["Transformer Word"] * len(trans_word)+
#              ["Transformer Grammar"] * len(trans_gram)

# })
data1 = pd.DataFrame({
    "value": pd.concat([ngram_pos, bow_ngram_pos, PCFG_syn, PCFG_pos, rnng_pos, trans_pos], ignore_index=True),
    "group": ["N-Gram POS"] * len(ngram_pos) +
             ["BOW N-Gram POS"] * len(bow_ngram_pos) +
             ["PCFG Syntactic"] * len(PCFG_syn) +
             ["PCFG POS"] * len(PCFG_pos) +
             ["RNNG POS"] * len(rnng_pos) +
             ["Transformer POS"] * len(trans_pos)

})
data1["value"] = pd.to_numeric(data1["value"], errors="coerce")
data1.dropna(inplace=True)

# Colour palette
colours1 = {
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
# group_order = [
#     "N-Gram Word", "N-Gram Word/POS", "BOW N-Gram Word", "BOW N-Gram Word/POS",
#     "PCFG Total", "PCFG Lexical", "RNNG Word", "Transformer Word", "Transformer Grammar"
# ]
group_order = [
    "N-Gram POS",  "BOW N-Gram POS",
    "PCFG Syntactic", "PCFG POS", "RNNG POS", "Transformer POS"
]

# Single axis
fig, ax1 = plt.subplots(figsize=(8, 6))

# Violin plot
vp1 = sns.violinplot(
    data=data1,
    x="group",
    y="value",
    inner=None,
    palette=colours1,
    bw=0.2,
    cut=0,
    gridsize=100,
    order=group_order,
    ax=ax1
)

for i, group in enumerate(group_order):
    violin = vp1.collections[i]
    violin.set_facecolor(colours1[group])
    violin.set_edgecolor('black')
    violin.set_alpha(1.0)
    violin.set_linewidth(0.75)

# Boxplot overlay
sns.boxplot(
    data=data1,
    x="group",
    y="value",
    width=0.2,
    showcaps=False,
    boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1},
    whiskerprops={'color': 'black', 'linewidth': 1},
    capprops={'color': 'black', 'linewidth': 1},
    medianprops={'color': 'black', 'linewidth': 2},
    showfliers=False,
    order=group_order,
    ax=ax1
)

# Axis formatting
ax1.set_xlabel("")
ax1.set_ylabel("Surprisal", size=18, weight='medium')
ax1.tick_params(axis='x', pad=1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=315, ha='left', size=14)

for label in ax1.get_xticklabels():
    offset = mtransforms.ScaledTranslation(-8/72, 0, fig.dpi_scale_trans)
    label.set_transform(label.get_transform() + offset)

# Remove spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

sns.set_theme(style="white")
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig("plots/violin_syntactic.svg", bbox_inches='tight', format="svg")
