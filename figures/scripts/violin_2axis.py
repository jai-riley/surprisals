import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.transforms as mtransforms

all = pd.read_csv("/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/all_data_cleaned.csv")
ngram_word = all['n_gram_word_surp']
ngram_word_pos = all['n_gram_word_POS_surp']
bow_ngram_word = all['n_gram_word_surp']
ngram_word_pos = all['n_gram_word_POS_surp']
PCFG_total = all['PCFG_total_surp']
PCFG_lex = all['PCFG_lex_surp']
rnng = all['RNNG_word_surp']
trans = all['transformer_word_surp']

ngram_pos = all['n_gram_POS_surp']
PCFG_syn = all['PCFG_syn_surp']
PCFG_pos = all['PCFG_pos_surp']

# First data group
data1 = pd.DataFrame({
    "value": pd.concat([ngram_word, ngram_word_pos, PCFG_total, PCFG_lex, rnng, trans], ignore_index=True),
    "group": ["N-Gram Word"] * len(ngram_word) +
             ["N-Gram Word/POS"] * len(ngram_word_pos) +
             ["PCFG Total"] * len(PCFG_total) +
             ["PCFG Lexical"] * len(PCFG_lex) +
             ["RNNG Word"] * len(rnng) +
             ["Transformer Word"] * len(trans)
})
data1["value"] = pd.to_numeric(data1["value"], errors="coerce")
data1.dropna(inplace=True)

# Second data group (new axis)
data2 = pd.DataFrame({
    "value": pd.concat([ngram_pos, PCFG_syn, PCFG_pos], ignore_index=True),
    "group": (["N-Gram POS"] * len(ngram_pos) +
              ["PCFG Syntactic"] * len(PCFG_syn) +
              ["PCFG POS"] * len(PCFG_pos))
})
data2["value"] = pd.to_numeric(data2["value"], errors="coerce")
data2.dropna(inplace=True)

# colours for both groups

colours1 = {
    "N-Gram Word": "#348ABD",
    "N-Gram Word/POS": "#FDBF11",
    "PCFG Total": "#E24A33",
    "PCFG Lexical": "#7E2F8E",
    "RNNG Word": "#33BBCC",
    "Transformer Word": "#E586B6"
}
colours2 = {
    "N-Gram POS": "#188487",
    "PCFG Syntactic": "#FC7E5E",
    "PCFG POS": "#AF7F3D"
}

# Create figure with 2 horizontally stacked axes
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(12, 6),
    sharey=True,
    gridspec_kw={'width_ratios': [2, 1]}  # ax1 is 3x wider than ax2
)

# First violin + boxplot on ax1
vp1 = sns.violinplot(
    data=data1,
    x="group",
    y="value",
    inner=None,
    palette=colours1,
    bw=0.2,
    cut=0,
    gridsize=100,
    ax=ax1
)

for i, artist in enumerate(vp1.collections):
    color_key = list(colours1.values())[i]  # each violin gets 2 entries
    artist.set_facecolor(color_key)
    artist.set_edgecolor('black')
    artist.set_alpha(1.0)
    artist.set_linewidth(.75)  


sns.boxplot(
    data=data1,
    x="group",
    y="value",
    width=0.2,
    showcaps=False,
    boxprops={'facecolor':'none', 'edgecolor':'black', 'linewidth':1},
    whiskerprops={'color':'black', 'linewidth':1},
    capprops={'color':'black', 'linewidth':1},
    medianprops={'color':'black', 'linewidth':2},
    showfliers=False,
    ax=ax1
)
# ax1.set_title("Word-based Models",size=18,weight='medium')
ax1.set_xlabel("")
ax1.set_ylabel("Surprisal",size=18,weight='medium')
ax1.tick_params(axis='x', pad=1)  # increase padding to 10 points (default is usually 3-5)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=315,ha='left',size=14)
for label in ax1.get_xticklabels():
    # get current position in display coords
    label_pos = label.get_position()
    # create an offset transform that moves the label to the left by a few pixels
    # negative x-offset shifts label left so first letter touches tick
    offset = mtransforms.ScaledTranslation(-8/72, 0, fig.dpi_scale_trans)  # 5 points left
    label.set_transform(label.get_transform() + offset)

# Second violin + boxplot on ax2
vp2 = sns.violinplot(
    data=data2,
    x="group",
    y="value",
    inner=None,
    palette=colours2,
    bw=0.2,
    cut=0,
    gridsize=100,
    ax=ax2
)

for i, artist in enumerate(vp2.collections):
    color_key = list(colours2.values())[i]  # each violin gets 2 entries
    artist.set_facecolor(color_key)
    artist.set_edgecolor('black')
    artist.set_alpha(1.0)
    artist.set_linewidth(0.75)  


sns.boxplot(
    data=data2,
    x="group",
    y="value",
    width=0.2,
    showcaps=False,
    boxprops={'facecolor':'none', 'edgecolor':'black', 'linewidth':1},
    whiskerprops={'color':'black', 'linewidth':1},
    capprops={'color':'black', 'linewidth':1},
    medianprops={'color':'black', 'linewidth':2},
    showfliers=False,
    ax=ax2
)
# ax2.set_title("POS-based Models",size=18,weight='medium')
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.yaxis.set_tick_params(labelleft=True)
ax2.tick_params(axis='x', pad=1)  # increase padding to 10 points (default is usually 3-5)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=315,ha='left',size=14)
for label in ax2.get_xticklabels():
    # get current position in display coords
    label_pos = label.get_position()
    # create an offset transform that moves the label to the left by a few pixels
    # negative x-offset shifts label left so first letter touches tick
    offset = mtransforms.ScaledTranslation(-8/72, 0, fig.dpi_scale_trans)  # 5 points left
    label.set_transform(label.get_transform() + offset)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
sns.set_theme(style="white")

plt.ylim(bottom=0)
plt.tight_layout(w_pad=4)
plt.savefig(f"plots/violin.svg",bbox_inches='tight',format="svg")

