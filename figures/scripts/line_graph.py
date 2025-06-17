import matplotlib.pyplot as plt
import math
# Perplexity values
# perplexities_1 = [72000.444,185.373528,144.579058,124.1262,111.591510, 101.8875,93.896296,87.718695,82.4451,77.929,73.801612,70.054443,66.9399,64.30057,61.717295,59.181058,57.2128554,55.132890, 53.318755]
# perplexities_2 = [73315.4999,227.3767,194.33888,179.546729,172.46626,166.202922,161.15574,159.4583,158.205343,155.196651,155.308439,155.011516,153.5277,153.356489,153.34343,154.517286,155.982270,154.073871,157.262394]
# perplexities_trans_word_1 = [11.056850,5.637766, 5.210139,4.999467,4.810527,4.695601,4.615421,4.554091,4.498509]
# perplexities_trans_word_2 = [11.056411,5.602124,5.337030,5.274257,5.205170,5.175300,5.187506,5.168190,5.182656]
# perplexities_trans_pos_1 = [4.539906, 1.988704, 1.963362, 1.936682, 1.929193, 1.919865,1.912779,1.907902,1.907270,1.904504,1.901246]
# perplexities_trans_pos_2 = [4.513649,1.974084,1.946312,1.924867,1.918178,1.909802,1.903337,1.898930,1.898149,1.895493,1.892643]
# perplexities_trans_word_pos_1 = [11.261, 5.671,5.244,4.969,4.821,4.704,4.629,4.548]
# perplexities_trans_word_pos_2 = [11.259,5.642,5.381,5.281,5.259,5.239,5.242,5.228]
# perplexities_1 = [11.261, 5.671,5.244,4.969,4.821,4.704,4.629,4.548]
# perplexities_2 = [11.259,5.642,5.381,5.281,5.259,5.239,5.242,5.228]
# perplexities_rnng_pos_1 = [47.310,2.581,2.511,2.4889,2.464,2.461,2.448,2.444,2.427,2.424,2.418]
# perplexities_rnng_pos_2 = [47.210,2.550,2.4889,2.466,2.450,2.450,2.442,2.443,2.430,2.430,2.426]
perplexities_1 = [6.9614363,2.2418158,1.9972852,1.7878889,1.5505807,1.5979264,1.5537995]
perplexities_2 = [6.9614363,2.316344086,2.1718919528,2.06047277142,2.003364550815,1.9718997572,1.9633097468]
loss_to_perplex = True

if loss_to_perplex:
    for x in range(len(perplexities_1)):
        perplexities_1[x] = 10**(perplexities_1[x])
        perplexities_2[x] = 10**(perplexities_2[x])

print(perplexities_1)
print(perplexities_2)
epochs = list(range(len(perplexities_1)))

# Create subplots with shared x-axis and broken y-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6), gridspec_kw={'height_ratios': [1, 3]})

# Plot on both axes
ax1.plot(epochs, perplexities_1, label="Training",  color='#1f77b4',linewidth=5)
ax1.plot(epochs, perplexities_2, label="Validation",  color='#ff7f0e',linewidth=5)
ax2.plot(epochs, perplexities_1,  color='#1f77b4',linewidth=5)
ax2.plot(epochs, perplexities_2, color='#ff7f0e',linewidth=5)

# Set y-limits to cut out the middle
ax1.set_ylim(9150000, 9151000)
ax2.set_ylim(30, 210)

# Hide spines and add diagonal lines
# ax1.tick_params(labeltop=False,labelbottom=False)
# # Only remove x-ticks and x-labels on top
ax1.tick_params(axis='x', which='both', length=0, labelbottom=False)
# ax1.tick_params(axis='y', which='both', direction='in', right=True)

  # Don't show top tick labels
ax2.xaxis.tick_bottom()

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_linewidth(1.5)

ax2.spines['left'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Diagonal lines
# kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.legend(loc="upper center",bbox_to_anchor=(0.5, 1.05),ncol=2,frameon=False,prop={'weight': 'medium', 'size': 14})
# ax1.plot((0, 0), (0, 0), **kwargs)        # Top-left
# ax1.plot((1, 1), (0,0), **kwargs)  # Top-right

# kwargs.update(transform=ax2.transAxes)
# ax2.plot((0, 0), (1, 1), **kwargs)  # Bottom-left
# ax2.plot((1, 1), (1, 1), **kwargs)  # Bottom-right

# Labels
# ax1.set_title("Word",fontsize=16,weight='medium')

ax2.set_xlabel("Epoch",fontsize=16,weight='medium')
# ax2.set_xticklabels([x for x in range(-2,21,2)])
ax2.set_xticks(range(0, 7,2))  # Ticks at every integer from 0 to 20

# Set x-limits with a bit of margin so the 0 tick isn’t at the edge
# ax2.set_xlim(-0.5, 8.5)

# ax1.set_y label("Perplexity")
# ax2.set_ylabel("Perplexity")
# fig.suptitle("RNNG Word Perplexity over Epochs")
fig.text(0.012, 0.5, 'Word Perplexity', va='center', rotation='vertical', fontsize=16,weight='medium')
fig.tight_layout()
plt.subplots_adjust(hspace=0.2,left=0.15)


plt.savefig(f"plots/TG_training.svg",bbox_inches='tight',format="svg")

