import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from collections import Counter


DATA_FILE_PATH = '/home/daniel/heMoji/data/data_3G_data01.pkl'
DATA_TYPE = 'data01'

"""
THAT SCRIPT IS VALID ONLY WHEN THE DATASET'S LABELS ARE THE EXACT LABELS OF THE DATA_TYPE
"""

# arg parsing #
if len(sys.argv) == 3:
    DATA_FILE_PATH = sys.argv[1]
    DATA_TYPE = sys.argv[2]
e2l_str = DATA_TYPE + "e2l"
l2e_str = "l2e" + DATA_TYPE
exec "from src.emoji2label import %s as e2l" % e2l_str
exec "from src.emoji2label import %s as l2e" % l2e_str
EMOJIS_NUM = len(l2e)


# load data labels #
with open(DATA_FILE_PATH, 'rb') as f:
    data = pickle.load(f)
Y = data['Y']  # list of labels [1, 2, 1, 5, ...]

# plot histogram
plt.hist(Y, bins=np.arange(EMOJIS_NUM+1)-0.5, facecolor='g', rwidth=0.95)
plt.xlabel('Emoji idx')
plt.ylabel('Count')
plt.title('Emojis Counts')
plt.xticks(range(EMOJIS_NUM))
# set emojis label xticks
x_labels = [(str(i)+'\n'+l2e[i]) for i in range(EMOJIS_NUM)]
ax = plt.axes()
ax.set_xticklabels(x_labels)
plt.show()
# file_name is like the data file name with other suffix
file_name = DATA_FILE_PATH.split('.')[0] + '_emojis_count.png'
fig = plt.gcf()
fig.set_size_inches((18, 8), forward=False)
plt.savefig(file_name)
print('Figure saved successfully')

# make frequency file
d = Counter(Y)
common_pairs = d.most_common()
file_name = DATA_FILE_PATH.split('.')[0] + '_emojis_freq.txt'
with open(file_name, 'w') as f:
    for c in common_pairs:
        emoji_index = c[0]
        emoji_freq = c[1]
        emoji_unicode = l2e[emoji_index]
        line = str(emoji_index) + '\t' + str(emoji_unicode.encode('utf-8')) + '\t' + str(emoji_freq)
        f.writelines(line)
        f.writelines('\n')
print('freq.txt saved successfully')


