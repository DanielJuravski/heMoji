import numpy as np
import pickle
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from src.emoji2label import e2l, l2e

DATA_FILE_PATH = '/home/daniel/heMoji/data/data.pkl'
EMOJIS_NUM = 64

if len(sys.argv) == 2:
    DATA_FILE_PATH = sys.argv[1]

with open(DATA_FILE_PATH, 'rb') as f:
    data = pickle.load(f)
Y = data['Y']

plt.hist(Y, bins=np.arange(EMOJIS_NUM)-0.5, facecolor='g')
plt.xlabel('Emoji idx')
plt.ylabel('Count')
plt.title('Emojis Counts')
plt.xticks(range(EMOJIS_NUM))


x_labels = [(str(i)+'\n'+l2e[i]) for i in range(EMOJIS_NUM)]
ax = plt.axes()
ax.set_xticklabels(x_labels)

# plt.show()
# file_name is like the data file name with other suffix
file_name = DATA_FILE_PATH.split('.')[0] + '_emojis_count.png'
fig = plt.gcf()
fig.set_size_inches((18, 8), forward=False)
plt.savefig(file_name, dpi=1200)

