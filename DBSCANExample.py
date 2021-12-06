from sklearn.datasets import make_blobs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

from matplotlib.patches import Circle  # $matplotlib/patches.py


def circle(xy, radius, color="lightsteelblue", facecolor="none", alpha=1, ax=None ):
    """ add a circle to ax= or current axes
    """
    # from .../pylab_examples/ellipse_demo.py
    e = Circle( xy=xy, radius=radius )
    if ax is None:
        ax = plt.gca()  # ax = subplot( 1,1,1 )
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_edgecolor( color )
    e.set_facecolor( facecolor )  # "none" not None
    e.set_alpha( alpha )

centers = [[.5, -1.25], [-1.25, 1.25], [1.5, .75]]
stds = [0.1, 0.4, 0.5]

X, labels_true = make_blobs(n_samples=30, centers=centers, cluster_std=stds, random_state=0)
fig = plt.figure()
fig.set_size_inches(12, 12)
ax = fig.add_subplot(1, 1, 1)

EPS = 1.0
db = DBSCAN(eps=EPS, min_samples=1).fit(X)
labels = db.labels_
print(labels)
sns.scatterplot(X[:,0], X[:,1], hue=["cluster-{}".format(x) for x in labels])

for i in range(len(X)):
    circle(X[i], EPS/2, ax=ax)

plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.title("DBSCAN with eps={}".format(EPS), fontsize=10)
fig.tight_layout()

plt.show()