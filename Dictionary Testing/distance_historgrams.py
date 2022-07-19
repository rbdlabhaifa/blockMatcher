import matplotlib.pyplot as plt
from python_block_matching import BlockMatching
import cv2
from mv_dictionary import MVMapping
from data_generator import DataGenerator

f1, f2 = DataGenerator.generate_translation([360, 360], "synthetic data/Triangle_full.png", [40, 0])
mv_dict = MVMapping("trained dicts/square")

mvs = BlockMatching.get_motion_vectors(f2, f1)
distances = mv_dict.get_min_distances(mvs)
print(distances)
n_bins = 100

fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs.hist(distances, bins=n_bins)

plt.show()
