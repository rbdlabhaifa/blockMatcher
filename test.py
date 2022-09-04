import os
from Dictionary.dictionary import MVMapping
from block_matching import BlockMatching


path = '/home/rani/PycharmProjects/blockMatcher/'

bm = BlockMatching.extract_motion_data(path + 'Extra Code/extract motion data/motionVectors', path + 'Dictionary/data/optitrack/2.mp4')
d = {}
# TODO: rerun this correctly for optitrack (use .csv files obviously)
for frame in range(len(bm)):
    if frame % 2 == 1:
        continue
    d[0.1 * (frame / 2 + 1)] = bm[frame]

for dictionary in os.listdir(path + 'Dictionary/saved dictionaries/'):
    if dictionary == 'old dictionaries':
        continue
    if not os.path.exists(path + 'Dictionary/comparisons/' + dictionary):
        os.mkdir(path + 'Dictionary/comparisons/' + dictionary)
    dic = MVMapping(path + f'Dictionary/saved dictionaries/{dictionary}')
    dic.compare(d, path + f'Dictionary/comparisons/{dictionary}/compared with optitrack_2.csv')
