from mv_dictionary import MVMapping
from python_block_matching import BlockMatching, BMFrame, BMVideo
from data_generator import DataGenerator
import cv2
import numpy as np


if __name__ == '__main__':
    mv_dict = MVMapping('trained dicts/circle')
    mv_dict.try_dictionary('synthetic data/Black.png')
