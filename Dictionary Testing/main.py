from mv_dictionary import MVMapping
from python_block_matching import BlockMatching, BMFrame, BMVideo
from data_generator import DataGenerator
import cv2
import numpy as np


if __name__ == '__main__':
    mv_dict = MVMapping()
    mv_dict.train('synthetic data/Black.png')
    mv_dict.save_to('trained dicts/square')
    mv_dict = MVMapping()
    mv_dict.train('synthetic data/Circle_full.png')
    mv_dict.save_to('trained dicts/circle')
    mv_dict = MVMapping()
    mv_dict.train('synthetic data/Triangle_full.png')
    mv_dict.save_to('trained dicts/triangle')
