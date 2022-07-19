from mv_dictionary import MVMapping


if __name__ == '__main__':
    mv_dict = MVMapping('trained dicts/circle')
    mv_dict.try_dictionary('synthetic data/Circle_full.png')
