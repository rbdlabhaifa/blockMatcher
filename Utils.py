def get_macro_block(x, y, image, macroblock_size):
    """
    :param x: x coordinate of top left of macro block
    :param y: y coordinate of top left of macro block
    :param image: ndarray of image in BGR format\
    :param macroblock_size: length of side of macroblock
    :return: slice of array
    """

    return image[y:min(macroblock_size + y, image.shape[0]), x:min(macroblock_size + x, image.shape[1])]
`    
    pass
