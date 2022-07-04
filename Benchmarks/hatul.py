def intra_frame_mb_partition(frame: np.ndarray, threshold, max_size=16, min_size=8):
    half_size = max_size // 2
    if half_size < min_size:
        return
    for tlx, tly in get_macro_blocks(frame, max_size):
        mb = slice_macro_block(frame, tlx, tly, max_size)
        v1, v2 = mb[:half_size, :], mb[half_size:, :]
        h1, h2 = mb[:, :half_size], mb[:, half_size:]
        vertical_cost = sse(v1, v2)
        horizontal_cost = sse(h1, h2)
        if vertical_cost > threshold and horizontal_cost > threshold:
            if half_size == min_size:
                yield tlx, tly, half_size, half_size
                yield tlx + half_size, tly, half_size, half_size
                yield tlx, tly + half_size, half_size, half_size
                yield tlx + half_size, tly + half_size, half_size, half_size
            else:
                if half_size // 2 >= min_size:
                    mb1 = slice_macro_block(frame, tlx, tly, half_size)
                    mb2 = slice_macro_block(frame, tlx + half_size, tly, half_size)
                    mb3 = slice_macro_block(frame, tlx, tly + half_size, half_size)
                    mb4 = slice_macro_block(frame, tlx + half_size, tly + half_size, half_size)
                    for qmb in (mb1, mb2, mb3, mb4):
                        for x, y, w, h in intra_frame_mb_partition(qmb, threshold, half_size, min_size):
                            yield tlx + x, tly + y, w, h
        elif vertical_cost > threshold:
            yield tlx, tly, max_size, half_size
            yield tlx, tly + half_size, max_size, half_size
        elif horizontal_cost > threshold:
            yield tlx, tly, half_size, max_size
            yield tlx + half_size, tly, half_size, max_size
        else:
            yield tlx, tly, max_size, max_size
