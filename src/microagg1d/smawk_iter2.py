import numpy as np
from numba import njit


@njit
def calc_len(start, stop, step_size):
    return (stop - start) // step_size + 1


@njit()
def interpolate(row_start, row_step, row_len, cols, calculator, result):
    curr_col = 0
    for i in range(2, row_len, 2):
        prev_row = row_start - 1 + row_step * (i - 1)
        start_col = result[prev_row]
        if i + 1 < row_len:
            # choose the next rows column as stop column
            next_row = row_start - 1 + row_step * (i + 1)
            stop_col = result[next_row]
        else:
            # the last column is the stop column
            stop_col = cols[len(cols) - 1]
        while cols[curr_col] < start_col:
            curr_col += 1
        best = curr_col
        curr_row = row_start + row_step * i
        best_val = calculator.calc(cols[best], curr_row)
        while cols[curr_col] < stop_col:
            tmp = calculator.calc(cols[curr_col + 1], curr_row)
            if best_val > tmp:
                best = curr_col + 1
                best_val = tmp

            curr_col += 1
        result[curr_row - 1] = cols[best]


@njit()
def reduce_iter(row_start, row_step, row_len, cols, calculator, col_buffer):
    # https://courses.engr.illinois.edu/cs473/sp2016/notes/06-sparsedynprog.pdf
    m = row_len
    S = col_buffer
    S[0] = cols[0]
    r = 0
    for curr_col in cols:
        if curr_col != S[r]:
            while r >= 0:
                if calculator.calc(S[r], row_start + row_step * r) > calculator.calc(
                    curr_col, row_start + row_step * r
                ):
                    r -= 1
                else:
                    break
        if r < m - 1:
            r += 1
            S[r] = curr_col
    return r + 1


@njit
def calc_max_col_space(n_rows, n_cols):
    max_cols = n_cols
    depth = 0
    col_starts = np.empty(2 * n_cols + 2, dtype=np.int64)
    col_starts[0] = 0
    col_starts[1] = n_cols
    while True:
        step_size = 2**depth
        val = min(n_rows // step_size, n_cols)
        max_cols += val
        depth += 1
        col_starts[depth + 1] = max_cols
        if val == 0:
            break
    return col_starts[: depth + 2], depth


@njit()
def _smawk_iter(row_start, row_stop, col_start, col_stop, calculator, result):
    # print("cols_in", col_start, col_stop)
    # print("rows", row_start, row_stop)
    if row_start - row_stop == 0 or col_stop - col_start == 0:
        return
    col_starts, _ = calc_max_col_space(
        calc_len(row_start + 1, row_stop, 1), col_stop - col_start
    )
    # col_buffer= np.empty(col_starts[-1], dtype=cols_in.dtype)
    col_buffer = np.empty(col_starts[-1], dtype=np.int64)

    # print(col_starts)
    # print(col_buffer)
    # print(max_depth)
    # print("max_depth", max_depth)
    __smawk_iter(
        row_start,
        row_stop,
        col_start,
        col_stop,
        calculator,
        result,
        col_starts,
        col_buffer,
    )


@njit
def __smawk_iter(
    row_start, row_stop, col_start, col_stop, calculator, result, col_starts, col_buffer
):
    if row_start - row_stop == 0 or col_stop - col_start == 0:
        return
    for i in range(col_stop - col_start):
        col_buffer[i] = col_start + i
    col_starts[0] = 0
    col_starts[1] = col_stop - col_start  # num columns
    depth = 0
    while True:
        step_size = 2**depth
        # rows = rows_in[step_size-1::step_size]
        curr_start = row_start + step_size
        cols = col_buffer[col_starts[depth] : col_starts[depth + 1]]
        # print(cols, row_stop)
        row_len = calc_len(curr_start, row_stop, step_size)
        # print(depth, rows, cols)

        if row_len == 0:
            break
        if len(cols) == 1:
            for r in range(curr_start - 1, row_stop, step_size):
                result[r] = cols[0]
            break
        S = col_buffer[col_starts[depth + 1] :]  # col_starts[depth+2]]
        # print(S)
        col_starts[depth + 2] = col_starts[depth + 1] + reduce_iter(
            curr_start, step_size, row_len, cols, calculator, S
        )

        # _cols = col_buffer[col_starts[depth+1]:col_ends[depth+1]]
        # print(_cols)
        # print()
        result[curr_start - 1] = col_buffer[col_starts[depth + 1]]
        depth += 1
    max_depth = depth

    for depth in range(max_depth, -1, -1):
        step_size = 2**depth
        curr_start = row_start + step_size
        row_len = calc_len(curr_start, row_stop, step_size)

        # print(rows)
        if row_len == 0:
            continue
        _cols = col_buffer[col_starts[depth + 1] : col_starts[depth + 2]]
        # print(cols, "\n")
        if row_len > 2:
            interpolate(curr_start, step_size, row_len, _cols, calculator, result)
    # print(result)
