import numpy as np


def compute_wer(outputs, targets, decoder, alphabet, ignore_case=False):
    outputs = outputs.transpose(0, 1).to("cpu")
    outputs = decoder(outputs)

    outputs = alphabet.int_to_text(outputs.tolist())
    targets = alphabet.int_to_text(targets.tolist())

    if ignore_case is True:
        targets = targets.lower()
        outputs = outputs.lower()

    targets = [t.split(alphabet.delimiter) for t in targets]
    outputs = [o.split(alphabet.delimiter) for o in outputs]

    wers = [levenshtein_distance(t, o) for t, o in zip(targets, outputs)]
    wers = sum(wers)
    n = sum(len(t) for t in targets)
    return wers, n


def levenshtein_distance(ref, hyp):
    m = len(ref)
    n = len(hyp)

    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # Initialize distance matrix
    for j in range(0, n + 1):
        distance[0][j] = j

    # Computation
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]
