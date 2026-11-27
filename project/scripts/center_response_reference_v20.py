"""Independent CPU references for preparation; no model or production kernel imports."""
import math
import numpy as np
from PIL import Image


def _labels(value):
    array = np.asarray(value)
    if array.ndim != 2 or not all(array.shape) or array.dtype.kind not in 'iu':
        raise ValueError('Require a nonempty two-dimensional integer label array')
    if np.any(array < 0) or np.any(array > 88):
        raise ValueError('Labels must be background 0 or object IDs 1..88')
    return array.astype(np.int64, copy=True)


def target_centroids(labels):
    """Match old float32 coordinate-mean reduction and ties-to-even rounding."""
    labels = _labels(labels); width = labels.shape[1]; result = []
    for identity in sorted(set(int(value) for value in labels.flat)-{0}):
        locations = np.flatnonzero(labels.reshape(-1) == identity)
        rows, columns = locations // width, locations % width
        row_mean = np.add.reduce(rows.astype(np.float32), dtype=np.float32)/np.float32(len(rows))
        col_mean = np.add.reduce(columns.astype(np.float32), dtype=np.float32)/np.float32(len(columns))
        row, col = int(np.rint(row_mean)), int(np.rint(col_mean))
        result.append({'id': identity, 'row': row, 'col': col, 'pixels': int(len(rows)),
                       'centroid_inside': bool(labels[row, col] == identity)})
    return result


def independent_targets(labels):
    """Construct original sigma-3 targets with explicit float32 arithmetic."""
    labels = _labels(labels); height, width = labels.shape
    heatmap = np.zeros((height, width), np.float32)
    offsets = np.zeros((2, height, width), np.float32)
    row_axis = np.arange(height, dtype=np.float32)
    col_axis = np.arange(width, dtype=np.float32)
    for center in target_centroids(labels):
        row, col, identity = center['row'], center['col'], center['id']
        dy2 = np.square(row_axis-np.float32(row), dtype=np.float32)
        dx2 = np.square(col_axis-np.float32(col), dtype=np.float32)
        radial = np.add(dy2[:, None], dx2[None, :], dtype=np.float32)
        exponent = np.divide(-radial, np.float32(18), dtype=np.float32)
        gaussian = np.exp(exponent, dtype=np.float32)
        np.maximum(heatmap, gaussian, out=heatmap)
        rr, cc = np.nonzero(labels == identity)
        offsets[0, rr, cc] = np.divide(np.float32(col)-cc.astype(np.float32), np.float32(width), dtype=np.float32)
        offsets[1, rr, cc] = np.divide(np.float32(row)-rr.astype(np.float32), np.float32(height), dtype=np.float32)
    return {'labels': labels, 'foreground': (labels != 0).astype(np.float32)[None],
            'centers': heatmap[None], 'offsets': offsets}


def _size(value):
    if len(value) != 2 or any(type(n) is not int or n <= 0 for n in value):
        raise ValueError('Require two positive integer dimensions')
    return tuple(value)


def pillow_training_resize(labels, output_hw):
    """Pillow nearest for label supervision; distinct from evaluation mapping."""
    values = _labels(labels); height, width = _size(output_hw)
    # The original labels are uint16 PNGs; preserve that Pillow mode here.
    image = Image.fromarray(values.astype(np.uint16))
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST), dtype=np.int64)


def exact_pixel_center_indices(source_size, destination_size):
    if type(source_size) is not int or type(destination_size) is not int or min(source_size, destination_size) <= 0:
        raise ValueError('Require positive integer dimensions')
    # Python integers keep the definition exact, independent of floating point.
    return np.asarray([((2*i+1)*source_size)//(2*destination_size) for i in range(destination_size)], dtype=np.int64)


def evaluation_resize(array, output_hw):
    array = np.asarray(array); height, width = _size(output_hw)
    if array.ndim != 2 or not all(array.shape): raise ValueError('Require a nonempty 2D grid')
    rows = exact_pixel_center_indices(array.shape[0], height)
    cols = exact_pixel_center_indices(array.shape[1], width)
    return array[rows[:, None], cols[None, :]].copy()


def distance_regions(shape, centers):
    """Independent scalar-grid partition by nearest unique integer center."""
    height, width = _size(shape); unique = sorted(set(tuple(point) for point in centers))
    if any(len(point) != 2 or any(type(v) is not int for v in point)
           or not (0 <= point[0] < height and 0 <= point[1] < width) for point in unique):
        raise ValueError('Center is outside the integer grid')
    result = np.full((height, width), 2, dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            squared = min(((row-y)**2+(col-x)**2 for y, x in unique), default=math.inf)
            result[row, col] = 0 if squared <= 9 else 1 if squared <= 81 else 2
    return result


def linear_quantiles(values):
    values = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in values): raise ValueError('Nonfinite quantile sample')
    result = []
    for q in (0., .5, .9, .99, 1.):
        if not values: value = None
        else:
            location = (len(values)-1)*q; lower, upper = math.floor(location), math.ceil(location)
            value = values[lower]+(values[upper]-values[lower])*(location-lower)
        result.append({'q': q, 'value': value})
    return result


def exhaustive_matching(adjacency, prediction_count):
    """Enumerate small bipartite graphs; no greedy or production augmenting path."""
    if type(prediction_count) is not int or not 0 <= prediction_count <= 8 or len(adjacency) > 8:
        raise ValueError('Exhaustive preparation reference is limited to 8 by 8 vertices')
    rows = []
    for neighbors in adjacency:
        values = list(neighbors)
        if any(type(x) is not int or not 0 <= x < prediction_count for x in values) or len(set(values)) != len(values):
            raise ValueError('Invalid or duplicate matching adjacency')
        rows.append(sorted(values))
    best = []; maximizing = 0
    def visit(index, occupied, pairs):
        nonlocal best, maximizing
        if index == len(rows):
            if len(pairs) > len(best): best = list(pairs); maximizing = 1
            elif len(pairs) == len(best): maximizing += 1
            return
        visit(index+1, occupied, pairs)
        for prediction in rows[index]:
            if prediction not in occupied:
                visit(index+1, occupied | {prediction}, pairs+[(index, prediction)])
    visit(0, set(), [])
    return {'cardinality': len(best), 'one_maximum_matching': [list(pair) for pair in best],
            'maximum_matching_count': maximizing, 'global_minimum_distance_claimed': False}


def radius_adjacency(ground_truth, predictions, radius):
    if type(radius) not in (int, float) or not math.isfinite(radius) or radius < 0:
        raise ValueError('Invalid matching radius')
    left, right = [tuple(x) for x in ground_truth], [tuple(x) for x in predictions]
    if any(len(x) != 2 or any(not math.isfinite(float(v)) for v in x) for x in left+right):
        raise ValueError('Invalid point')
    return [[j for j, (y, x) in enumerate(right) if (row-y)**2+(col-x)**2 <= radius**2] for row, col in left]
