"""Fixed observation-only spatial correspondence features for v50.

No filesystem, ground truth, source-index, true-warp or training-label access.
The first sixteen channels spatially retain the frozen v43 observables.
"""
import numpy as np
from scipy.ndimage import (uniform_filter, median_filter, maximum_filter,
                           minimum_filter, gaussian_filter, distance_transform_edt)
from src import uncertainty_mixture_v43 as old

CANDIDATES = old.CANDIDATES.copy()
CANDIDATES.flags.writeable = False
FEATURE_NAMES = tuple(old.WARP_FEATURES) + (
    'instance_confidence', 'instance_soft_boundary', 'instance_boundary_proximity',
    'instance_boundary_depth_product', 'instance_proximity_depth_product',
    'instance_interior_depth_product', 'observed_valid_fraction3',
    'instance_interior_normal_variation', 'instance_interior_plane_residual',
    'instance_confidence_surface_valid', 'instance_confidence_depth_edge',
    'foreground_depth_valid')
MAP_NAMES = ('foreground_probability', 'instance_confidence', 'soft_boundary',
             'boundary_proximity')


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def fixed_partition(shape):
    """Old 2048 deterministic sites, assigned to fixed row-major 4x4 cells.

    Counts do not depend on a candidate, depth validity, or instance prediction.
    Weighted cell means reproduce the old sixteen global means, up to rounding.
    """
    require(len(shape) == 2 and min(shape) >= 24, 'Spatial features need at least 24x24')
    h, w = shape
    ids = old.fixed_feature_indices(shape)
    y, x = np.divmod(ids, w)
    region = ((y-8)*4//(h-16))*4 + ((x-8)*4//(w-16))
    counts = np.bincount(region, minlength=16)
    require(len(counts) == 16 and np.all(counts > 0), 'Empty fixed spatial cell')
    return ids, region, counts


def instance_maps(instances, foreground_probability, scores):
    """Derive soft maps solely from the deployed RGB decoder outputs.

    A hard predicted contour is a geometric support, never a correspondence vote.
    It is confidence weighted and smoothed with fixed sigma=1/radius=3. A soft
    foreground range adds boundary evidence when no instance center survives.
    Empty instances remain valid observations, with explicit zero confidence.
    """
    labels = np.asarray(instances)
    fg = np.asarray(foreground_probability, np.float64)
    require(labels.ndim == 2 and np.issubdtype(labels.dtype, np.integer)
            and labels.shape == fg.shape and np.all(labels >= 0), 'Invalid predicted instances')
    require(np.isfinite(fg).all() and np.all((fg >= 0) & (fg <= 1)), 'Invalid foreground probabilities')
    confidence = np.zeros(fg.shape, np.float64)
    supplied = set()
    for row in scores:
        ident, score = row['instance_id'], row['score']
        require(type(ident) is int and ident > 0 and ident not in supplied
                and np.isfinite(score) and 0 <= score <= 1, 'Invalid retained-center confidence')
        supplied.add(ident)
        confidence[labels == ident] = float(score)
    require(supplied == set(map(int, np.unique(labels[labels > 0]))), 'Instance confidence identity differs')
    confidence *= fg
    pad = np.pad(labels, 1, mode='constant')
    contour = (labels > 0) & ((labels != pad[:-2, 1:-1]) | (labels != pad[2:, 1:-1])
                               | (labels != pad[1:-1, :-2]) | (labels != pad[1:-1, 2:]))
    smooth = gaussian_filter(contour*confidence, sigma=1., mode='constant', truncate=3.)
    soft = np.clip(smooth + maximum_filter(fg, 3, mode='nearest')
                   - minimum_filter(fg, 3, mode='nearest'), 0., 1.)
    proximity = np.exp(-distance_transform_edt(~contour)/4.) if np.any(contour) else np.zeros_like(fg)
    return dict(foreground_probability=fg.astype(np.float32),
                instance_confidence=confidence.astype(np.float32),
                soft_boundary=soft.astype(np.float32), boundary_proximity=proximity.astype(np.float32))


def geometry_maps(depth, valid, intrinsic, original_size):
    """Observed surface conditions using exact nearest original source pixels.

    A camera ray is computed at its observed grid location; no hidden true warp
    or clean source depth is used. Central tangent normals and four-neighbor
    deviations require observed local validity. Missing terms are zero and the
    separate validity channel preserves their absence, with fixed denominators.
    """
    z, ok = old.observations(depth, valid)
    h, w = z.shape
    yy, xx = np.mgrid[:h, :w]
    uvz = np.stack((xx, yy, z), axis=-1)
    xyz = old.backproject_uvz(uvz, intrinsic, original_size, (w, h))
    valid3 = uniform_filter(ok.astype(np.float64), 3, mode='constant')
    dense = valid3 > .999
    normal = np.zeros((*z.shape, 3), np.float64)
    cross = np.cross(xyz[1:-1, 2:]-xyz[1:-1, :-2], xyz[2:, 1:-1]-xyz[:-2, 1:-1])
    length = np.linalg.norm(cross, axis=-1)
    normal[1:-1, 1:-1] = np.divide(cross, length[..., None], out=np.zeros_like(cross),
                                   where=length[..., None] > 1e-12)
    surface_valid = dense & (np.linalg.norm(normal, axis=-1) > .5)
    normal[~surface_valid] = 0.
    variation, residual = np.zeros_like(z), np.zeros_like(z)
    # Fixed four-neighbor denominator, not the number of surviving neighbors.
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor_n = np.roll(normal, (dy, dx), axis=(0, 1))
        neighbor_xyz = np.roll(xyz, (dy, dx), axis=(0, 1))
        both = surface_valid & np.roll(surface_valid, (dy, dx), axis=(0, 1))
        both[[0, -1], :] = False; both[:, [0, -1]] = False
        dot = np.sum(normal*neighbor_n, axis=-1)
        variation += np.where(both, np.clip(1.-dot, 0., 2.)/2., 0.)/4.
        distance = np.abs(np.sum(normal*(neighbor_xyz-xyz), axis=-1))
        residual += np.where(both, np.clip(distance/.01, 0., 1.), 0.)/4.
    return dict(valid_fraction3=valid3, normal_variation=variation,
                plane_residual=residual, surface_valid=surface_valid.astype(np.float64))


def local_features(rgb, depth, valid, predicted_maps, intrinsic, original_size):
    """Return float32 [81,4,4,28]; only observable inputs are accepted.

    Candidate (dx,dy) reads RGB at (x-dx,y-dy) for fixed observed depth (x,y),
    exactly the frozen v43 convention. No variable candidate support/filtering.
    """
    z, ok = old.observations(depth, valid)
    color = np.asarray(rgb, np.float64)
    require(color.shape == (*z.shape, 3) and np.isfinite(color).all()
            and np.all((color >= 0) & (color <= 1)), 'RGB must be finite [0,1]')
    maps = {name: np.asarray(predicted_maps[name], np.float64) for name in MAP_NAMES}
    require(all(a.shape == z.shape and np.isfinite(a).all() and np.all((a >= 0) & (a <= 1))
                for a in maps.values()), 'Invalid predicted RGB maps')
    h, w = z.shape
    ids, regions, counts = fixed_partition(z.shape)
    yy, xx = np.divmod(ids, w)
    gray = color @ np.array([.299, .587, .114])
    ry, rx = np.gradient(gray); zy, zx = np.gradient(z)
    geom = geometry_maps(z, ok, intrinsic, original_size)
    dense = geom['valid_fraction3'] > .999
    zy, zx = np.where(dense, zy, 0.), np.where(dense, zx, 0.)
    rm, zm = np.hypot(rx, ry), np.hypot(zx, zy)
    re, de = rm/(rm+.02), zm/(zm+.02)
    rx = np.divide(rx, rm, out=np.zeros_like(rx), where=rm > 0)
    ry = np.divide(ry, rm, out=np.zeros_like(ry), where=rm > 0)
    zx = np.divide(zx, zm, out=np.zeros_like(zx), where=zm > 0)
    zy = np.divide(zy, zm, out=np.zeros_like(zy), where=zm > 0)
    r3, d3 = uniform_filter(re, 3), uniform_filter(de, 3)
    r7, d7 = uniform_filter(re, 7), uniform_filter(de, 7)
    domain = np.zeros(z.shape, bool); domain[8:-8, 8:-8] = True
    interior = domain & dense & (zm < .02)
    med = median_filter(z, 3, mode='nearest')
    noise = min(20., float(np.median(np.abs(z[interior]-med[interior])))/.005) if interior.any() else 0.
    valid_fraction = float(ok[domain].mean())
    fixed_valid = ok[yy, xx].astype(float)
    g = {name: array[yy, xx] for name, array in geom.items()}
    result = np.zeros((81, 16, len(FEATURE_NAMES)), np.float64)
    for index, (dx, dy) in enumerate(CANDIDATES):
        sy, sx = yy-dy, xx-dx
        require(np.all((sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)), 'Fixed support escaped image')
        r, d = re[sy, sx], de[yy, xx]
        direction = rx[sy, sx]*zx[yy, xx]+ry[sy, sx]*zy[yy, xx]
        product, absolute = r*d, r*d*np.abs(direction)
        confidence = maps['instance_confidence'][sy, sx]
        boundary = maps['soft_boundary'][sy, sx]
        proximity = maps['boundary_proximity'][sy, sx]
        inside_instance = confidence*(1.-boundary)
        values = np.column_stack((product, product*direction, absolute, np.abs(r-d),
            ((r > .5) & (d > .5)).astype(float), r*(d > .5), d*(r > .5),
            r3[sy, sx]*d3[yy, xx], r7[sy, sx]*d7[yy, xx], np.ones(len(ids)),
            product*dx/4., product*dy/4., product*(dx*dx+dy*dy)/32.,
            product*noise, product*valid_fraction, absolute*valid_fraction,
            confidence, boundary, proximity, boundary*d, proximity*d,
            inside_instance*d, g['valid_fraction3'], inside_instance*g['normal_variation'],
            inside_instance*g['plane_residual'], confidence*g['surface_valid'],
            confidence*d, maps['foreground_probability'][sy, sx]*fixed_valid))
        for region in range(16):
            result[index, region] = values[regions == region].sum(axis=0)/counts[region]
    require(np.isfinite(result).all(), 'Nonfinite local features')
    return result.reshape(81, 4, 4, len(FEATURE_NAMES)).astype(np.float32)
