"""Point annotation on the frozen IP140 pad in its URDF link frame (metres).

The collision surface is the 0.5 mm offset of the inset box. The projection
below is onto the inset box, never a correction to either native witness.
Labels describe the pad witness. They never authorize a contact or an action.
"""
import numpy as np

PAD_LENGTH_M = .060
PAD_THICKNESS_M = .002
PAD_HEIGHT_M = .020
MARGIN_M = .0005
GEOMETRY_TOL_M = 1e-9


def classify_pad_surface(side, pad_point_m, normal_pad_to_target_pad=None):
    if side not in ("left", "right"):
        raise ValueError("unregistered pad side")
    point = np.asarray(pad_point_m, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("invalid pad-local witness")
    normal = None if normal_pad_to_target_pad is None else np.asarray(normal_pad_to_target_pad, dtype=float)
    if normal is not None and (normal.shape != (3,) or not np.isfinite(normal).all()):
        raise ValueError("invalid pad-local contact normal")

    y_min = -PAD_THICKNESS_M if side == "left" else 0.
    lo = np.array([0., y_min, -PAD_HEIGHT_M / 2])
    hi = np.array([PAD_LENGTH_M, y_min + PAD_THICKNESS_M, PAD_HEIGHT_M / 2])
    centre = np.clip(point, lo + MARGIN_M, hi - MARGIN_M)
    radial = point - centre
    radius = float(np.linalg.norm(radial))
    residual = radius - MARGIN_M
    axis_mask = np.abs(radial) > GEOMETRY_TOL_M
    inward_sign = 1. if side == "left" else -1.
    inner = inward_sign * radial[1] > GEOMETRY_TOL_M
    tip = radial[0] > GEOMETRY_TOL_M
    axes = int(np.count_nonzero(axis_mask))
    if abs(residual) > GEOMETRY_TOL_M:
        label = "witness_off_nominal_rounded_surface"
    elif axes == 1 and inner:
        label = "flat_inner_face"
    elif axes == 3 and inner and tip:
        label = "rounded_tip_corner_contact_risk"
    elif axes == 2 and inner and tip:
        label = "rounded_tip_edge_contact_risk"
    elif axes >= 2:
        label = "other_rounded_corner" if axes == 3 else "other_rounded_edge"
    elif axes == 1 and tip:
        label = "flat_tip_face"
    else:
        label = "other_flat_face"
    radial_normal = radial / radius if radius > 0 else None
    alignment = None if normal is None or radial_normal is None else float(radial_normal @ normal)
    return dict(label=label, side=side, point_local_m=point.tolist(),
                tip_depth_m=float(PAD_LENGTH_M - point[0]), inner_face_offset_m=float(point[1]),
                inward_signed_offset_m=float(inward_sign * point[1]), margin_m=MARGIN_M,
                core_closest_local_m=centre.tolist(), radial_vector_m=radial.tolist(),
                radial_length_m=radius, round_radius_residual_m=float(residual),
                outward_normal_from_geometry=None if radial_normal is None else radial_normal.tolist(),
                normal_alignment_with_geometry=alignment, geometry_tolerance_m=GEOMETRY_TOL_M)
