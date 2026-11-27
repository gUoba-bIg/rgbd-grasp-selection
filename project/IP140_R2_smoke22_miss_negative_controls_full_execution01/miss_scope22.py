"""One frozen first-lift certificate policy for the three previously registered misses."""
import hashlib
import json
import math

REGISTRATIONS = {
    10: ("021", "ip140bridge:slot10:undo-legacy-z021:miss", "0141ab0063cb8fa8eda7d3cb7ea3ddbdafc5f16b2edc0f12cb49577ff6b45ef3",
         "ddc1ae3ba47a83501a957d50bf3543bdd594f316c615e7ac9c221ae851017c8d"),
    11: ("014", "ip140bridge:slot11:undo-legacy-z014:miss", "9188133d215648ae722354a77bbe0b5869c9772ef29b466b09761770277c0c6d",
         "eda6d1ca31d0cac0eb99aa39093cb944789e8dba319272e363a0f254582c0e20"),
    12: ("017", "ip140bridge:slot12:undo-legacy-z017:miss", "f2ac9a9225545f0e614bb2a10160e40060ea21853f9a54364530c5f8345f868c",
         "c41b6b0ee31822080bd775411041937b567b9f3e5e3469032bbefd3565b91c4d"),
}
SIDES = frozenset(("left","right"))

def registered_miss(slot, row, preregistration_digest):
    expected = REGISTRATIONS.get(slot)
    if (expected is None or type(slot) is not int or type(row.get("slot")) is not int
            or row["slot"] != slot or row.get("not_p3") is not True or row.get("kind") != "miss"):
        return False
    candidate = row.get("candidate_world")
    if (type(candidate) is not list or len(candidate) != 17 or
            not all(type(x) in (int,float) and math.isfinite(x) for x in candidate)):
        return False
    encoded = json.dumps(candidate,separators=(",",":"),allow_nan=False).encode("ascii")
    actual = (row.get("object_id"),row.get("candidate_id"),hashlib.sha256(encoded).hexdigest(),preregistration_digest)
    return actual == expected

def pre_lift_gate(roots, scoped_miss):
    if not set(roots) <= SIDES or any(cert.get("passed") is not True for cert in roots.values()):
        raise ValueError("existing_entry_certificate_failed_or_unknown_side")
    missing = sorted(SIDES - roots.keys())
    if missing and scoped_miss is not True:
        raise ValueError("both_registered_entries_required_before_lift")
    return dict(registered_miss_scope=scoped_miss,pre_lift_bilateral_entry_absent=bool(missing),
                missing_entry_sides=missing, valid_entry_sides=sorted(roots))

def enforce_nonpad_clearance(item, source):
    """Preserve the nonpad-target collision gate on the frozen-target survey."""
    if (item.get("contact_part")=="nonpad" and
            (source == "manifold" or item.get("geometric_contact") is True)):
        item["violations"].append("nonpad_target_collision")
