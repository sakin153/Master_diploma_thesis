"""Per-category physics profiles for MuJoCo scene assembly.

Determines whether an object is static (welded, furniture) or dynamic
(freejoint, movable), and what physics parameters to use.

Mass strategy
-------------
MuJoCo computes body mass as  density × geom_volume  when density is set on a
geom. Since Objaverse meshes have arbitrary bounding-box volumes that don't
reflect the real object's hollow interior, we instead derive a *synthetic
density* from a realistic target mass:

    density = target_mass_kg / (bbox_volume × fill_factor)

where `fill_factor` accounts for hollow geometry (cup ≈ 0.15, book ≈ 0.85).
The result is passed to `get_physics_profile_for_model()` which takes the
actual model size.
"""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class PhysicsProfile:
    # If True: body has no joint → welded to worldbody (never moves).
    # This is correct for furniture. Objects will NEVER fly off.
    # If False: body gets <joint type="free"/> → can be moved by physics.
    is_static: bool

    # Friction coefficients: [sliding, torsional, rolling]
    friction: Tuple[float, float, float]

    # Material density in kg/m³ (used by MuJoCo for mass/inertia auto-compute).
    # For dynamic objects this is overridden per-instance by
    # compute_density_for_mass() to achieve a realistic target mass.
    density: float

    # Contact dimensionality: 3 = full friction (x,y,z,torque,roll)
    condim: int

    # Constraint solver reference: [timeconst_s, damping_ratio]
    solref: Tuple[float, float]

    # Constraint impedance: [d_min, d_max, width, midpoint, power]
    solimp: Tuple[float, float, float, float, float]


# ---------------------------------------------------------------------------
# Predefined profiles (density here is a sensible fallback; the pipeline
# overrides it with compute_density_for_mass() for dynamic objects)
# ---------------------------------------------------------------------------

# Heavy, stable furniture: tables, chairs, sofas, cabinets…
STATIC_FURNITURE = PhysicsProfile(
    is_static=True,
    friction=(0.8, 0.005, 0.0001),
    density=600.0,   # wood/fabric mix (kg/m³)
    condim=3,
    solref=(0.01, 1.0),
    solimp=(0.95, 0.99, 0.001, 0.5, 2),
)

# Light static decoration: lamps, plants, TVs
STATIC_LIGHT = PhysicsProfile(
    is_static=True,
    friction=(0.7, 0.004, 0.0001),
    density=200.0,
    condim=3,
    solref=(0.01, 1.0),
    solimp=(0.9, 0.99, 0.001, 0.5, 2),
)

# Small dynamic objects: books, cups, bottles, phones, laptops…
DYNAMIC_SMALL = PhysicsProfile(
    is_static=False,
    friction=(0.6, 0.004, 0.0001),
    density=400.0,   # overridden per-instance by compute_density_for_mass()
    condim=3,
    solref=(0.01, 1.0),
    solimp=(0.9, 0.95, 0.001, 0.5, 2),
)

# Heavy appliances: refrigerator, washing machine
STATIC_APPLIANCE = PhysicsProfile(
    is_static=True,
    friction=(0.9, 0.005, 0.0001),
    density=400.0,
    condim=3,
    solref=(0.01, 1.0),
    solimp=(0.95, 0.99, 0.001, 0.5, 2),
)


# ---------------------------------------------------------------------------
# Keyword → profile mapping
# ---------------------------------------------------------------------------

_PROFILE_RULES: Tuple[Tuple[Tuple[str, ...], PhysicsProfile], ...] = (
    # Dynamic small objects
    (
        (
            "book", "cup", "mug", "glass", "bottle", "plate", "dish", "bowl",
            "phone", "smartphone", "laptop", "keyboard", "mouse", "pen",
            "pencil", "marker", "ball", "toy", "vase", "pot", "flower",
            "remote", "controller", "pillow", "cushion", "apple", "fruit",
            "candle", "figurine", "stapler", "scissors", "tape",
        ),
        DYNAMIC_SMALL,
    ),
    # Heavy appliances (static)
    (
        (
            "refrigerator", "fridge", "washing machine", "washer", "dryer",
            "dishwasher", "oven", "stove", "microwave", "air conditioner",
        ),
        STATIC_APPLIANCE,
    ),
    # Light decoration (static but light)
    (
        (
            "lamp", "chandelier", "ceiling light", "light fixture",
            "plant stand", "tv", "television", "monitor", "screen",
            "picture frame", "painting", "wall art", "clock", "mirror",
        ),
        STATIC_LIGHT,
    ),
    # Heavy furniture (static) — catch-all for furniture
    (
        (
            "table", "desk", "chair", "sofa", "couch", "bed", "mattress",
            "cabinet", "wardrobe", "closet", "shelf", "bookcase", "bookshelf",
            "rack", "bench", "armchair", "ottoman", "stool", "seat",
            "nightstand", "dresser", "chest", "sideboard", "buffet",
            "toilet", "sink", "bathtub", "shower", "door", "window",
            "stairs", "staircase", "counter", "bar", "podium",
        ),
        STATIC_FURNITURE,
    ),
)


def get_physics_profile(model_name: str) -> PhysicsProfile:
    """Return the most appropriate physics profile for a model by name."""
    lname = (model_name or "").lower()
    for keywords, profile in _PROFILE_RULES:
        if any(kw in lname for kw in keywords):
            return profile
    # Default: treat unknown objects as static furniture (safe — prevents
    # unknown objects from flying off).
    return STATIC_FURNITURE


def is_static_object(model_name: str) -> bool:
    return get_physics_profile(model_name).is_static


# ---------------------------------------------------------------------------
# Realistic mass → density conversion
# ---------------------------------------------------------------------------

# Real-world target masses per object category (kg).
# Sources: product specs, physics textbooks, common sense.
_MASS_TARGETS_KG: Tuple[Tuple[Tuple[str, ...], float], ...] = (
    # Tableware / kitchenware
    (("plate", "dish"),             0.55),   # ceramic dinner plate ~500-600 g
    (("bowl",),                     0.45),   # ceramic bowl ~400-500 g
    (("cup", "mug"),                0.35),   # ceramic mug + contents ~300-400 g
    (("glass",),                    0.25),   # drinking glass ~200-300 g
    (("bottle",),                   0.60),   # half-full water bottle
    # Stationery / tech
    (("book",),                     0.60),   # average paperback/hardcover
    (("laptop",),                   1.80),   # laptop ~1.5-2 kg
    (("keyboard",),                 0.80),   # desktop keyboard
    (("phone", "smartphone"),       0.20),   # ~180-200 g
    (("mouse",),                    0.12),
    (("remote", "controller"),      0.15),
    (("pen", "pencil", "marker"),   0.02),
    (("stapler",),                  0.40),
    (("scissors",),                 0.10),
    (("tape",),                     0.10),
    # Decorative
    (("vase",),                     0.80),   # ceramic/glass vase empty
    (("pot", "flower"),             1.20),   # flower pot with soil
    (("candle",),                   0.20),
    (("figurine",),                 0.30),
    (("pillow", "cushion"),         0.60),
    (("ball",),                     0.45),   # soccer-ball size
    # Food
    (("apple", "fruit"),            0.18),
    # Default for unmatched dynamic objects
)

# Fill factor: what fraction of the bounding-box volume is actually occupied
# by the object's material (accounts for hollow/open geometry).
_FILL_FACTORS: Tuple[Tuple[Tuple[str, ...], float], ...] = (
    (("cup", "mug", "glass"),              0.15),  # hollow cylinder
    (("bowl",),                            0.20),  # hemispherical shell
    (("vase", "pot"),                      0.15),  # hollow vessel
    (("bottle",),                          0.25),  # hollow + some liquid
    (("plate", "dish"),                    0.90),  # solid flat disk
    (("pillow", "cushion"),                0.30),  # fluffy filling
    (("book",),                            0.85),  # dense pages
    (("ball",),                            0.10),  # hollow sphere
    (("laptop",),                          0.35),  # thin shell with internals
    (("keyboard",),                        0.50),
    (("phone", "smartphone"),              0.70),
    (("figurine",),                        0.60),
    (("apple", "fruit"),                   0.90),  # solid
    (("candle",),                          0.80),
)

_DEFAULT_MASS_KG = 0.50
_DEFAULT_FILL = 0.50


def _get_mass_target(model_name: str) -> float:
    lname = (model_name or "").lower()
    for keywords, mass in _MASS_TARGETS_KG:
        if any(kw in lname for kw in keywords):
            return mass
    return _DEFAULT_MASS_KG


def _get_fill_factor(model_name: str) -> float:
    lname = (model_name or "").lower()
    for keywords, fill in _FILL_FACTORS:
        if any(kw in lname for kw in keywords):
            return fill
    return _DEFAULT_FILL


def compute_density_for_mass(
    model_name: str,
    size: Sequence[float],
) -> float:
    """Return a MuJoCo geom density (kg/m³) that achieves the real-world target
    mass for this object category given the model's actual bounding-box size.

    Formula:
        density = target_mass / (bbox_volume * fill_factor)

    Clamped to [50, 8000] kg/m³ to avoid degenerate values.
    """
    if len(size) < 3:
        return DYNAMIC_SMALL.density

    w = max(0.01, float(size[0]))   # scene X  (width)
    h = max(0.01, float(size[1]))   # scene Z  (height)
    d = max(0.01, float(size[2]))   # scene Y  (depth)
    bbox_vol = w * h * d

    target_mass = _get_mass_target(model_name)
    fill = _get_fill_factor(model_name)
    effective_vol = bbox_vol * max(0.05, fill)

    density = target_mass / max(1e-6, effective_vol)
    # Clamp: denser than gold (19300) is unrealistic; < 10 is a ghost
    return max(50.0, min(8000.0, density))


def get_physics_profile_for_model(
    model_name: str,
    size: Optional[Sequence[float]] = None,
) -> PhysicsProfile:
    """Return a PhysicsProfile with density tuned to achieve realistic mass.

    For dynamic objects the density is computed from the target mass and the
    actual model bounding box so that MuJoCo computes the correct mass.
    For static objects the profile is returned unchanged (mass doesn't affect
    their dynamics since they are welded).
    """
    profile = get_physics_profile(model_name)
    if profile.is_static or size is None or len(size) < 3:
        return profile

    # Dynamic object: override density to match realistic mass
    density = compute_density_for_mass(model_name, size)
    # Return a new frozen instance with updated density
    return PhysicsProfile(
        is_static=profile.is_static,
        friction=profile.friction,
        density=density,
        condim=profile.condim,
        solref=profile.solref,
        solimp=profile.solimp,
    )
