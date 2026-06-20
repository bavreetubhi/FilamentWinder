# Filament Winding Paths and Patterns

This document explains the winding concepts used by the planner in this
repository. It is written for someone building filament winding software, not
for someone already familiar with winding CAM.

## What A Winding Path Is

A filament winding path is the centreline followed by the tow or band on the
mandrel surface. It is not just a decorative spiral. The path must describe:

- where the tow touches the mandrel surface
- how the mandrel rotates
- how the carriage moves along the mandrel
- how the payout eye turns
- how the next pass is offset from the previous pass
- how the layer eventually closes without leaving an arbitrary mismatch

The machine axes used here are:

| Axis | Meaning |
|---|---|
| `A` | Mandrel rotation in degrees |
| `Z` | Carriage position along the mandrel |
| `X` | Radial payout-eye distance from the centreline |
| `B` | Tow-eye orientation or payout-eye rotation |

The planner always starts from mandrel surface geometry. For an axisymmetric
mandrel this means a radius function:

```text
r = r(z)
```

The surface point at a longitudinal position and mandrel angle is:

```text
P(z, theta) = [r(z) cos(theta), r(z) sin(theta), z]
```

## Winding Angle

The winding angle is the angle between the tow direction and the mandrel
longitudinal direction. A low angle runs mostly along the length of the part. A
high angle approaches a hoop wrap around the circumference.

On a constant-radius cylinder, the relationship is simple:

```text
tan(angle) = radius * dtheta / dz
```

On domes, transitions, and nosecones, the radius and surface slope change. The
planner therefore calculates local radius and local path angle instead of
assuming one flat cylinder unwrap is valid everywhere.

## Pattern Closure

A winding pattern closes when the tow returns to a compatible angular position
after a finite number of circuits. If it does not close, the final pass lands at
an arbitrary phase and the layer will not repeat cleanly.

For a cylinder, one axial pass has:

```text
turns_per_pass = tan(angle) * length / (2*pi*radius)
```

A closed cylinder pattern usually chooses an integer `turns_per_pass`, then
chooses enough passes to make the band spacing match the tow width.

The planner reports:

- target winding angle
- actual achievable winding angle
- number of circuits
- number of starts
- angular shift between circuits
- tow spacing
- gap or overlap
- coverage percentage
- whether the pattern closes
- whether the pattern is acceptable

When the requested angle cannot close cleanly, the cylinder optimizer searches
nearby closed integer-turn alternatives and ranks them by coverage and angle
error.

## Circuits, Starts, And Index Shift

A circuit is one repeated winding pass pattern. On a cylinder this is usually
one axial traversal. On a dome path it may include an outbound span, turnaround,
return span, and second turnaround.

The angular shift is the mandrel phase offset between neighbouring circuits. If
the shift is too large, there will be gaps. If it is too small, the band will
overlap too heavily.

For a target full layer:

```text
tow_spacing ~= tow_width / coverage_target
cylinder_helical_circuits ~= circumference * cos(angle) / tow_spacing
dome_circuits ~= max_circumference / tow_spacing
```

The cosine term matters on a cylinder because neighbouring helical bands are
spaced perpendicular to the tow, not just around a raw circumferential phase
offset. Dome coverage still uses the maximum circumference as a conservative
first-order estimate.

## Coverage

Coverage is checked by counting how many tow bands cover each sampled surface
cell. A count of zero is a gap. A count greater than one is overlap.

The current planner provides two coverage paths:

- cylinder coverage using the established Z-theta cylinder map
- axisymmetric surface coverage using local radius at each Z sample

The profile coverage map is still approximate, but it is surface-aware: the
local circumference changes with radius rather than using a single cylinder
diameter for the whole part.

## Helical Winding

Helical winding is used for structural axial and torsional strength. A helical
layer has an angle such as `+35 deg` or `-35 deg`.

For a complete helical layer the planner:

1. Searches for a closed cylinder pattern near the requested angle.
2. Calculates the number of passes needed for the tow width and coverage target.
3. Applies a phase shift between passes.
4. Alternates carriage direction where appropriate.
5. Reports gap, overlap, and closure quality.

For `-angle` layers the planner mirrors the mandrel rotation and payout-eye
angle. This lets schedules build balanced `+angle / -angle` laminates.

## Hoop Winding

Hoop winding is near `90 deg`. The carriage advances by roughly one tow width
between circumferential wraps.

For a hoop layer the planner:

1. Calculates ring positions along `Z`.
2. Generates one circumferential circuit per ring.
3. Sets the payout-eye angle to `90 deg`.
4. Reports axial tow spacing, gap, and overlap.

Hoop paths are useful on cylinders and straight sections. They are not a good
model for pointed noses or deep domes.

## Dome, Polar, Nosecone, And Transition Winding

Changing-radius regions need more care than cylinders. The path must respond to:

- changing local radius
- changing surface slope
- increasing curvature near turnaround zones
- steering limits
- possible tow slip
- possible bridging over concave or tight regions

For dome and polar-style profile winding, the planner uses the profile radius
function and a geodesic-style relation. The requested angle is treated as the
cylinder or max-radius angle. As the tow approaches smaller radii, the local
winding angle rises until the turnaround radius is reached.

The automatic geodesic turnaround radius is:

```text
turnaround_radius = max_radius * sin(target_angle)
```

The path then turns around at constant `Z` before returning along the other
direction. The output includes variable `B` angle, so the payout eye can follow
the changing local tow angle.

In this codebase:

- `dome` uses the profile dome generator
- `polar` mirrors the dome generator direction
- `nosecone` uses a fixed-angle profile turnaround to a configured minimum
  radius, which is useful for pointed or tapered one-ended profiles
- `axisymmetric` uses a profile helix when the whole imported shape is safely
  above the minimum radius, with a safe turnaround fallback when the profile has
  poles or small-radius ends
- `transition` is represented as a controlled connection between layer paths

Future work can add calibrated non-geodesic steering and friction-limit rules
for each specific profile region.

## Layer Schedules

A winding schedule is a list of layers. Each layer defines:

- layer name
- winding type
- target angle
- tow width
- layer thickness
- coverage target
- direction
- turnaround rule
- closure rule
- transition mode

Example schedule:

```text
Layer 1: hoop
Layer 2: +35 deg helical
Layer 3: -35 deg helical
Layer 4: dome reinforcement
```

The planner produces one combined program with metadata for each point:

- layer index
- layer name
- circuit index
- pass index
- winding type
- local radius
- local winding angle
- surface position
- A/X/Z/B machine position
- feedrate target
- curvature and slip-risk estimates

## Layer Transitions

A real machine should not teleport from the end of one layer to the start of
the next. The planner adds controlled transition moves when the layer is marked
as continuous.

A transition interpolates:

- surface `Z`
- mandrel angle `theta`
- payout-eye angle `B`

For production CAM this may later become a cut/restart, dwell, clamp, or manual
operator step. The current representation keeps the path continuous and clearly
labels the points as `transition`.

## Validation

Validation checks whether a planned program should be trusted. The current
planner reports:

- open patterns
- excessive angle error
- tow gaps
- excessive overlap
- high slip-risk estimates
- non-finite machine motion

Existing machine validation can additionally check:

- A/X/Z/B axis limits
- rectangular no-go zones

Useful warnings are better than silent bad output. A visually plausible path
can still be a bad winding path if it cannot close, leaves uneven gaps, turns
too sharply, or asks the tow to steer unrealistically.

## Practical Limits

The planner in this repository is now path-level and layer-aware, but it is not
yet a complete production CAM system. Remaining production features include:

- exact tow footprint projection on arbitrary profile curvature
- detailed compaction and resin model
- collision geometry for the payout head
- acceleration-limited motion planning
- machine-specific post-processing
- non-geodesic dome steering with calibrated slip factors
- operator events such as cut, clamp, restart, and dwell

Those features should be added on top of the current planner and validation
model, not by replacing the mandrel geometry or GUI.
