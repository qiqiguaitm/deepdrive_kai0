# Rerun Mesh Transparency — A Debugging Postmortem

A long debugging session chasing "why aren't the arms transparent in Rerun
when `verify_calibration.py` shows them semi-transparent?" ended up being a
silent **Rerun version mismatch between two Python environments**. Capturing
the lessons so next time we find it in 5 minutes instead of 3 hours.

## TL;DR

- `calib/verify_calibration.py` runs with **conda/system Python** → `rerun 0.31.1` → mesh `albedo_factor` alpha **works**
- `ros2_ws/.../rerun_viz_node.py` ran with **`kai0/.venv`** → `rerun 0.23.1` → mesh alpha **silently ignored**, mesh drawn fully opaque
- Mesh alpha blending for `Mesh3D` was added to Rerun's renderer between 0.23 and 0.31
- Fix: `uv pip install --python kai0/.venv/bin/python rerun-sdk==0.31.1` (and keep `numpy==1.26.4` — the upgrade pulls numpy 2.x which breaks JAX/torch)

## Symptom

`rr.Mesh3D(..., albedo_factor=[r, g, b, 10])` rendered as **fully opaque**
colored mesh in our ROS2 viz node, even though the exact same pattern in
`verify_calibration.py` produced obviously see-through arms over the point
cloud. No warning, no error — the alpha channel was just dropped.

## Wrong hypotheses we chased (and why they were wrong)

All of these *felt* plausible and wasted real time:

1. **"Rerun `Mesh3D` never supports alpha, the calib arms just look dim."**
   - We checked GitHub issue [#1611](https://github.com/rerun-io/rerun/issues/1611),
     which is still open. Concluded alpha is not implemented. **Wrong** —
     the issue is old; alpha was quietly added in a later release.
   - Lesson: an "open" issue isn't proof the feature doesn't exist. Check the
     actual installed package behavior.

2. **"`vertex_colors` RGBA alpha works, `albedo_factor` alpha doesn't."** (or vice versa)
   - Swapped between the two patterns repeatedly. Neither worked. Both are
     honored on 0.31, both are ignored on 0.23.
   - Lesson: if two supposedly-different encodings both fail identically,
     the problem is one level up (renderer, not encoding).

3. **"The mesh entity path structure is wrong."**
   - Moved mesh between `world/{arm}/{link}` (same entity as dynamic
     `Transform3D`) and `world/{arm}/{link}/mesh` (child entity). Caused
     real regressions on position updates but had no effect on transparency.
   - Lesson: entity path is a correctness issue for transforms, not a
     material issue.

4. **"Missing `vertex_normals` makes Rerun fall back to an unlit shader
   that doesn't alpha-blend."**
   - Very plausible — `trimesh.load()` provides normals, my raw STL parser
     didn't. Computed area-weighted per-vertex normals, re-tested. **Still
     opaque** on 0.23. Still opaque means normals weren't the gate.
   - Lesson: vertex normals *do* affect shading quality but don't gate
     alpha support. Keep them anyway — they're free and improve shading.

## The actual diagnosis (what finally worked)

Instead of reasoning about Rerun internals, **actually reproduce the calib
pattern in isolation and inspect the output:**

```bash
# Use the SAME python that verify_calibration.py runs with
/data1/miniconda3/bin/python << 'PY'
import rerun as rr
print(rr.__version__)         # <-- this line broke the case open
PY
# conda python: 0.31.1

source kai0/.venv/bin/activate
python -c "import rerun; print(rerun.__version__)"
# kai0 venv: 0.23.1
```

Two different Python interpreters → two different `rerun-sdk` versions →
two different mesh renderers. `verify_calibration.py` is invoked as a bare
script and picks up whichever `python3` is first on PATH (conda). The ROS2
node is invoked by `ros2 launch` with a shebang + PYTHONPATH pointing at
`kai0/.venv/lib/python3.12/site-packages`, so it uses the pinned kai0 deps.

## Fix

```bash
# From kai0/ root
uv pip install --python .venv/bin/python "rerun-sdk==0.31.1"
# rerun upgrade pulls numpy 2.x — pin it back before anything else breaks
uv pip install --python .venv/bin/python "numpy==1.26.4"
```

No source code changes are required beyond this; the `set_time(...,
timestamp=...)` API is the same on 0.23 and 0.31. The `vertex_normals`
fix in `_load_binary_stl` is unrelated to transparency but still worth
keeping — trimesh-equivalent shading quality comes for free.

## Lessons learned

### 1. Identify the interpreter BEFORE you debug behavior

When two scripts behave differently, the **first question** should be "are
they running with the same Python environment?" Not "what's different in the
code?" A 30-second version check would have skipped the entire 3-hour detour
through entity paths, vertex normals, and Rerun renderer speculation.

Things to check, in order:

```bash
which python           # which interpreter
python -c "import X; print(X.__version__)"    # which package version
python -c "import X; print(X.__file__)"       # which install location
```

Any time a script "works over there but not here," these three commands
first.

### 2. ROS2 launch files silently pick an interpreter

Our `inference_full_launch.py` sets `PYTHONPATH` to prepend `kai0/.venv`'s
`site-packages` to the ROS2 node's environment. That means the node's
shebang (`#!/usr/bin/python3`) imports packages from kai0's venv, *not* from
whatever `/usr/bin/python3` would normally resolve. The launch file is the
hidden "interpreter selector," and its effects don't show up in
`which python`.

If you ever need to know what a ROS2 python node actually imports, add this
at the top of the node:

```python
import sys, rerun  # or whatever package you suspect
print(f"[diag] python={sys.executable}")
print(f"[diag] rerun={rerun.__version__} from {rerun.__file__}")
```

### 3. "Works on my machine" scripts and "fails in production" nodes are
   fighting over `PATH`/`PYTHONPATH`

The calib script and the ROS2 node exist *in the same repo*, on the *same
machine*, using *the same Rerun import statement*. The difference was
entirely which `site-packages` they resolved against. Expect this class of
bug whenever you have:

- A user-interactive script (`python script.py`) — picks up whichever env
  is currently active.
- A systemd service / ROS2 launch / cron job — picks up whatever env its
  launcher configured.

The two can easily drift apart and nothing in the code will warn you.

### 4. Reproduce in the SIMPLEST possible isolated script first

What finally found the bug was a 10-line standalone script: load one STL,
log one mesh, save to rrd. That stripped away ROS2, JAX, blueprints,
timers, entity hierarchies, everything. At that point the only variable
left was the Python interpreter — and comparing `rerun.__version__` across
interpreters trivially showed the mismatch.

Rule of thumb: when a complex system is producing a "material" bug (wrong
colors, wrong alpha, wrong layout, etc.), reproduce it in 20 lines of
pure-library code before poking at the integration.

### 5. An "open issue" on GitHub is NOT documentation

Rerun issue #1611 ("Support transparency/alpha for spatial primitives") is
still open as of April 2026 even though `Mesh3D` alpha actually works in
0.31. The issue tracker is a wishlist, not a feature matrix. Trust the
installed package's behavior, not the issue status.

### 6. When upgrading one package, pin everything else explicitly

`uv pip install rerun-sdk==0.31.1` silently upgraded `numpy` from 1.26 to
2.4. numpy 2.x has breaking C-ABI changes that would have broken JAX, torch,
OpenCV, and likely crashed the policy server on next launch. Always follow
up a targeted upgrade with `pip list | diff` against the previous state and
re-pin anything that moved without permission.

## Debugging checklist for future Rerun "it doesn't look right" issues

1. [ ] Which Python interpreter runs the code that produces the broken output?
2. [ ] What version of `rerun-sdk` does *that* interpreter see?
3. [ ] What version of the **Rerun viewer binary** is being used? (`rerun --version`)
       SDK and viewer are separate; a mismatch between them can also cause
       silent feature drops.
4. [ ] Reproduce the exact archetype call in a 10-line standalone script using
       the same interpreter.
5. [ ] Save to `.rrd` and open with `rerun file.rrd` — bypasses any spawn
       / TCP weirdness in the original pipeline.
6. [ ] Only after 1–5 pass should you suspect your code.
