"""Bridge between the pebble-trough bundle and an IsaacLab :class:`InteractiveScene`.

Two problems are solved here, both learned the hard way.

1. **Rigid bodies must exist before the first ``sim.reset()``.**
   ``pebble_loader.build_pebble_scene()`` may be called at any time, but if it is called
   *after* the environment has been created, PhysX has already built its fabric views
   (``GpuRigidContactView``, rigid-body fabric buffers) from the set of rigid bodies that
   existed at that moment. Appending ~4.5k new dynamic bodies afterwards leaves the views
   stale and every contact query dies with::

       GpuRigidContactView.cpp: CUDA error: illegal memory access
       Failed to get rigid body fabric buffer view

   which is why the trough demo only ever ran with ``--pebble_mode static``.
   ``InteractiveScene.__init__`` calls ``_add_entities_from_cfg()`` *before*
   ``ManagerBasedEnv`` calls ``sim.reset()``, so spawning the field from a ``SceneCfg``
   entry makes the pebbles part of the very first PhysX view build. That is the whole
   trick: the spawner below is registered as an :class:`AssetBaseCfg` and does its work in
   ``func(prim_path, spawn_cfg, translation=..., orientation=...)``.

2. **A generated terrain must not be used for this demo.**
   The parkour terrain generator undulates by up to ~0.2 m — the same size as the 0.20 m
   trough walls, so the trough gets buried — and it moves ``env_origins`` around on reset,
   which forced a *post-reset* translation of the whole field (a transform PhysX has already
   consumed). The trough floor is modelled at ``z = 0`` (``trough_floor`` spans ``[-0.05, 0]``),
   so switching the terrain to a plane makes the plane *be* the trough floor, leaves
   ``env_origins`` identically zero and needs no late transform at all.

The module deliberately keeps all Isaac-API imports inside the functions, so importing it
from ``play.py`` before ``AppLauncher`` starts is harmless.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

#: spawner/appearance settings that worked in the 4467-pebble demo
DEFAULT_PRIM_PATH = "/World/pebble"

#: last scene dictionary built by the spawner (see :func:`last_scene`)
_LAST_SCENE: dict = {}


# ---------------------------------------------------------------- bundle helpers
def loader_module(bundle_dir):
    """Import ``pebble_loader`` from the directory that contains ``bundle_dir``."""
    parent = str(Path(bundle_dir).resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    import pebble_loader  # noqa: PLC0415  (deliberately lazy: needs no Isaac import)

    return pebble_loader


def trough_boxes(bundle_dir):
    """``static/stage_boxes.json`` as a list of dicts (boxes, pad, bulk fill)."""
    with open(os.path.join(bundle_dir, "static", "stage_boxes.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def standing_pad(bundle_dir):
    """``(center_xy, top_z)`` of the raised standing pad, in trough-local coordinates."""
    for box in trough_boxes(bundle_dir):
        if "stand" in box["name"]:
            cx, cy, cz = (float(v) for v in box["center"])
            return (cx, cy), cz + float(box["size"][2]) / 2.0
    raise ValueError("%s/static/stage_boxes.json has no standing pad" % bundle_dir)


def robot_spawn(bundle_dir, root_height, yaw_deg, clearance=0.005, spawn_xy=None, surface_z=None):
    """Root pose for the robot so that it *stands on* the pad, facing ``yaw_deg``.

    ``ArticulationCfg.init_state.pos`` is the **root** position, not the foot position:
    the G1 config uses ``z = 0.9`` for a standing robot. Handing the pad surface height
    straight to ``init_state.pos`` buries the robot up to the hips and no policy can walk.

    ``spawn_xy`` / ``surface_z`` override the pad centre and the surface the robot stands
    on (e.g. ``spawn_xy=(0.0, -1.6), surface_z=0.14`` puts it inside the trough, on top of
    the pebble bed, so that it can wade along the 4 m trough instead of crossing it).
    """
    import math

    (cx, cy), top = standing_pad(bundle_dir)
    if spawn_xy is not None:
        cx, cy = float(spawn_xy[0]), float(spawn_xy[1])
    if surface_z is not None:
        top = float(surface_z)
    yaw = math.radians(yaw_deg)
    pos = (cx, cy, top + clearance + float(root_height))
    rot = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    return pos, rot


# ---------------------------------------------------------------- spawner
def _apply_robot_contact_offsets(stage, root_path, contact_offset, rest_offset):
    """Shrink the contact band of every collider under ``root_path``.

    A contact pair generates contacts within the **sum** of both contact offsets, so telling
    the pebbles to use 1 mm is pointless while the robot's feet still carry the 20 mm engine
    default. The robot is already spawned by the time this runs (the scene builds assets in
    cfg order and the pebble field is last), and it is still before ``sim.reset()``, which is
    the only moment PhysX picks the value up.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    prim_root = stage.GetPrimAtPath(root_path)
    if not prim_root or not prim_root.IsValid():
        print("[pebble] WARNING: robot prim %s not found" % root_path, flush=True)
        return 0
    count = 0
    for prim in Usd.PrimRange(prim_root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        if contact_offset and contact_offset > 0.0:
            api.CreateContactOffsetAttr(float(contact_offset))
        if rest_offset and rest_offset > 0.0:
            api.CreateRestOffsetAttr(float(rest_offset))
        count += 1
    print("[pebble] robot colliders: contact_offset set on %d prims under %s" % (count, root_path), flush=True)
    return count


def _spawn_pebble_field(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """``SpawnerCfg.func``: build the trough + pebbles under ``prim_path``."""
    import omni.usd
    from pxr import Gf, UsdGeom

    pl = loader_module(cfg.bundle_dir)
    layout = cfg.layout_id
    layout_id = layout if str(layout) == "random" else int(layout)

    info = pl.build_pebble_scene(
        cfg.bundle_dir,
        n_envs=1,
        layout_id=layout_id,
        mode=cfg.mode,
        prim_root=str(prim_path),
        env_spacing=0.0,
        make_invisible=cfg.hide,
        contact_offset=(cfg.contact_offset if cfg.contact_offset and cfg.contact_offset > 0.0 else None),
        rest_offset=(cfg.rest_offset if cfg.rest_offset and cfg.rest_offset > 0.0 else None),
        verbose=True,
    )
    # apply the transform while we are still *before* sim.reset(): PhysX consumes prim
    # transforms at reset, so this is free (doing it afterwards would be ignored).
    stage = omni.usd.get_context().get_stage()
    _set_translate(stage, str(prim_path), translation)
    if cfg.robot_contact_offset and cfg.robot_contact_offset > 0.0 and cfg.robot_prim_path:
        _apply_robot_contact_offsets(
            stage, cfg.robot_prim_path, cfg.robot_contact_offset, cfg.robot_rest_offset
        )

    _LAST_SCENE.clear()
    _LAST_SCENE.update(info)
    _LAST_SCENE["_loader"] = pl
    return stage.GetPrimAtPath(str(prim_path))


def _set_translate(stage, prim_path, translation):
    """Set the translate of an Xform prim, standardising its op stack.

    ``pebble_loader`` defines the root ``Xform`` without any xform op (so
    ``GetOrderedXformOps()[0]`` raises ``IndexError``) and ``XformPrimView`` insists on the
    canonical ``[translate, orient, scale]`` order, so let IsaacLab standardise it.
    """
    from isaaclab.sim.utils.transforms import standardize_xform_ops

    t = tuple(float(v) for v in (translation if translation is not None else (0.0, 0.0, 0.0)))
    standardize_xform_ops(stage.GetPrimAtPath(prim_path), translation=t)


def last_scene():
    """The scene dictionary produced by the spawner (``None`` if nothing was spawned)."""
    return dict(_LAST_SCENE) if _LAST_SCENE else None


_SPAWN_CFG_CLS = None


def spawn_cfg_class():
    """``SpawnerCfg`` subclass carrying the extra parameters, created lazily.

    The ``@configclass`` decorator is what regenerates ``__init__`` from the dataclass
    fields; without it the subclass silently keeps ``SpawnerCfg.__init__`` and every extra
    keyword argument raises ``TypeError``. It is built on demand so that importing this
    module does not pull in IsaacLab before ``AppLauncher`` has started the app.
    """
    global _SPAWN_CFG_CLS
    if _SPAWN_CFG_CLS is None:
        from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
        from isaaclab.utils import configclass

        @configclass
        class PebbleFieldSpawnCfg(SpawnerCfg):
            """Extra parameters for :func:`_spawn_pebble_field`."""

            bundle_dir: str = ""
            layout_id: str = "0"
            mode: str = "dynamic"
            hide: bool = False
            contact_offset: float = 0.0
            rest_offset: float = 0.0
            robot_prim_path: str = ""
            robot_contact_offset: float = 0.0
            robot_rest_offset: float = 0.0

        _SPAWN_CFG_CLS = PebbleFieldSpawnCfg
    return _SPAWN_CFG_CLS


def make_scene_asset(
    bundle_dir,
    layout_id="0",
    mode="dynamic",
    hide=False,
    contact_offset=0.0,
    rest_offset=0.0,
    robot_prim_path="",
    robot_contact_offset=0.0,
    robot_rest_offset=0.0,
    prim_path=DEFAULT_PRIM_PATH,
):
    """Build the ``SceneCfg`` entry that spawns the trough during scene construction."""
    from isaaclab.assets import AssetBaseCfg

    asset = AssetBaseCfg(
        prim_path=prim_path,
        spawn=spawn_cfg_class()(
            func=_spawn_pebble_field,
            bundle_dir=str(bundle_dir),
            layout_id=str(layout_id),
            mode=str(mode),
            hide=bool(hide),
            contact_offset=float(contact_offset or 0.0),
            rest_offset=float(rest_offset or 0.0),
            robot_prim_path=str(robot_prim_path),
            robot_contact_offset=float(robot_contact_offset or 0.0),
            robot_rest_offset=float(robot_rest_offset or 0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    return asset


# ---------------------------------------------------------------- post-reset checks
def realign_field(scene, origin, tolerance=1e-9):
    """Translate the field to ``origin`` *after* reset, for non-plane terrains.

    Only the USD transform is moved, so this is exact for visuals but **not** for collision:
    PhysX read the prim transforms at ``sim.reset()``. Use it as a diagnostic, not as a fix.
    """
    o = [float(v) for v in origin]
    if max(abs(v) for v in o) <= tolerance:
        return False
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    _set_translate(stage, scene["prim_root"], o)
    print(
        "[pebble] WARNING: non-zero terrain origin %s; field was translated in USD only "
        "(collision still sits at the spawner origin). Use a plane terrain." % (o,),
        flush=True,
    )
    return True


def summary(scene):
    """One-line, printable description of the spawned field."""
    return (
        "pebbles=%d layout=%s mode=%s fill_z=%.5f"
        % (scene["n_pebbles"], scene.get("layout_ids"), scene["mode"], scene["fill_z"])
    )


# ---------------------------------------------------------------- displacement report
def pebble_positions(scene):
    """(N, 3) world positions read straight from PhysX (dynamic mode only)."""
    view = scene["_loader"].make_pebble_view(scene)
    return scene["_loader"].view_positions(view).copy()


def report_displacement(scene, before, threshold_mm=2.0, label="pebbles"):
    """Print how far the pebble bed moved since ``before``.

    This is the assertion that dynamic mode actually *is* dynamic: with the engine-default
    20 mm contact offset the bed behaves like a rigid slab and this number stays at 0.
    """
    import numpy as np

    now = pebble_positions(scene)
    d = np.linalg.norm(now - before, axis=1) * 1000.0
    moved = int((d > threshold_mm).sum())
    print(
        "[pebble] %s displacement: moved>%.1fmm = %d/%d (%.1f%%), max=%.2f mm, mean=%.2f mm, p99=%.2f mm"
        % (label, threshold_mm, moved, d.size, 100.0 * moved / max(d.size, 1), d.max(), d.mean(), np.percentile(d, 99)),
        flush=True,
    )
    return d
