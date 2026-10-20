# -*- coding: utf-8 -*-
"""export_mujoco.py —— 把鹅卵石床导出成 MuJoCo 资产（32 个形状 .obj + 一份可 include 的 XML）。

为什么不是"转成一个 .obj"：MuJoCo 里 .obj 只能当**静态视觉/碰撞**用（一个 mesh geom），
要么整床变成一块刚性壳（石头不能动），要么把几万颗塞进一个 mesh 里做三角面碰撞（更慢且
无法推开）。要让鹅卵石像真石头一样被脚推开，必须"每颗一个 body + 一个 mesh geom"，
所以这里导出的是 **池子里的 32 个形状各一个 .obj** + 一颗石头一条 <body>（用 scale 复用到
各自的实际尺寸）。真要整床的合并 .obj（当刚性壳/给别的引擎看），加 --merged-obj。

用法（在 pebble_scene/ 下）:
    python export_mujoco.py --config config_mujoco_basin.json --out mj_out
    python export_mujoco.py --bundle <bundle_dir> --out mj_out --offset -5.5 5.5
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_pebbles as gp  # noqa: E402


def unit_shapes(cfg, n_shapes=None):
    """按 bundle 的种子/噪声重新造出形状池（与 bundle 里的 rock_pool.obj 一致）。"""
    rng = np.random.default_rng(int(cfg["seed"]))
    n = int(n_shapes or cfg["pebbles"]["pool_size"])
    pool = [gp.build_unit_shape(rng, int(cfg["pebbles"]["n_dirs"]),
                                float(cfg["pebbles"]["noise_amp"]), int(cfg["pebbles"]["n_waves"]))
            for _ in range(n)]
    return pool


def write_unit_obj(path, verts, faces):
    with io.open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("# unit cobble, circumscribed radius 1.0 (scale = semi-axes in metres)\n")
        for v in verts:
            f.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        for tri in faces:
            f.write("f %d %d %d\n" % (tri[0] + 1, tri[1] + 1, tri[2] + 1))


def write_merged_obj(path, pos, rot, semi, shapes):
    """整床合并 OBJ（静态用；不参与动力学）。"""
    with io.open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("# merged cobble bed (static visual only)\n")
        off = 0
        for k in range(len(pos)):
            v, faces = shapes[int(semi[k, 3])]
            R = rot.as_matrix()[k]
            pts = (v * semi[k, :3]) @ R.T + pos[k]
            for p in pts:
                f.write("v %.6f %.6f %.6f\n" % (p[0], p[1], p[2]))
            for tri in faces:
                f.write("f %d %d %d\n" % (tri[0] + 1 + off, tri[1] + 1 + off, tri[2] + 1 + off))
            off += len(v)


def _generate(cfg, args, size_scale=1.0):
    cfg = gp.deep_update(gp.default_config(), cfg)
    geo = dict(lx=float(args.footprint[0]), ly=float(args.footprint[1]), depth=float(args.depth))
    nominal_top = args.nominal_top
    if nominal_top is None:
        nominal_top = float((cfg.get("pebble_bed") or {}).get("nominal_top") or 0.0) or None
    if nominal_top is None:
        nominal_top = float((cfg.get("pebble_bed") or {}).get("bed_depth") or 0.15)
    if size_scale != 1.0:
        for lc in gp.pebble_layers(cfg["pebbles"]):
            pass
        base_sr = list(cfg["pebbles"]["size_range"])
        cfg["pebbles"]["size_range"] = [base_sr[0] * size_scale, base_sr[1] * size_scale]
        for lc in (cfg["pebbles"].get("layers") or []):
            sr = list(lc.get("size_range") or base_sr)
            lc["size_range"] = [sr[0] * size_scale, sr[1] * size_scale]
    rng = np.random.default_rng(int(cfg["seed"]) + 7)
    shapes = unit_shapes(cfg)
    pool_vol = np.array([p[2] for p in shapes])
    lay = gp.generate_layout(rng, cfg, geo, pool_vol, None, float(nominal_top))
    return cfg, lay


def build_field(args):
    cfg = gp.load_config(args.config) if args.config else None
    if args.bundle:
        man = json.load(io.open(os.path.join(args.bundle, "manifest.json"), encoding="utf-8"))
        cfg = gp.load_config(os.path.join(args.bundle, "layout_config.json"))
        z = np.load(os.path.join(args.bundle, "layouts.npz"))
        off = np.concatenate([[0], np.cumsum(z["n_per_layout"])]).astype(int)
        li = int(args.layout)
        s, e = int(off[li]), int(off[li + 1])
        pos, quat, scale = z["pos"][s:e].copy(), z["quat"][s:e].copy(), z["scale"][s:e].copy()
        sid = z["shape_id"][s:e]
        fill_z = float(man["coordinate_frame"]["bulk_fill_top_z"])
        return cfg, pos, quat, scale, sid, fill_z

    cfg, lay = _generate(cfg, args)
    if args.max_pebbles and lay["info"]["n"] > args.max_pebbles:
        # 直接删石头会把床面弄秃（覆盖率掉一半），所以按目标颗数放大粒径重生成：
        # 颗数 ∝ 1/粒径²，两轮牛顿迭代足够收敛。
        scale = 1.0
        for _ in range(3):
            n_now = lay["info"]["n"]
            if abs(n_now - args.max_pebbles) <= 0.08 * args.max_pebbles:
                break
            scale *= float(np.sqrt(n_now / float(args.max_pebbles))) ** 0.9
            cfg, lay = _generate(gp.load_config(args.config), args, size_scale=scale)
        print("[mujoco] rescaled cobbles by %.2fx to hit <= %d bodies" % (scale, args.max_pebbles))
    info = lay["info"]
    print("[mujoco] generated n=%d bed=%.3f m top_max=%.3f fill_z=%.3f exposed=%.2f%% wide=%.2f%%"
          % (info["n"], info["bed_depth"], info["top_max"], info["fill_z"],
             100 * info["exposed_plate_frac"], 100 * info["exposed_plate_frac_wide"]))
    return (cfg, lay["pos"], lay["quat"], lay["scale"], lay["shape_id"], info["fill_z"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="场景配置 json（生成模式）")
    ap.add_argument("--bundle", help="已有 bundle 目录（导入模式，优先于 --config）")
    ap.add_argument("--layout", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--footprint", nargs=2, type=float, default=[2.84, 2.84])
    ap.add_argument("--depth", type=float, default=0.30)
    ap.add_argument("--nominal-top", type=float, default=None,
                    help="床顶的绝对 z；缺省取配置里的 pebble_bed.nominal_top（再缺省=bed_depth）")
    ap.add_argument("--offset", nargs=2, type=float, default=[0.0, 0.0],
                    help="整块床在 MuJoCo 世界里的 XY 平移")
    ap.add_argument("--z-offset", type=float, default=0.0, help="地面顶面对应的 z")
    ap.add_argument("--max-pebbles", type=int, default=0, help="超过这个数就按空间均匀抽稀（0=不抽）")
    ap.add_argument("--joints", choices=["free", "slide"], default="free")
    ap.add_argument("--collision", choices=["mesh", "ellipsoid"], default="ellipsoid",
                    help="mesh=真形状碰撞(贵,几千颗会爆内存); ellipsoid=视觉用 mesh、碰撞用椭球")
    ap.add_argument("--ellipsoid-inflate", type=float, default=1.02,
                    help="碰撞椭球相对名义 semi-axes 的放大（形状噪声 7%% 的补偿）")
    ap.add_argument("--slide-range", type=float, default=0.03)
    ap.add_argument("--merged-obj", action="store_true")
    ap.add_argument("--name", default="pebble_field")
    args = ap.parse_args()

    cfg, pos, quat, scale, sid, fill_z = build_field(args)
    shapes = unit_shapes(cfg)
    os.makedirs(args.out, exist_ok=True)
    mesh_dir = os.path.join(args.out, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    # 形状文件（只写实际用到的）
    used = sorted(set(int(v) for v in sid))
    for k in used:
        v, faces, _ = shapes[k]
        write_unit_obj(os.path.join(mesh_dir, "pebble_s%02d.obj" % k), v, faces)
    if args.merged_obj:
        semi4 = np.concatenate([scale, sid.reshape(-1, 1)], axis=1)
        write_merged_obj(os.path.join(args.out, "%s_merged.obj" % args.name), pos, 
                         Rotation.from_quat(np.stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]], axis=1)),
                         semi4, [(s[0], s[1]) for s in shapes])

    # 每颗石头一个 mesh asset（file + scale 组合；MuJoCo 的 scale 是 mesh 级而不是 geom 级）。
    # 注意 MJCF 的 include 会把子元素插到**父元素**里，所以资产和本体必须分成两份：
    #   *_assets.xml  → 放在 <mujoco> 顶层
    #   *.xml         → 放在 <worldbody> 里
    alines = ['<mujocoinclude>', '  <asset>']
    for i in range(len(pos)):
        alines.append('    <mesh name="m%05d" file="meshes/pebble_s%02d.obj" scale="%.5f %.5f %.5f"/>'
                      % (i, int(sid[i]), scale[i, 0], scale[i, 1], scale[i, 2]))
    alines.append('  </asset>')
    alines.append('</mujocoinclude>')
    assets_path = os.path.join(args.out, "%s_assets.xml" % args.name)
    io.open(assets_path, "w", encoding="utf-8", newline="\n").write("\n".join(alines) + "\n")

    lines = ['<mujocoinclude>']
    lines.append('  <!-- bodies only: include INSIDE <worldbody>, and the *_assets.xml at <mujoco> level -->')
    for i in range(len(pos)):
        p = pos[i] + np.array([args.offset[0], args.offset[1], args.z_offset])
        # MJCF quat 顺序是 w x y z；bundle 存的就是 wxyz
        lines.append('    <body name="pebble_%05d" pos="%.5f %.5f %.5f" quat="%.6f %.6f %.6f %.6f">'
                     % (i, p[0], p[1], p[2], quat[i, 0], quat[i, 1], quat[i, 2], quat[i, 3]))
        if args.joints == "free":
            lines.append('      <freejoint/>')
        else:
            r = args.slide_range
            for ax in ("1 0 0", "0 1 0", "0 0 1"):
                lines.append('      <joint type="slide" axis="%s" range="-%.3f %.3f" damping="0.02"/>'
                             % (ax, r, r))
        if args.collision == "mesh":
            lines.append('      <geom type="mesh" mesh="m%05d" density="2600" condim="4" '
                         'friction="1.3 0.03 0.001" solref="0.004 1" solimp="0.95 0.99 0.001"/>' % i)
        else:
            k = float(args.ellipsoid_inflate)
            # 视觉：真形状 mesh（不参与碰撞），质量/惯量也由它给
            lines.append('      <geom type="mesh" mesh="m%05d" density="2600" contype="0" conaffinity="0"/>' % i)
            # 碰撞：解析椭球，便宜得多；density=0 避免质量翻倍
            lines.append('      <geom name="c%05d" type="ellipsoid" size="%.5f %.5f %.5f" density="0" '
                         'condim="4" friction="1.3 0.03 0.001" solref="0.004 1" solimp="0.95 0.99 0.001"/>'
                         % (i, scale[i, 0] * k, scale[i, 1] * k, scale[i, 2] * k))
        lines.append('    </body>')
    lines.append('</mujocoinclude>')
    xml_path = os.path.join(args.out, "%s.xml" % args.name)
    io.open(xml_path, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")

    print("[mujoco] wrote %s + %s (%d bodies, %d shapes, joints=%s, collision=%s)"
          % (xml_path, os.path.basename(assets_path), len(pos), len(used), args.joints, args.collision))
    print("[mujoco] footprint %.2f x %.2f m, offset (%.2f, %.2f), fill_z=%.4f"
          % (args.footprint[0], args.footprint[1], args.offset[0], args.offset[1], fill_z))
    print("         include at <mujoco> level: <include file=\"%s\"/>"
          % os.path.basename(assets_path))
    print("         include inside <worldbody>: <include file=\"%s\"/>" % os.path.basename(xml_path))


if __name__ == "__main__":
    main()
