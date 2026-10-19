from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


SCENE = Path(__file__).parents[1] / "sim2sim" / "assets" / "scene" / "scene_29dof_terrain_selector.xml"
GENERATOR = Path(__file__).parents[1] / "sim2sim" / "generate_pebble_terrain.py"
GENERATED_PEBBLES = SCENE.parent / "generated" / "pebble_field_seed_20261019.xml"
TOP_VIEW = SCENE.parent / "terrain_selector_top_view.svg"


def generate_pebbles(output: Path) -> None:
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )


def pebble_bodies(include_path: Path) -> list[ET.Element]:
    root = ET.parse(include_path).getroot()
    return list(root.findall("./worldbody/body"))


def cast_down(model, data, x: float, y: float) -> tuple[float, str]:
    geom_id = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(
        model,
        data,
        np.array([x, y, 2.0]),
        np.array([0.0, 0.0, -1.0]),
        None,
        True,
        -1,
        geom_id,
    )
    return distance, model.geom(geom_id[0]).name


def test_combined_course_has_center_spawn_and_all_four_terrain_directions():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    data.qpos[:3] = [30.0, 30.0, 3.0]
    mujoco.mj_forward(model, data)

    assert model.geom("outer_ground").type == mujoco.mjtGeom.mjGEOM_PLANE
    central_ground = model.geom("ground_front")
    assert np.allclose(central_ground.size[:2], [6.0, 6.875])
    assert central_ground.pos[2] + central_ground.size[2] == 0.0

    gap_floor = model.geom("gap_floor")
    gap_platform = model.geom("gap_central_platform")
    assert np.allclose(gap_floor.size[:2], [1.65, 1.65])
    assert np.allclose(gap_platform.size[:2], [1.35, 1.35])

    spawn_distance, spawn_name = cast_down(model, data, 0.0, 0.0)
    forward_step_distance, forward_step_name = cast_down(model, data, 2.15, 0.0)
    rear_step_distance, rear_step_name = cast_down(model, data, -2.15, 0.0)
    gap_floor_distance, gap_floor_name = cast_down(model, data, 1.5, 5.5)
    gap_platform_distance, gap_platform_name = cast_down(model, data, 0.0, 5.5)
    slope_base_distance, slope_base_name = cast_down(model, data, 0.0, -2.6)
    slope_peak_distance, slope_peak_name = cast_down(model, data, 0.01, -5.5)

    assert spawn_name == "ground_front"
    assert np.isclose(spawn_distance, 2.0)
    assert forward_step_name == "forward_step_up_1"
    assert np.isclose(forward_step_distance, 1.925)
    assert rear_step_name == "rear_step_down_1"
    assert np.isclose(rear_step_distance, 2.075)
    assert gap_floor_name == "gap_floor"
    assert np.isclose(gap_floor_distance, 2.5)
    assert gap_platform_name == "gap_central_platform"
    assert np.isclose(gap_platform_distance, 2.0)
    assert slope_base_name == "slope_pyramid_15deg"
    assert np.isclose(slope_base_distance, 2.0 - 0.1 * np.tan(np.deg2rad(15.0)))
    assert slope_peak_name == "slope_pyramid_15deg"
    assert np.isclose(slope_peak_distance, 2.0 - 2.99 * np.tan(np.deg2rad(15.0)))


def test_slope_pyramid_mesh_has_no_coplanar_base_faces():
    mesh = ET.parse(SCENE).find(".//mesh[@name='slope_pyramid_mesh']")

    assert mesh is not None
    assert len(mesh.attrib["face"].split()) == 12


def test_pebble_generator_is_deterministic_and_records_its_parameters(tmp_path):
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"

    generate_pebbles(first)
    generate_pebbles(second)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == GENERATED_PEBBLES.read_bytes()
    contents = first.read_text()
    assert "seed=20261019" in contents
    assert "count=784" in contents
    assert "layers=4" in contents
    assert "three translation-only slide joints" in contents
    assert "MuJoCo native ellipsoid" in contents


def test_pebble_include_has_dynamic_translation_only_ellipsoids_in_the_basin(tmp_path):
    include_path = tmp_path / "pebbles.xml"
    generate_pebbles(include_path)
    pebbles = pebble_bodies(include_path)

    assert len(pebbles) == 784
    assert not ET.parse(include_path).findall(".//freejoint")

    layer_centres = (0.032, 0.096, 0.160, 0.224)
    layer_counts = {centre: 0 for centre in layer_centres}
    pebble_bounds = []
    for pebble in pebbles:
        assert pebble.attrib["name"].startswith("pebble_")
        joints = pebble.findall("joint")
        assert len(joints) == 3
        assert [joint.attrib["type"] for joint in joints] == ["slide", "slide", "slide"]
        assert [joint.attrib["axis"] for joint in joints] == ["1 0 0", "0 1 0", "0 0 1"]
        assert all(axis in joint.attrib["name"] for axis, joint in zip("xyz", joints, strict=True))
        assert all(joint.attrib["limited"] == "true" for joint in joints)

        geom = pebble.find("geom")
        assert geom is not None
        assert geom.attrib["type"] == "ellipsoid"
        assert "density" in geom.attrib or pebble.find("inertial") is not None
        assert geom.attrib["contype"] == "1"
        assert geom.attrib["conaffinity"] == "1"
        assert float(geom.attrib["friction"].split()[0]) >= 1.0
        assert geom.attrib["solref"] == "0.004 1"
        assert tuple(map(float, geom.attrib["solimp"].split()[:3])) == (0.95, 0.99, 0.001)

        x, y, z = map(float, pebble.attrib["pos"].split())
        radius_x, radius_y, radius_z = map(float, geom.attrib["size"].split())
        assert 0.055 <= radius_x <= 0.085
        assert 0.055 <= radius_y <= 0.085
        assert 0.020 <= radius_z <= 0.030
        assert -6.92 <= x - radius_x <= x + radius_x <= -4.08
        assert 4.08 <= y - radius_y <= y + radius_y <= 6.92
        x_range = tuple(map(float, joints[0].attrib["range"].split()))
        y_range = tuple(map(float, joints[1].attrib["range"].split()))
        z_range = tuple(map(float, joints[2].attrib["range"].split()))
        assert -6.92 <= x + x_range[0] - radius_x <= x + x_range[1] + radius_x <= -4.08
        assert 4.08 <= y + y_range[0] - radius_y <= y + y_range[1] + radius_y <= 6.92
        assert 0.0 <= z + z_range[0] - radius_z <= z + z_range[1] + radius_z <= 0.30
        matching_centres = [centre for centre in layer_centres if np.isclose(z, centre)]
        assert len(matching_centres) == 1
        layer_counts[matching_centres[0]] += 1
        pebble_bounds.append((np.array([x, y, z]) - np.array([radius_x, radius_y, radius_z]), np.array([x, y, z]) + np.array([radius_x, radius_y, radius_z])))

    assert layer_counts == {centre: 196 for centre in layer_centres}
    for first_index, (first_low, first_high) in enumerate(pebble_bounds):
        for second_low, second_high in pebble_bounds[first_index + 1 :]:
            assert np.any(first_high < second_low) or np.any(second_high < first_low)


def test_pebble_rendering_is_disabled_but_the_scene_still_loads():
    # disable-pebble-rendering switch: the generated field include and the
    # basin block stay in the scene as XML comments, so either can be restored
    # by removing its comment delimiters. The generator contract itself is
    # covered by the tests above, which run the generator directly.
    scene_text = SCENE.read_text()
    assert "<!-- Pebble field disabled temporarily to reduce rendering load." in scene_text
    assert "<!-- Pebble basin disabled temporarily to reduce rendering load." in scene_text
    root = ET.parse(SCENE).getroot()
    active_includes = [include.attrib.get("file", "") for include in root.iter("include")]
    assert not any("generated/pebble_field" in file for file in active_includes)
    active_geoms = {geom.attrib.get("name") for geom in root.iter("geom")}
    assert not any(name and name.startswith("pebble_basin") for name in active_geoms)

    assert GENERATED_PEBBLES.is_file()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    assert np.isclose(model.opt.timestep, 0.002)

    geom_names = [model.geom(index).name for index in range(model.ngeom)]
    assert "pebble_platform" not in geom_names
    assert not any(name.startswith("pebble_basin") for name in geom_names)
    pebble_body_ids = [
        body_id
        for body_id in range(model.nbody)
        if model.body(body_id).name.startswith("pebble_")
    ]
    assert pebble_body_ids == []
    pebble_joint_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if model.body(model.jnt_bodyid[joint_id]).name.startswith("pebble_")
    ]
    assert pebble_joint_ids == []

    # The approach strips surrounding the disabled basin stay active so the
    # course ground remains continuous around x=[-7,-4], y=[4,7].
    for name in (
        "ground_gap_west_left",
        "ground_gap_west_right",
        "ground_gap_west_south",
        "ground_gap_west_north",
        "ground_gap_east",
    ):
        assert name in geom_names

    mujoco.mj_forward(model, data)
    assert np.isfinite(data.qpos).all()


def test_top_view_describes_the_dynamic_pebble_basin_at_the_course_coordinates():
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    root = ET.parse(TOP_VIEW).getroot()
    basin = root.find(".//svg:rect[@id='pebble_basin_top_view']", namespace)
    labels = {text.text for text in root.findall(".//svg:text", namespace)}

    assert basin is not None
    assert basin.attrib["x"] == "105"
    assert basin.attrib["y"] == "105"
    assert basin.attrib["width"] == "105"
    assert basin.attrib["height"] == "105"
    assert "Dynamic pebble basin" in labels
    assert "4 layers; dynamic translation-only pebbles settle and translate" in labels
    assert "Static walls; 0.30 m outer one-step approaches" in labels

    expected_walls = {
        "pebble_basin_wall_west_top_view": (105.0, 107.8, 2.8, 99.4),
        "pebble_basin_wall_east_top_view": (207.2, 107.8, 2.8, 99.4),
        "pebble_basin_wall_south_top_view": (105.0, 207.2, 105.0, 2.8),
        "pebble_basin_wall_north_top_view": (105.0, 105.0, 105.0, 2.8),
    }
    expected_steps = {
        "pebble_basin_step_west_top_view": (94.5, 115.5, 10.5, 84.0),
        "pebble_basin_step_east_top_view": (210.0, 115.5, 10.5, 84.0),
        "pebble_basin_step_south_top_view": (105.0, 210.0, 105.0, 10.5),
        "pebble_basin_step_north_top_view": (105.0, 94.5, 105.0, 10.5),
    }
    for element_id, expected_box in expected_walls.items():
        wall = root.find(f".//svg:rect[@id='{element_id}']", namespace)
        assert wall is not None
        assert np.allclose([float(wall.attrib[key]) for key in ("x", "y", "width", "height")], expected_box)
    for element_id, expected_box in expected_steps.items():
        step = root.find(f".//svg:rect[@id='{element_id}']", namespace)
        assert step is not None
        assert np.allclose([float(step.attrib[key]) for key in ("x", "y", "width", "height")], expected_box)
