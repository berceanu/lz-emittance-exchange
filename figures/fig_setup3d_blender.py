"""Blender script: 3D schematic of magnetised LWFA emittance exchange.

Run with:
  /Applications/Blender.app/Contents/MacOS/Blender --background \
    --python fig_setup3d_blender.py
"""

import math
import sys
from pathlib import Path

import bpy
import numpy as np


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for c in bpy.data.collections:
        if c.name != "Collection":
            bpy.data.collections.remove(c)


def smoothstep(x):
    s = np.clip(x, 0, 1)
    return s * s * (3 - 2 * s)


def make_trajectory_curve(n_pts=1500):
    """Create a curve object for the electron trajectory."""
    z = np.linspace(0, 10, n_pts)
    z_cross, width = 5.0, 2.5
    sig = smoothstep((z - z_cross + width / 2) / width)

    amp = 0.55
    freq = 6.0
    phase = 2 * np.pi * freq * z / 10

    x = amp * (1 - sig) * np.sin(phase)
    y = amp * sig * np.cos(phase)

    # Create curve
    curve_data = bpy.data.curves.new("trajectory", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = 0.055
    curve_data.bevel_resolution = 4

    spline = curve_data.splines.new("POLY")
    spline.points.add(n_pts - 1)

    for i in range(n_pts):
        spline.points[i].co = (z[i], x[i], y[i], 1)

    obj = bpy.data.objects.new("Trajectory", curve_data)
    bpy.context.collection.objects.link(obj)

    # Vertex color via material with color ramp driven by position
    mat = bpy.data.materials.new("TrajMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Shader
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.25
    bsdf.inputs["Emission Strength"].default_value = 0.5
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Color ramp driven by object Z position (propagation axis)
    tex_coord = nodes.new("ShaderNodeTexCoord")
    sep_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tex_coord.outputs["Object"], sep_xyz.inputs["Vector"])

    map_range = nodes.new("ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = 0
    map_range.inputs["From Max"].default_value = 10
    links.new(sep_xyz.outputs["X"], map_range.inputs["Value"])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.0, 0.15, 0.95, 1)  # strong blue
    ramp.color_ramp.elements.new(0.45)
    ramp.color_ramp.elements[1].color = (0.5, 0.05, 0.8, 1)   # violet
    ramp.color_ramp.elements.new(0.55)
    ramp.color_ramp.elements[2].color = (0.8, 0.05, 0.5, 1)
    ramp.color_ramp.elements[3].position = 1.0
    ramp.color_ramp.elements[3].color = (0.95, 0.15, 0.0, 1)  # strong red

    links.new(map_range.outputs["Result"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Emission Color"])

    obj.data.materials.append(mat)
    return obj


def make_solenoid(n_turns=18, radius=0.85, length=10.0):
    """Helical solenoid coil."""
    n_pts = n_turns * 80
    t = np.linspace(0, 2 * np.pi * n_turns, n_pts)
    z = t / (2 * np.pi * n_turns) * length
    x = radius * np.cos(t)
    y = radius * np.sin(t)

    curve_data = bpy.data.curves.new("solenoid", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = 0.018
    curve_data.bevel_resolution = 3

    spline = curve_data.splines.new("POLY")
    spline.points.add(n_pts - 1)
    for i in range(n_pts):
        spline.points[i].co = (z[i], x[i], y[i], 1)

    obj = bpy.data.objects.new("Solenoid", curve_data)
    bpy.context.collection.objects.link(obj)

    mat = bpy.data.materials.new("SolenoidMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.55, 0.55, 0.55, 1)
    bsdf.inputs["Metallic"].default_value = 0.7
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Alpha"].default_value = 0.7
    mat.blend_method = "BLEND" if hasattr(mat, "blend_method") else None
    obj.data.materials.append(mat)
    return obj


def make_capillary(radius=0.81, length=10.0):  # 95% of solenoid radius 0.85
    """Semi-transparent capillary cylinder."""
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=length,
        location=(5, 0, 0), rotation=(0, math.pi / 2, 0),
        vertices=32
    )
    obj = bpy.context.active_object
    obj.name = "Capillary"

    mat = bpy.data.materials.new("CapillaryMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.55, 0.82, 1.0, 1)
    bsdf.inputs["Alpha"].default_value = 0.18
    bsdf.inputs["Roughness"].default_value = 0.15
    bsdf.inputs["Transmission Weight"].default_value = 0.8
    obj.data.materials.append(mat)
    return obj


def make_crossing_ring(z_pos=5.0, radius=0.7):
    """Dashed ring at the crossing point xi*."""
    n_pts = 80
    t = np.linspace(0, 2 * np.pi, n_pts)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.full_like(t, z_pos)

    curve_data = bpy.data.curves.new("crossing_ring", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = 0.008
    curve_data.bevel_resolution = 2

    spline = curve_data.splines.new("POLY")
    spline.points.add(n_pts - 1)
    for i in range(n_pts):
        spline.points[i].co = (z[i], x[i], y[i], 1)

    obj = bpy.data.objects.new("CrossingRing", curve_data)
    bpy.context.collection.objects.link(obj)

    mat = bpy.data.materials.new("CrossingMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.4, 0.4, 0.4, 1)
    bsdf.inputs["Alpha"].default_value = 0.6
    obj.data.materials.append(mat)
    return obj


def make_laser_arrow():
    """Orange cone + cylinder for the laser."""
    # Shaft
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.08, depth=2.0,
        location=(-1.5, 0, 0), rotation=(0, math.pi / 2, 0)
    )
    shaft = bpy.context.active_object
    shaft.name = "LaserShaft"

    # Tip
    bpy.ops.mesh.primitive_cone_add(
        radius1=0.16, depth=0.4,
        location=(-0.3, 0, 0), rotation=(0, math.pi / 2, 0)
    )
    tip = bpy.context.active_object
    tip.name = "LaserTip"

    mat = bpy.data.materials.new("LaserMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.9, 0.62, 0, 1)
    bsdf.inputs["Emission Color"].default_value = (0.9, 0.62, 0, 1)
    bsdf.inputs["Emission Strength"].default_value = 0.3

    shaft.data.materials.append(mat)
    tip.data.materials.append(mat)


def make_b_arrow():
    """Dark red arrow for B-field."""
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.05, depth=1.0,
        location=(10.7, 0, 0), rotation=(0, math.pi / 2, 0)
    )
    shaft = bpy.context.active_object
    shaft.name = "BShaft"

    bpy.ops.mesh.primitive_cone_add(
        radius1=0.10, depth=0.25,
        location=(11.35, 0, 0), rotation=(0, math.pi / 2, 0)
    )
    tip = bpy.context.active_object
    tip.name = "BTip"

    mat = bpy.data.materials.new("BMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.5, 0.05, 0.05, 1)
    shaft.data.materials.append(mat)
    tip.data.materials.append(mat)


def setup_camera():
    bpy.ops.object.camera_add(
        location=(-3, -8, 7),
        rotation=(0, 0, 0)
    )
    cam = bpy.context.active_object
    cam.data.lens = 28
    cam.data.clip_end = 100
    bpy.context.scene.camera = cam

    # Track to centre of scene
    constraint = cam.constraints.new("TRACK_TO")
    empty = bpy.data.objects.new("CamTarget", None)
    empty.location = (5, 0, 0)
    bpy.context.collection.objects.link(empty)
    constraint.target = empty
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"


def setup_lighting():
    # Key light — from above-front
    bpy.ops.object.light_add(type="SUN", location=(5, -5, 8))
    sun = bpy.context.active_object
    sun.data.energy = 2.0
    sun.data.angle = math.radians(20)

    # Fill light — from the other side
    bpy.ops.object.light_add(type="AREA", location=(5, 5, 3))
    fill = bpy.context.active_object
    fill.data.energy = 80
    fill.data.size = 5

    # World background — pure white
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = 2.0


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 256
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 2400
    scene.render.resolution_y = 900
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


def main():
    clear_scene()

    make_trajectory_curve()
    make_solenoid()
    make_capillary()
    make_crossing_ring()
    make_b_arrow()
    setup_camera()
    setup_lighting()
    setup_render()

    out = str(Path(__file__).parent / "fig_setup3d")
    bpy.context.scene.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
