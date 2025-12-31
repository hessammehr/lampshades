import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import re
    import numpy as np
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon
    import trimesh
    import networkx
    import scipy
    import seaborn as sns

    from matplotlib import pyplot as plt
    import marimo as mo

    sns.set_theme("talk", "ticks", font="Arial", font_scale=1.0, rc={"svg.fonttype": "none"})
    return ET, Polygon, mo, np, plt, re, trimesh


@app.cell
def _(mo, np, plt):
    def r(theta):
        # r = cos (5θ) + 0.2cos (9θ) + 0.05cos (200θ) + 4
        return np.cos(5 * theta) + 0.2 * np.cos(9 * theta) + 0.05 * np.cos(200 * theta) + 4.0


    def profile(d):
        "Returns the scale factor for layer where d is normalised distance to top of shape."
        return np.sqrt(d) * (0.2 + np.abs(0.2 + 0.5 * np.sin(d * np.pi)))


    thetas = np.arange(0, 2 * np.pi, 0.01)
    ds = np.linspace(0, 1, 100)
    rs = r(thetas)
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)
    xys = np.stack([xs, ys], axis=-1)
    f, (a1, a2) = plt.subplots(ncols=2, figsize=(9, 4))
    a1.plot(xs, ys)
    a2.plot(ds, profile(ds))
    mo.mpl.interactive(plt.gcf())
    return profile, xys


@app.cell
def _(ET, np, re):
    def load_svg_path_points(svg_path: str) -> np.ndarray:
        """
        Extracts the first non-empty <path d="..."> and returns an (N,2) array
        of XY points in SVG user units.
        Assumes the path is already discretised (lots of L commands), which matches your file.
        """
        root = ET.parse(svg_path).getroot()

        d = None
        for el in root.iter():
            if el.tag.split("}")[-1] == "path":
                cand = el.attrib.get("d", "")
                if len(cand) > 0:
                    d = cand
                    break
        if d is None:
            raise ValueError("No non-empty <path d='...'> found in the SVG.")

        # Extract all numbers; the file is "M x y L x y L x y ...", so numeric stream is x0,y0,x1,y1,...
        nums = list(map(float, re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", d)))
        if len(nums) % 2 != 0:
            raise ValueError("Odd number of coordinates extracted from SVG path.")
        pts = np.array(nums, dtype=float).reshape(-1, 2)

        # Remove consecutive duplicates and remove explicit closing point if present
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.hstack([[True], diffs > 1e-9])
        pts = pts[keep]
        if np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
            pts = pts[:-1]

        return pts
    return


@app.cell
def _(Polygon, np, profile, trimesh, xys):
    def polygon_area_centroid(pts: np.ndarray) -> tuple[float, np.ndarray]:
        x = pts[:, 0]
        y = pts[:, 1]
        x1 = np.roll(x, -1)
        y1 = np.roll(y, -1)
        a = 0.5 * np.sum(x * y1 - x1 * y)
        cx = (1.0 / (6.0 * a)) * np.sum((x + x1) * (x * y1 - x1 * y))
        cy = (1.0 / (6.0 * a)) * np.sum((y + y1) * (x * y1 - x1 * y))
        return a, np.array([cx, cy], dtype=float)


    def resample_closed_polyline(pts: np.ndarray, n: int) -> np.ndarray:
        """
        Uniform arclength resampling of a closed polyline.
        pts: (N,2) without duplicated end point.
        Returns (n,2).
        """
        # segment lengths
        seg = np.linalg.norm(np.diff(np.vstack([pts, pts[0]]), axis=0), axis=1)
        s = np.hstack([[0.0], np.cumsum(seg)])
        total = s[-1]
        target = np.linspace(0.0, total, n + 1)[:-1]

        # interpolate along the polyline
        out = np.empty((n, 2), dtype=float)
        j = 0
        pts_closed = np.vstack([pts, pts[0]])
        for i, t in enumerate(target):
            while not (s[j] <= t < s[j + 1]):
                j += 1
                if j >= len(seg):
                    j = len(seg) - 1
                    break
            u = (t - s[j]) / (s[j + 1] - s[j] + 1e-15)
            out[i] = (1 - u) * pts_closed[j] + u * pts_closed[j + 1]
        return out


    def rotate_xy(xy: np.ndarray, theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=float)
        return xy @ R.T


    def twisted_shell_mesh(
        outer_xy: np.ndarray,
        inner_xy: np.ndarray | None,
        height_mm: float,
        n_layers: int,
        total_twist_rad: float,
    ) -> trimesh.Trimesh:
        """
        Generates a watertight mesh:
        - outer surface: stacked rotated rings connected by quads (triangulated)
        - optional inner surface (reverse orientation) and stitched rims at top/bottom

        outer_xy and inner_xy are (N,2) arrays, implicitly closed, centred about (0,0).
        """

        def ring_vertices(profile_xy: np.ndarray) -> np.ndarray:
            N = profile_xy.shape[0]
            M = n_layers
            zs = np.linspace(0.0, height_mm, M)
            thetas = np.linspace(0.0, total_twist_rad, M)
            verts = np.empty((M * N, 3), dtype=float)
            for i, (z, th) in enumerate(zip(zs, thetas)):
                distance = (M - i - 1) / (M - 1)
                xy_rot = rotate_xy(profile_xy, th)
                verts[i * N : (i + 1) * N, 0:2] = xy_rot * profile(distance)
                verts[i * N : (i + 1) * N, 2] = z
            return verts

        def side_faces(N: int, M: int, offset: int, flip: bool) -> list[list[int]]:
            faces = []
            for i in range(M - 1):
                base0 = offset + i * N
                base1 = offset + (i + 1) * N
                for j in range(N):
                    jn = (j + 1) % N
                    a = base0 + j
                    b = base0 + jn
                    c = base1 + jn
                    d = base1 + j
                    if not flip:
                        faces.append([a, b, c])
                        faces.append([a, c, d])
                    else:
                        # reverse winding
                        faces.append([a, c, b])
                        faces.append([a, d, c])
            return faces

        outer_xy = np.asarray(outer_xy, dtype=float)
        N = outer_xy.shape[0]
        M = n_layers

        verts_outer = ring_vertices(outer_xy)
        faces = side_faces(N=N, M=M, offset=0, flip=False)
        verts = verts_outer

        if inner_xy is not None:
            inner_xy = np.asarray(inner_xy, dtype=float)
            if inner_xy.shape[0] != N:
                raise ValueError("Inner and outer profiles must have the same point count after resampling.")
            verts_inner = ring_vertices(inner_xy)

            offset_inner = len(verts)
            verts = np.vstack([verts, verts_inner])
            faces += side_faces(N=N, M=M, offset=offset_inner, flip=True)

            # stitch bottom rim (z=0): outer ring 0 to inner ring 0
            for j in range(N):
                jn = (j + 1) % N
                o0 = j
                o1 = jn
                i0 = offset_inner + j
                i1 = offset_inner + jn
                faces.append([o0, i1, i0])
                faces.append([o0, o1, i1])

            # stitch top rim (z=H): outer ring M-1 to inner ring M-1
            top_outer = (M - 1) * N
            top_inner = offset_inner + (M - 1) * N
            for j in range(N):
                jn = (j + 1) % N
                o0 = top_outer + j
                o1 = top_outer + jn
                i0 = top_inner + j
                i1 = top_inner + jn
                faces.append([o0, i0, i1])
                faces.append([o0, i1, o1])

        mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces, dtype=int), process=False)
        mesh.merge_vertices()
        mesh.fix_normals()
        mesh.process(validate=True)
        return mesh


    def main():
        svg_path = "desmos-graph.svg"

        # --- user parameters ---
        height_mm = 200.0
        total_twist_turns = 0.1
        n_layers = 100  # increase for smoother twist; decrease for fewer triangles
        n_profile_points = 500  # controls ripple fidelity and triangle count
        target_outer_diameter_mm = 220.0  # scale the SVG to this OD
        wall_thickness_mm = 1.2  # set None for a single surface
        join_style = 2  # 1=round, 2=mitre, 3=bevel (Shapely)

        # --- load & centre profile ---
        area, centroid = polygon_area_centroid(xys)
        pts = xys - centroid

        # Ensure outer polygon is CCW for consistent normals
        if area < 0:
            pts = pts[::-1].copy()

        # --- scale to target diameter ---
        r = np.max(np.linalg.norm(pts, axis=1))
        scale = (0.5 * target_outer_diameter_mm) / r
        pts *= scale

        # --- resample to control point count ---
        outer = resample_closed_polyline(pts, n_profile_points)

        inner = None
        if wall_thickness_mm is not None and wall_thickness_mm > 0:
            poly = Polygon(outer)
            # Shapely buffer with negative distance creates an inward offset.
            inner_poly = poly.buffer(-wall_thickness_mm, join_style=join_style)

            if inner_poly.is_empty:
                raise ValueError("Inward offset collapsed the profile; reduce wall_thickness_mm or simplify the outline.")

            # If buffer returns multiple polygons, take the largest by area.
            if inner_poly.geom_type == "MultiPolygon":
                inner_poly = max(inner_poly.geoms, key=lambda g: g.area)

            inner_coords = np.array(inner_poly.exterior.coords[:-1], dtype=float)  # drop duplicated closure
            # resample inner to match point count
            inner = resample_closed_polyline(inner_coords, n_profile_points)

        total_twist_rad = 2.0 * np.pi * total_twist_turns
        mesh = twisted_shell_mesh(
            outer_xy=outer,
            inner_xy=inner,
            height_mm=height_mm,
            n_layers=n_layers,
            total_twist_rad=total_twist_rad,
        )

        print("Watertight:", mesh.is_watertight)
        print("Triangles:", len(mesh.faces))
        return mesh
    return (main,)


@app.cell
def _(main, mo):
    mesh = main()
    btn = mo.ui.button(label="Save STL", on_click=lambda _: mesh.export("lampshade.stl"))
    return btn, mesh


@app.cell
def _(btn, mesh, mo):
    mo.vstack([mesh.show(), btn])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
