#!/usr/bin/env python3
"""Flip font OBJ files 180 degrees around center (5, 5)."""

import sys

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <file.obj> [file2.obj ...]")
    sys.exit(1)

for path in sys.argv[1:]:
    vertices = []
    faces = []
    comment = ""

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), parts[3]
                vertices.append((10 - x, 10 - y, z))
            elif line.startswith("f "):
                parts = line.split()
                # Reverse winding order to preserve face orientation
                faces.append((parts[1], parts[3], parts[2]))
            elif line.startswith("#"):
                comment = line

    with open(path, "w") as f:
        if comment:
            f.write(f"{comment}\n")
        for x, y, z in vertices:
            f.write(f"v {x:g} {y:g} {z}\n")
        for i0, i1, i2 in faces:
            f.write(f"f {i0} {i1} {i2}\n")

    print(f"Flipped {path}")
