#!/usr/bin/env python3
"""Convert font data from objects.json into individual OBJ files."""

import json
import os

with open("objects.json") as f:
    data = json.load(f)

font = data["font"]
all_vertices = font["vertices"]  # list of [x, y] pairs
characters = font["characters"]  # list of {"char": "A", "indices": [...]}

os.makedirs("models/font", exist_ok=True)

for entry in characters:
    char = entry["char"]
    indices = entry["indices"]

    # Collect only the vertices used by this character
    used = sorted(set(indices))
    old_to_new = {old: new + 1 for new, old in enumerate(used)}  # OBJ is 1-indexed

    # Use the character as filename, handle special chars
    if char == " ":
        continue  # skip space
    filename = f"models/font/{char}.obj"
    if char in "/\\:*?\"<>|":
        print(f"Skipping '{char}' (bad filename character)")
        continue

    with open(filename, "w") as obj:
        obj.write(f"# Character: {char}\n")

        # Write vertices (2D -> 3D with z=0)
        for old_idx in used:
            x, y = all_vertices[old_idx]
            obj.write(f"v {x} {y} 0\n")

        # Write faces (triangles, groups of 3 indices)
        for i in range(0, len(indices), 3):
            i0 = old_to_new[indices[i]]
            i1 = old_to_new[indices[i + 1]]
            i2 = old_to_new[indices[i + 2]]
            obj.write(f"f {i0} {i1} {i2}\n")

    print(f"Wrote {filename} ({len(used)} vertices, {len(indices) // 3} triangles)")
