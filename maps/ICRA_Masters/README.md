# ICRA Masters — PF bridge maps

Placeholder maps for the dual-region (under/over) bridge stack on the PF side.

These are MIRRORS of the maps in `mppi_bringup/maps/ICRA_Masters/`. PF reads
the OVER map directly from disk (no `map_server` round-trip), so it needs a
local copy. After SLAM, copy the same .pgm + .yaml here.

The UNDER map continues to flow through `map_server` from
`particle_filter/config/localize.yaml` — set:
```
map_server:
  ros__parameters:
    map: 'ICRA_Masters/under_map'
```

For the OVER map, edit `particle_filter/config/localize.yaml`. PF resolves
this either as an absolute path (leading `/`) or relative to the
`particle_filter` install share's `maps/` dir. **Use the relative form for
portability between dev (cedric) and Jetson (nvidia):**
```
particle_filter:
  ros__parameters:
    over_map_yaml: 'ICRA_Masters/over_map.yaml'    # ← preferred
```

See `mppi_bringup/maps/ICRA_Masters/README.md` for the SLAM workflow.
