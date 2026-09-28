# Shared shell prologue for tools/crossbench/*.sh (sourced, not executed).
# Paths come from paths.py through tools/_release.py, so nothing is hardcoded here.
# PYTHON: interpreter of the environment the calling script needs (default: python on PATH).
XB_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$XB_HERE/../.." && pwd)
PY=${PYTHON:-python}
_xb_path () {
  "$PY" -c "import sys; sys.path.insert(0, '$REPO'); from tools._release import paths, cache_dir, out_path; print($1)"
}
NUSC_ROOT=${NUSC_ROOT:-$(_xb_path "paths.NUSC_ROOT")}
WAYMO_ROOT=${WAYMO_ROOT:-$(_xb_path "paths.WAYMO_ROOT")}
UPSTREAM_ROOT=${UPSTREAM_ROOT:-$(_xb_path "paths.UPSTREAM_ROOT")}
NUSC_WORK=${NUSC_WORK:-$(_xb_path "cache_dir('crossbench', 'nusc')")}
XB_OUT=${XB_OUT:-$(_xb_path "__import__('os').path.dirname(out_path('crossbench/README'))")}
# Released checkpoints (nuScenes: FCOS3D, PGD, CenterTrack e140, EPro-PnP-Det basic).
CKPT_DIR=${CKPT_DIR:-$UPSTREAM_ROOT/checkpoints}
