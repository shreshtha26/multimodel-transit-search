#!/usr/bin/env bash
set -Eeuo pipefail

die() {
  echo
  echo "ERROR: $*" >&2
  echo "The consolidation stopped rather than guessing." >&2
  exit 1
}

note() {
  echo
  echo "================================================================"
  echo "$*"
  echo "================================================================"
}

need() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

need git
need python
need rsync

CHAR_ROOT="$(pwd -P)"
[[ "$(git branch --show-current 2>/dev/null || true)" == "characterization-v2" ]] || \
  die "Run this from the trident-characterization-v2 worktree on branch characterization-v2."

COMMON_GIT_DIR="$(git rev-parse --git-common-dir)"
COMMON_GIT_DIR="$(cd "$COMMON_GIT_DIR" && pwd -P)"
CANONICAL_ROOT="$(dirname "$COMMON_GIT_DIR")"

[[ -d "$CANONICAL_ROOT/.git" ]] || \
  die "Could not identify canonical trident root: $CANONICAL_ROOT"

[[ "$(git -C "$CANONICAL_ROOT" branch --show-current)" == "clean-injection-benchmark" ]] || \
  die "Expected canonical trident worktree on clean-injection-benchmark."

CHAR_REGISTERED=no
CANONICAL_REGISTERED=no
while IFS= read -r line; do
  [[ "$line" == "worktree $CHAR_ROOT" ]] && CHAR_REGISTERED=yes
  [[ "$line" == "worktree $CANONICAL_ROOT" ]] && CANONICAL_REGISTERED=yes
done < <(git worktree list --porcelain)

[[ "$CHAR_REGISTERED" == yes ]] || die "Current directory is not the registered characterization worktree."
[[ "$CANONICAL_REGISTERED" == yes ]] || die "Canonical trident directory is not a registered worktree."

note "1/12 Fetch and verify repository topology"

git fetch origin --prune

CHAR_BASE="$(git rev-parse characterization-v2)"
CLEAN_BASE="$(git rev-parse clean-injection-benchmark)"
MAIN_BASE="$(git rev-parse main)"
REMOTE_CLEAN="$(git rev-parse origin/clean-injection-benchmark)"
REMOTE_MAIN="$(git rev-parse origin/main)"

[[ "$CHAR_BASE" == "$CLEAN_BASE" ]] || \
  die "characterization-v2 and clean-injection-benchmark no longer start from the same commit."

git merge-base --is-ancestor main clean-injection-benchmark || \
  die "main is no longer an ancestor of clean-injection-benchmark."

[[ "$REMOTE_CLEAN" == "$CLEAN_BASE" ]] || \
  die "origin/clean-injection-benchmark changed since inspection."

[[ "$REMOTE_MAIN" == "$MAIN_BASE" ]] || \
  die "origin/main changed since inspection."

echo "Shared benchmark/characterization base: $CLEAN_BASE"
echo "Current main:                         $MAIN_BASE"

note "2/12 Verify dirty files are exactly the expected work"

# Characterization worktree: these are the known v2 code/results.
CHAR_STATUS="$(mktemp)"
git -C "$CHAR_ROOT" status --porcelain > "$CHAR_STATUS"

python - "$CHAR_STATUS" <<'PY'
import sys
from pathlib import Path

status = Path(sys.argv[1]).read_text().splitlines()

allowed_exact = {
    ".gitignore",
    "scripts/consolidate_repository_to_main.sh",
    "scripts/run_light_curve_characterization.py",
    "scripts/run_multistar_challenger_benchmark.py",
    "src/adaptive_transit/noise_models/characterization.py",
    "scripts/build_characterization_validation_report.py",
    "scripts/build_stellar_variability_profiles.py",
    "scripts/build_validation_visual_panels.py",
    "scripts/plot_variability_diagnostics.py",
    "scripts/run_characterization_validation10.py",
    "src/adaptive_transit/noise_models/stellar_variability.py",
    "tests/test_stellar_variability.py",
    "outputs/target_selection/kepler_characterization_validation10.csv",
}

allowed_prefixes = (
    "outputs/experiments/characterization/",
    "outputs/experiments/characterization_population40/",
    "outputs/experiments/characterization_validation/",
    "outputs/experiments/characterization_validation10/",
    "outputs/experiments/characterization_validation50/",
)

bad = []
for raw in status:
    if not raw.strip():
        continue
    path = raw[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1].strip()

    if path in allowed_exact:
        continue
    if path.endswith(".patch") and "/" not in path:
        continue
    if any(path.startswith(prefix) for prefix in allowed_prefixes):
        continue
    bad.append(raw)

if bad:
    print("Unexpected changes in characterization worktree:")
    for item in bad:
        print("  " + item)
    raise SystemExit(1)
PY
rm -f "$CHAR_STATUS"

# Canonical benchmark worktree: these are the local pieces discovered by the
# first safety run.  Anything else makes this script stop.
CANON_STATUS="$(mktemp)"
git -C "$CANONICAL_ROOT" status --porcelain > "$CANON_STATUS"

python - "$CANON_STATUS" <<'PY'
import sys
from pathlib import Path

status = Path(sys.argv[1]).read_text().splitlines()

allowed_exact = {
    "src/adaptive_transit/noise_models/characterization.py",
    "streamlit_transit_demo.py",
}

allowed_prefixes = (
    "outputs/experiments/multistar_challenger_benchmark/clean_q5_50star/star_calibration/",
)

bad = []
for raw in status:
    if not raw.strip():
        continue
    path = raw[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1].strip()

    if path in allowed_exact:
        continue
    if path.endswith(".patch") and "/" not in path:
        continue
    if any(path.startswith(prefix) for prefix in allowed_prefixes):
        continue
    bad.append(raw)

if bad:
    print("Unexpected changes in canonical trident worktree:")
    for item in bad:
        print("  " + item)
    raise SystemExit(1)
PY
rm -f "$CANON_STATUS"

echo "Both worktrees contain only expected consolidation inputs."

note "3/12 Create external safety backup of BOTH worktrees"

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_ROOT="$(dirname "$CANONICAL_ROOT")/trident-consolidation-backup-$STAMP"
mkdir -p "$BACKUP_ROOT"

mkdir -p "$BACKUP_ROOT/git"
git log --graph --decorate --oneline --all -80 > "$BACKUP_ROOT/git/history_before.txt"
git branch -a -vv > "$BACKUP_ROOT/git/branches_before.txt"
git worktree list --porcelain > "$BACKUP_ROOT/git/worktrees_before.txt"

mkdir -p "$BACKUP_ROOT/characterization"
git -C "$CHAR_ROOT" diff > "$BACKUP_ROOT/characterization/tracked_changes.patch"
git -C "$CHAR_ROOT" status --short > "$BACKUP_ROOT/characterization/status_before.txt"

mkdir -p "$BACKUP_ROOT/canonical"
git -C "$CANONICAL_ROOT" diff > "$BACKUP_ROOT/canonical/tracked_changes.patch"
git -C "$CANONICAL_ROOT" status --short > "$BACKUP_ROOT/canonical/status_before.txt"

# Explicitly preserve the overlapping file from each root before we resolve it.
mkdir -p "$BACKUP_ROOT/overlap"
cp -p \
  "$CHAR_ROOT/src/adaptive_transit/noise_models/characterization.py" \
  "$BACKUP_ROOT/overlap/characterization_from_validated_v2.py"
cp -p \
  "$CANONICAL_ROOT/src/adaptive_transit/noise_models/characterization.py" \
  "$BACKUP_ROOT/overlap/characterization_from_original_trident.py"

# Preserve the original-root Streamlit work independently.
cp -p \
  "$CANONICAL_ROOT/streamlit_transit_demo.py" \
  "$BACKUP_ROOT/canonical/streamlit_transit_demo.py"

# Preserve untracked characterization source/test/manifest files.
CHAR_BACKUP_PATHS=(
  scripts/build_characterization_validation_report.py
  scripts/build_stellar_variability_profiles.py
  scripts/build_validation_visual_panels.py
  scripts/plot_variability_diagnostics.py
  scripts/run_characterization_validation10.py
  src/adaptive_transit/noise_models/stellar_variability.py
  tests/test_stellar_variability.py
  outputs/target_selection/kepler_characterization_validation10.csv
)

for relative in "${CHAR_BACKUP_PATHS[@]}"; do
  if [[ -e "$CHAR_ROOT/$relative" ]]; then
    mkdir -p "$BACKUP_ROOT/characterization/$(dirname "$relative")"
    cp -p "$CHAR_ROOT/$relative" "$BACKUP_ROOT/characterization/$relative"
  fi
done

# Archive downloaded patch files from both worktrees. They are implementation
# provenance, not source code, and should not remain in the final repo root.
mkdir -p "$BACKUP_ROOT/downloaded_patches/characterization"
mkdir -p "$BACKUP_ROOT/downloaded_patches/canonical"

shopt -s nullglob
for patch in "$CHAR_ROOT"/*.patch; do
  cp -p "$patch" "$BACKUP_ROOT/downloaded_patches/characterization/"
done
for patch in "$CANONICAL_ROOT"/*.patch; do
  cp -p "$patch" "$BACKUP_ROOT/downloaded_patches/canonical/"
done
shopt -u nullglob

echo "External safety backup created:"
echo "  $BACKUP_ROOT"

note "4/12 Commit original trident local code separately"

# Generated calibration products stay local and untracked/ignored. Patch files
# are moved to the external backup after integration.
git -C "$CANONICAL_ROOT" add -- \
  src/adaptive_transit/noise_models/characterization.py \
  streamlit_transit_demo.py

if git -C "$CANONICAL_ROOT" diff --cached --quiet; then
  echo "No canonical code changes required a commit."
else
  echo "Canonical code being preserved:"
  git -C "$CANONICAL_ROOT" diff --cached --name-status
  git -C "$CANONICAL_ROOT" commit -m \
    "Preserve clean benchmark UI and local characterization updates"
fi

CANON_LOCAL_COMMIT="$(git -C "$CANONICAL_ROOT" rev-parse HEAD)"

note "5/12 Prepare ignore rules and commit validated characterization v2"

GITIGNORE="$CHAR_ROOT/.gitignore"
touch "$GITIGNORE"

append_ignore() {
  grep -Fqx "$1" "$GITIGNORE" || echo "$1" >> "$GITIGNORE"
}

append_ignore ""
append_ignore "# Local characterization experiment products"
append_ignore "/outputs/experiments/characterization/"
append_ignore "/outputs/experiments/characterization_population40/"
append_ignore "/outputs/experiments/characterization_validation/"
append_ignore "/outputs/experiments/characterization_validation10/"
append_ignore "/outputs/experiments/characterization_validation50/"
append_ignore "/outputs/experiments/multistar_challenger_benchmark/clean_q5_50star/star_calibration/"
append_ignore "/*.patch"

CHAR_COMMIT_PATHS=(
  .gitignore
  scripts/consolidate_repository_to_main.sh
  scripts/run_light_curve_characterization.py
  scripts/run_multistar_challenger_benchmark.py
  src/adaptive_transit/noise_models/characterization.py
  scripts/build_characterization_validation_report.py
  scripts/build_stellar_variability_profiles.py
  scripts/build_validation_visual_panels.py
  scripts/plot_variability_diagnostics.py
  scripts/run_characterization_validation10.py
  src/adaptive_transit/noise_models/stellar_variability.py
  tests/test_stellar_variability.py
  outputs/target_selection/kepler_characterization_validation10.csv
)

for relative in "${CHAR_COMMIT_PATHS[@]}"; do
  [[ -e "$CHAR_ROOT/$relative" ]] || die "Expected characterization file missing: $relative"
done

python -m py_compile \
  "$CHAR_ROOT/scripts/run_light_curve_characterization.py" \
  "$CHAR_ROOT/scripts/run_multistar_challenger_benchmark.py" \
  "$CHAR_ROOT/scripts/build_characterization_validation_report.py" \
  "$CHAR_ROOT/scripts/build_stellar_variability_profiles.py" \
  "$CHAR_ROOT/scripts/build_validation_visual_panels.py" \
  "$CHAR_ROOT/scripts/plot_variability_diagnostics.py" \
  "$CHAR_ROOT/scripts/run_characterization_validation10.py" \
  "$CHAR_ROOT/src/adaptive_transit/noise_models/characterization.py" \
  "$CHAR_ROOT/src/adaptive_transit/noise_models/stellar_variability.py"

if python -c 'import pytest' >/dev/null 2>&1; then
  (
    cd "$CHAR_ROOT"
    python -m pytest -q tests/test_stellar_variability.py
  )
else
  echo "pytest not installed; py_compile validation passed."
fi

git -C "$CHAR_ROOT" add -- "${CHAR_COMMIT_PATHS[@]}"

if git -C "$CHAR_ROOT" diff --cached --name-only | \
   grep -E '^(outputs/experiments/|[^/]+\.patch$)' >/dev/null; then
  git -C "$CHAR_ROOT" diff --cached --name-only
  die "Generated experiment outputs or downloaded patch files were staged."
fi

echo "Validated characterization code being committed:"
git -C "$CHAR_ROOT" diff --cached --name-status

git -C "$CHAR_ROOT" commit -m "Integrate validated statistical characterization v2"
CHAR_FINAL="$(git -C "$CHAR_ROOT" rev-parse HEAD)"

note "6/12 Archive all important pre-consolidation states with tags"

TAG_MAIN="archive/main-pre-consolidation-$STAMP"
TAG_CLEAN_REMOTE="archive/clean-benchmark-remote-$STAMP"
TAG_CLEAN_LOCAL="archive/clean-benchmark-local-$STAMP"
TAG_CHAR="archive/characterization-v2-final-$STAMP"

git -C "$CANONICAL_ROOT" tag -a "$TAG_MAIN" "$MAIN_BASE" \
  -m "Main before benchmark and characterization consolidation"

git -C "$CANONICAL_ROOT" tag -a "$TAG_CLEAN_REMOTE" "$CLEAN_BASE" \
  -m "Published clean benchmark before local consolidation"

git -C "$CANONICAL_ROOT" tag -a "$TAG_CLEAN_LOCAL" "$CANON_LOCAL_COMMIT" \
  -m "Clean benchmark including preserved local trident changes"

git -C "$CANONICAL_ROOT" tag -a "$TAG_CHAR" "$CHAR_FINAL" \
  -m "Validated statistical characterization v2"

note "7/12 Merge the two lines of work"

cd "$CANONICAL_ROOT"

# The benchmark branch now has its small local commit; characterization-v2 has
# the validated v2 commit. A normal merge preserves both histories.
set +e
git merge --no-ff --no-commit characterization-v2
MERGE_RC=$?
set -e

if [[ "$MERGE_RC" -ne 0 ]]; then
  CONFLICTS="$(git diff --name-only --diff-filter=U)"

  # The only expected overlap is characterization.py.  For that file the
  # validated characterization-v2 implementation is deliberately the final
  # source of truth. The original-root version is safely backed up above.
  EXPECTED_CONFLICT="src/adaptive_transit/noise_models/characterization.py"

  if [[ "$CONFLICTS" == "$EXPECTED_CONFLICT" ]]; then
    echo
    echo "Expected overlap detected:"
    echo "  $EXPECTED_CONFLICT"
    echo "Resolution policy: keep the validated characterization-v2 version."

    git checkout --theirs -- "$EXPECTED_CONFLICT"
    git add -- "$EXPECTED_CONFLICT"
  else
    echo "Unexpected merge conflict(s):"
    echo "$CONFLICTS"
    git merge --abort || true
    die "Automatic consolidation will not guess how to resolve these conflicts."
  fi
fi

# Ensure there are no remaining unmerged files.
if [[ -n "$(git diff --name-only --diff-filter=U)" ]]; then
  git diff --name-only --diff-filter=U
  die "Unresolved merge files remain."
fi

git commit -m \
  "Consolidate clean benchmark and validated statistical characterization v2"

INTEGRATED_HEAD="$(git rev-parse HEAD)"

note "8/12 Copy characterization experiment outputs into the canonical root"

OUTPUT_PATHS=(
  outputs/experiments/characterization
  outputs/experiments/characterization_population40
  outputs/experiments/characterization_validation
  outputs/experiments/characterization_validation10
  outputs/experiments/characterization_validation50
)

for relative in "${OUTPUT_PATHS[@]}"; do
  src="$CHAR_ROOT/$relative"
  dst="$CANONICAL_ROOT/$relative"

  [[ -d "$src" ]] || continue

  if [[ -e "$dst" ]]; then
    backup_dst="$BACKUP_ROOT/canonical_preexisting_outputs/$relative"
    mkdir -p "$(dirname "$backup_dst")"
    mv "$dst" "$backup_dst"
  fi

  mkdir -p "$(dirname "$dst")"
  rsync -a "$src/" "$dst/"
done

# The star_calibration directory already lives in the canonical worktree; the
# ignore rule added by characterization-v2 now prevents it from dirtying Git.
if [[ -d \
  "$CANONICAL_ROOT/outputs/experiments/multistar_challenger_benchmark/clean_q5_50star/star_calibration" ]]; then
  echo "Preserved canonical star_calibration outputs."
fi

note "9/12 Validate the consolidated code before touching GitHub"

python -m py_compile \
  scripts/run_light_curve_characterization.py \
  scripts/run_multistar_challenger_benchmark.py \
  scripts/build_characterization_validation_report.py \
  scripts/build_stellar_variability_profiles.py \
  scripts/build_validation_visual_panels.py \
  scripts/plot_variability_diagnostics.py \
  scripts/run_characterization_validation10.py \
  src/adaptive_transit/noise_models/characterization.py \
  src/adaptive_transit/noise_models/stellar_variability.py \
  streamlit_transit_demo.py

if python -c 'import pytest' >/dev/null 2>&1; then
  python -m pytest -q tests/test_stellar_variability.py
fi

# Confirm the merge retained the canonical Streamlit file and the validated
# characterization implementation.
cmp -s \
  "$CANONICAL_ROOT/src/adaptive_transit/noise_models/characterization.py" \
  "$BACKUP_ROOT/overlap/characterization_from_validated_v2.py" || \
  die "Final characterization.py is not byte-identical to validated v2."

[[ -f "$CANONICAL_ROOT/streamlit_transit_demo.py" ]] || \
  die "streamlit_transit_demo.py was lost during consolidation."

note "10/12 Advance main and push the consolidated repository"

# main is an ancestor of the integrated history, so this remains a fast-forward.
git switch main
git merge --ff-only clean-injection-benchmark

[[ "$(git rev-parse HEAD)" == "$INTEGRATED_HEAD" ]] || \
  die "main did not fast-forward to the integrated commit."

git push origin \
  "$TAG_MAIN" \
  "$TAG_CLEAN_REMOTE" \
  "$TAG_CLEAN_LOCAL" \
  "$TAG_CHAR"

git push origin main

git fetch origin
[[ "$(git rev-parse origin/main)" == "$INTEGRATED_HEAD" ]] || \
  die "origin/main does not match the integrated local main. Old branches were NOT deleted."

note "11/12 Remove obsolete branch/worktree clutter"

# Only now, after origin/main is verified, delete the old remote benchmark branch.
git push origin --delete clean-injection-benchmark

# Move root patch files to the external backup rather than leaving them in the
# final canonical project directory.
shopt -s nullglob
for patch in "$CANONICAL_ROOT"/*.patch; do
  mv "$patch" "$BACKUP_ROOT/downloaded_patches/canonical/"
done
shopt -u nullglob

# The characterization worktree's code is now committed, merged and pushed;
# its local outputs were copied to the canonical root and its patches backed up.
git worktree remove --force "$CHAR_ROOT"
git worktree prune

git branch -d characterization-v2
git branch -d clean-injection-benchmark

note "12/12 Final verification"

[[ "$(git branch --show-current)" == "main" ]] || die "Final worktree is not on main."
[[ "$(git rev-parse HEAD)" == "$INTEGRATED_HEAD" ]] || die "Final main HEAD changed unexpectedly."

echo
echo "FINAL WORKTREES"
git worktree list

echo
echo "FINAL BRANCHES"
git branch -a -vv

echo
echo "FINAL STATUS"
git status --short

echo
echo "REMOTE BRANCHES"
git ls-remote --heads origin

echo
echo "ARCHIVE TAGS"
printf '  %s\n' \
  "$TAG_MAIN" \
  "$TAG_CLEAN_REMOTE" \
  "$TAG_CLEAN_LOCAL" \
  "$TAG_CHAR"

echo
echo "External safety backup:"
echo "  $BACKUP_ROOT"

echo
echo "Canonical project from now on:"
echo "  $CANONICAL_ROOT"
echo "  branch: main"

echo
echo "CONSOLIDATION COMPLETE"
