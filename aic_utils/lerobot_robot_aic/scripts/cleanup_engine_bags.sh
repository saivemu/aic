#!/bin/bash
# Watcher: continuously delete aic_engine's per-trial mcap bag dirs older
# than 60 seconds. Each bag is ~3 GB; without this watcher 100 trials would
# put 300 GB on disk. The 60s window leaves the active trial's bag untouched
# (engine writes for ~30s + 5s settle, then closes the bag and tier-2
# scoring reads it within seconds).
#
# Bags are written by the eval container running as root, but the bind-mount
# means they appear on the host with root ownership. To delete without sudo,
# we shell *into* the container (where we *are* root) and rm there.
#
# Usage:
#   ./cleanup_engine_bags.sh                    # uses $HOME/aic_results
#   ./cleanup_engine_bags.sh /path/inside/ctnr  # custom path
#   AIC_CTR=my_eval ./cleanup_engine_bags.sh    # override container name

set -u

DIR="${1:-${AIC_RESULTS_DIR:-$HOME/aic_results}}"
CONTAINER="${AIC_CTR:-aic_eval}"
INTERVAL_SEC=30
AGE_MIN=1  # delete bags older than 1 minute

# Ensure dir exists on host so the eval container's bind mount works.
mkdir -p "$DIR" 2>/dev/null || true

echo "[cleanup] container=$CONTAINER dir=$DIR age=${AGE_MIN}min interval=${INTERVAL_SEC}s"

while true; do
    deleted=$(docker exec "$CONTAINER" \
        find "$DIR" -mindepth 1 -maxdepth 1 -type d -name "bag_trial_*" -mmin "+${AGE_MIN}" -print -exec rm -rf {} + \
        2>/dev/null | wc -l)
    if (( deleted > 0 )); then
        free=$(df -h "$DIR" | tail -1 | awk '{print $4}')
        echo "[cleanup] $(date +%H:%M:%S) deleted $deleted stale bag dir(s); host free=$free"
    fi
    sleep "$INTERVAL_SEC"
done
