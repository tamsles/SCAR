#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

if [[ ( -n "${PJM_JOBID:-}" || "${PJM_ENVIRONMENT:-}" == "BATCH" ) && \
      "${IWPU_ALLOW_BATCH_PREFETCH:-0}" != "1" ]]; then
    echo "Run this network-only prefetch on a Wisteria login node." >&2
    exit 2
fi
if ! command -v curl >/dev/null || ! command -v unzip >/dev/null || \
   ! command -v flock >/dev/null; then
    echo "curl, unzip, and flock are required." >&2
    exit 2
fi

CACHE_DIR="${TABLESHIFT_CACHE:-${REMOTE_ROOT}/data/tableshift_cache}/2018/1-Year"
BASE_URL="https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year"
mkdir -p "${CACHE_DIR}"
exec 9>"${CACHE_DIR}/.acs-household-prefetch.lock"
if ! flock -n 9; then
    echo "Another ACS household prefetch is active." >&2
    exit 3
fi

states=(
    AL:01 AK:02 AZ:04 AR:05 CA:06 CO:08 CT:09 DE:10 FL:12 GA:13 HI:15
    ID:16 IL:17 IN:18 IA:19 KS:20 KY:21 LA:22 ME:23 MD:24 MA:25 MI:26
    MN:27 MS:28 MO:29 MT:30 NE:31 NV:32 NH:33 NJ:34 NM:35 NY:36 NC:37
    ND:38 OH:39 OK:40 OR:41 PA:42 RI:44 SC:45 SD:46 TN:47 TX:48 UT:49
    VT:50 VA:51 WA:53 WV:54 WI:55 WY:56 PR:72
)

completed=0
for item in "${states[@]}"; do
    state="${item%%:*}"
    code="${item##*:}"
    lowercase="${state,,}"
    csv_name="psam_h${code}.csv"
    destination="${CACHE_DIR}/${csv_name}"
    if [[ -s "${destination}" ]]; then
        echo "Reusing ${csv_name}"
        completed=$((completed + 1))
        continue
    fi

    url="${BASE_URL}/csv_h${lowercase}.zip"
    archive="${CACHE_DIR}/.${csv_name}.zip.partial"
    extracted="${CACHE_DIR}/.${csv_name}.partial"
    success=0
    for attempt in 1 2 3 4 5 6; do
        echo "Downloading ${state} household data (attempt ${attempt}/6)"
        if curl --fail --location --http1.1 --silent --show-error \
            --connect-timeout 30 --max-time 900 \
            --retry 8 --retry-delay 10 --retry-max-time 600 \
            --user-agent "iwpu-reproduction/1.0 academic-research" \
            --output "${archive}" "${url}" && \
           unzip -tqq "${archive}" && \
           unzip -p "${archive}" "${csv_name}" > "${extracted}" && \
           [[ -s "${extracted}" ]]; then
            mv "${extracted}" "${destination}"
            rm -f "${archive}"
            success=1
            completed=$((completed + 1))
            echo "Ready ${csv_name}: $(stat -c '%s' "${destination}") bytes"
            break
        fi
        echo "Invalid or incomplete response for ${state}; backing off." >&2
        sleep $((attempt * 20))
    done
    if [[ "${success}" -ne 1 ]]; then
        echo "Failed to prepare ${csv_name} after bounded retries." >&2
        exit 4
    fi
    # Avoid triggering the Census/Cloudflare burst-rate protection.
    sleep "${ACS_PREFETCH_PAUSE_SECONDS:-5}"
done

if [[ "${completed}" -ne "${#states[@]}" ]]; then
    echo "ACS household completeness mismatch: ${completed}/${#states[@]}" >&2
    exit 5
fi
echo "ACS household cache complete: ${completed} state/territory files"
