#!/usr/bin/env bash
# Download PDFs for Module 3 exercises
# Run: bash download_data.sh

set -e

UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

download() {
    local dir="$1" file="$2" url="$3"
    mkdir -p "$dir"
    local path="$dir/$file"
    if [ -f "$path" ]; then
        echo "  ✓ Already exists: $file"
        return
    fi
    echo "  ↓ Downloading: $file"
    curl -sL -A "$UA" -o "$path" "$url"
    # Verify it's actually a PDF
    if ! file "$path" | grep -q "PDF"; then
        echo "  ✗ FAILED (not a PDF): $file"
        rm -f "$path"
        return 1
    fi
    local size=$(ls -lh "$path" | awk '{print $5}')
    local pages=$(pdfinfo "$path" 2>/dev/null | grep Pages | awk '{print $2}')
    echo "  ✓ OK: $file ($size, ${pages:-?} pages)"
}

echo "=== Climate Action Plans ==="
CAP="data/climate-action-plans"
download "$CAP" "oakland-ecap-2020.pdf" \
    "https://cao-94612.s3.amazonaws.com/documents/Oakland-ECAP-07-24.pdf"
download "$CAP" "portland-cap-2015.pdf" \
    "https://www.portland.gov/sites/default/files/2019-07/cap-2015_june30-2015_web_0.pdf"
download "$CAP" "austin-climate-equity-2021.pdf" \
    "https://www.austintexas.gov/sites/default/files/files/Sustainability/Climate%20Equity%20Plan/Climate%20Equity%20Plan%20Full%20Document__FINAL.pdf"
download "$CAP" "seattle-cap-2013.pdf" \
    "https://www.seattle.gov/documents/Departments/Environment/ClimateChange/2013_CAP_20130612.pdf"
download "$CAP" "annarbor-a2zero-2020.pdf" \
    "https://www.a2gov.org/departments/sustainability/Documents/A2ZERO%20Climate%20Action%20Plan%20_3.0.pdf"
download "$CAP" "ca-scoping-plan-2022.pdf" \
    "https://ww2.arb.ca.gov/sites/default/files/2023-04/2022-sp.pdf"
download "$CAP" "ipcc-ar6-wg3-spm.pdf" \
    "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_SummaryForPolicymakers.pdf"

echo ""
echo "=== Corporate Sustainability Reports ==="
CORP="data/corporate-sustainability"
download "$CORP" "apple-env-2024.pdf" \
    "https://www.apple.com/environment/pdf/Apple_Environmental_Progress_Report_2024.pdf"
download "$CORP" "google-env-2024.pdf" \
    "https://www.gstatic.com/gumdrop/sustainability/google-2024-environmental-report.pdf"
download "$CORP" "amazon-sustainability-2023.pdf" \
    "https://sustainability.aboutamazon.com/2023-amazon-sustainability-report.pdf"
download "$CORP" "bp-sustainability-2023.pdf" \
    "https://www.bp.com/content/dam/bp/business-sites/en/global/corporate/pdfs/sustainability/group-reports/bp-sustainability-report-2023.pdf"

echo ""
echo "=== Energy Planning Reports ==="
ENERGY="data/utility-irps"
download "$ENERGY" "doe-solar-futures-2021.pdf" \
    "https://www.energy.gov/sites/default/files/2021-09/Solar%20Futures%20Study.pdf"
download "$ENERGY" "eia-aeo-2023.pdf" \
    "https://www.eia.gov/outlooks/aeo/pdf/AEO2023_Narrative.pdf"
download "$ENERGY" "nrel-electrification-futures.pdf" \
    "https://www.nrel.gov/docs/fy18osti/71500.pdf"
download "$ENERGY" "nrel-re-futures-vol1.pdf" \
    "https://www.nrel.gov/docs/fy12osti/52409-1.pdf"

echo ""
echo "=== Download Complete ==="
echo ""

# Summary
total=0
for dir in "$CAP" "$CORP" "$ENERGY"; do
    count=$(ls -1 "$dir"/*.pdf 2>/dev/null | wc -l)
    total=$((total + count))
    echo "  $dir: $count PDFs"
done
echo "  Total: $total PDFs"
