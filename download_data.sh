#!/usr/bin/env bash
# Download the Module 3 corpus: tech company environmental reports and validation data.
#
#   bash download_data.sh                  # all five companies (~500 MB)
#   bash download_data.sh google microsoft # just these companies
#
# The list of reports, with their URLs and the year each report's data covers, is in reports.csv.

UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
failed=0

download() {
    local path="$1" url="$2"
    if [ -f "$path" ]; then
        echo "  ✓ already have $path"
        return
    fi
    mkdir -p "$(dirname "$path")"
    curl -sL -A "$UA" --max-time 600 -o "$path.tmp" "$url"
    # Websites sometimes return an HTML error page instead of the file
    local ok=0
    if [[ "$path" == *.pdf ]]; then
        head -c 5 "$path.tmp" | grep -q "%PDF" && ok=1
    else
        [ -s "$path.tmp" ] && ok=1
    fi
    if [ "$ok" -eq 1 ]; then
        mv "$path.tmp" "$path"
        echo "  ✓ $path ($(du -h --apparent-size "$path" | cut -f1))"
    else
        rm -f "$path.tmp"
        echo "  ✗ FAILED: $path  ($url)"
        failed=$((failed + 1))
    fi
}

echo "=== Environmental reports ==="
while IFS=, read -r company report data_year url; do
    if [ $# -gt 0 ] && [[ ! " $* " == *" $company "* ]]; then
        continue
    fi
    download "data/reports/$company/$company-$report.pdf" "$url"
done < <(tail -n +2 reports.csv)

echo ""
echo "=== Validation data ==="
download "data/validation/sbti-companies.xlsx" \
    "https://files.sciencebasedtargets.org/production/files/companies-excel.xlsx"
download "data/validation/newclimate-ccrm2025-tech.pdf" \
    "https://web.archive.org/web/20250627104745id_/https://newclimate.org/sites/default/files/2025-06/newclimate-publication-ccrm2025-tech-Jun25.pdf"

echo ""
echo "Reports downloaded: $(ls data/reports/*/*.pdf 2>/dev/null | wc -l)"
[ "$failed" -eq 0 ] || echo "Some downloads failed. The exercises work with whatever subset you have."
