#!/bin/bash
df -h /Data1 /Data2 /Data3 /Data4 2>&1 | grep -v '^Filesystem'
echo "=== writability check ==="
for d in /Data1 /Data2 /Data3 /Data4; do
  echo "== $d =="
  ls "$d" 2>&1 | head -3
  mkdir -p "$d/ee_24126016" 2>/dev/null && touch "$d/ee_24126016/.wtest" 2>/dev/null && echo "WRITABLE" || echo "NOT-WRITABLE"
done
