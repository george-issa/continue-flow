#!/bin/sh
# The basic-spectra judge at four settings, then its figures.
#
#   sh scripts/step3/run_basic_spectra.sh FLOW.pt OUT [PER=10]
set -u
FLOW=$1
OUT=$2
PER=${3:-10}
export PYTHONPATH=src
mkdir -p $OUT
for setting in "8 1e-3" "8 1e-4" "2 1e-3" "16 1e-3"; do
    tag=$(echo $setting | tr ' ' '_')
    python scripts/step3/basic_spectra_judge.py run $OUT $FLOW $setting $PER > $OUT/run_$tag.log 2>&1 &
done
wait
python scripts/step3/basic_spectra_judge.py fig $OUT
