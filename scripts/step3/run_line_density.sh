#!/bin/sh
# The line-density judge at three settings, eight line counts each, then its figure.
#
#   sh scripts/step3/run_line_density.sh FLOW.pt OUT [PER=12]
set -u
FLOW=$1
OUT=$2
PER=${3:-12}
export PYTHONPATH=src
mkdir -p $OUT
for setting in "8 1e-3" "8 1e-4" "2 1e-3"; do
    tag=$(echo $setting | tr ' ' '_')
    (for N in 4 8 16 32 64 128 256 512; do
        python scripts/step3/line_density_judge.py run $OUT $FLOW $setting $N $PER
    done > $OUT/run_$tag.log 2>&1) &
done
wait
python scripts/step3/line_density_judge.py fig $OUT
