#!/bin/sh
# The bin-count judge on the three DQMC runs with known answers, then its figure.
#
#   sh scripts/dqmc/run_bin_count.sh FLOW.pt OUT [SUBSETS=6]
set -u
FLOW=$1
OUT=$2
K=${3:-6}
export PYTHONPATH=src
mkdir -p $OUT
for sim in hubbard_dimer_U4.00_b8.00_dt0.050-1 hubbard_ring4_U4.00_b8.00_dt0.050-1 hubbard_dimer_U4.00_b2.00_dt0.050-1; do
    python scripts/dqmc/bin_count_judge.py run $OUT $FLOW out/dqmc/$sim 4.0 $K > $OUT/run_$sim.log 2>&1 &
done
wait
python scripts/dqmc/bin_count_judge.py fig $OUT
