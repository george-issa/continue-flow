#!/bin/sh
# Leave-families-out arm for the recipe of run_lines.sh (exact line spectra as targets): the same
# eight families as run_leave_out.sh are withheld.  Five lattice families and three line families
# leave the label bank, and the three line families leave the generator of broadened lines.
#
#   sh scripts/step3/run_leave_out_lines.sh [STEPS=60000]
#
# Writes out/step3/bank_lines_leave_out/, out/step3/F3L_<k>k_lo.pt, out/eval/exact_models_F3L_<k>k_lo/
# and, with seed 1 of the full recipe judged, out/step3/leave_out_F3L.{json,png}.
set -u
STEPS=${1:-60000}
N=F3L_$((STEPS / 1000))k_lo
B=out/step3/bank_lines_leave_out/label_bank_trusted.pt
VAE=out/step2/data_arm/D24_60k.pt
export PYTHONPATH=src
python scripts/step3/leave_out_bank.py out/step3/bank_lines/label_bank_merged.pt out/step3/bank_lines_leave_out \
    tb_honeycomb tb_square ssh_chain bcs falicov_kimball hubbard_dimer hubbard_ring4 holstein_atom \
    > out/step3/$N.bank.log 2>&1 || exit 1
python scripts/flow/train_flow.py --out out/step3/$N.pt --steps $STEPS --seed 1 --device mps \
    --smooth-bank $B --p-smooth 0.45 --p-exact 0.10 \
    --exact-families hubbard_atom hubbard_trimer anderson_cluster \
    --beta-choices 1 1.5 2 3 4 6 8 12 16 20 --noise-range 1e-5 3e-2 \
    --dtau-choices 0.02 0.05 0.1 0.125 0.2 0.25 0.5 --name step3-lines-$((STEPS / 1000))k-leave-out \
    > out/step3/$N.log 2>&1 || { echo "training $N failed"; exit 1; }
echo "$(date +%H:%M:%S) trained $N"
for setting in "1e-3 8" "1e-4 8" "1e-2 8" "1e-3 2" "1e-3 16"; do
    python scripts/eval/exact_models_judge.py out/eval/exact_models_$N out/step3/$N.pt $VAE $setting \
        >> out/eval/exact_models_$N.log 2>&1
done
echo "$(date +%H:%M:%S) judged $N"
SEEDS=""
for s in 2 3; do [ -d out/eval/exact_models_F3L_$((STEPS / 1000))k_s$s ] && SEEDS="$SEEDS F3L_$((STEPS / 1000))k_s$s"; done
python scripts/step3/leave_out_summary.py out/step3/leave_out_F3L $N F3L_$((STEPS / 1000))k_s1 $SEEDS \
    > out/step3/leave_out_F3L.log 2>&1
echo "$(date +%H:%M:%S) compared $N with the full recipe"
