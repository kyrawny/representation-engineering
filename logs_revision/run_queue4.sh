#!/bin/bash
# Queue 4: all jobs with orthogonalised directions (paper configuration). Sequential; failures do not stop the queue.
cd /c/Users/Kyra/Documents/Repos/representation-engineering
export PYTHONIOENCODING=utf-8
PY=/c/Users/Kyra/mambaforge-pypy3/envs/repeng/python.exe
M=examples.act_three.experiments.revision
S=logs_revision/queue4.status
job() { name=$1; shift; echo "$(date +%T) START $name" >> $S; $PY -m "$@" > logs_revision/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> $S; }
job r1_norms $M.15_sycophancy_baselines --model llama3 --stage norms
job r2_repro40 $M.00_repro_check --model llama3 --n 40 --ortho on
job r3_rescore_ministral $M.17_cross_reader_rescore --reader ministral
job r3_rescore_llama3 $M.17_cross_reader_rescore --reader llama3
job r3_rescore_mistral7b $M.17_cross_reader_rescore --reader mistral7b
job r4_repe_const $M.14_act_target_ablation --model llama3 --conditions repe_const --subset all
job r5_shuf_idonly $M.14_act_target_ablation --model llama3 --conditions repe_shuffled repe_idonly --subset 650
job r6_baselines $M.15_sycophancy_baselines --model llama3 --stage gen --conditions caa antisyc_prompt
job r7_ays_quick $M.16_sycophancy_benchmarks --model llama3 --quick
job r7_ays $M.16_sycophancy_benchmarks --model llama3
job r8_localgrade $M.19_grade_are_you_sure_local
job r9_rescore_inc_ministral $M.17_cross_reader_rescore --reader ministral
job r9_rescore_inc_llama3 $M.17_cross_reader_rescore --reader llama3
job r9_rescore_inc_mistral7b $M.17_cross_reader_rescore --reader mistral7b
job r10_mt_quick $M.18_multiturn_act --model llama3 --quick
job r10_multiturn $M.18_multiturn_act --model llama3
echo "$(date +%T) QUEUE4 DONE" >> $S
