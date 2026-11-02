#!/bin/bash
# Queue 5: continues queue 4 after the running r5 job; baselines and incremental rescoring come before the long benchmark.
cd /c/Users/Kyra/Documents/Repos/representation-engineering
export PYTHONIOENCODING=utf-8
PY=/c/Users/Kyra/mambaforge-pypy3/envs/repeng/python.exe
M=examples.act_three.experiments.revision
S=logs_revision/queue5.status
while powershell -NoProfile -ExecutionPolicy Bypass -File logs_revision/wait_r5.ps1; do sleep 20; done
job() { name=$1; shift; echo "$(date +%T) START $name" >> $S; $PY -m "$@" > logs_revision/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> $S; }
job s1_baselines $M.15_sycophancy_baselines --model llama3 --stage gen --conditions caa antisyc_prompt
job s2_rescore_ministral $M.17_cross_reader_rescore --reader ministral
job s2_rescore_llama3 $M.17_cross_reader_rescore --reader llama3
job s2_rescore_mistral7b $M.17_cross_reader_rescore --reader mistral7b
job s3_ays_quick $M.16_sycophancy_benchmarks --model llama3 --quick
job s3_ays $M.16_sycophancy_benchmarks --model llama3
job s4_localgrade $M.19_grade_are_you_sure_local
job s5_mt_quick $M.18_multiturn_act --model llama3 --quick
job s5_multiturn $M.18_multiturn_act --model llama3
echo "$(date +%T) QUEUE5 DONE" >> $S
