#!/bin/bash
# GPU queue 1 (sequential). Each job logs to logs_revision/<name>.log; failures do not stop the queue.
cd /c/Users/Kyra/Documents/Repos/representation-engineering
export PYTHONIOENCODING=utf-8
PY=/c/Users/Kyra/mambaforge-pypy3/envs/repeng/python.exe
M=examples.act_three.experiments.revision
job() { name=$1; shift; echo "$(date +%T) START $name" >> logs_revision/queue1.status; $PY -m "$@" > logs_revision/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> logs_revision/queue1.status; }
job q2_caa_prepare $M.15_sycophancy_baselines --model llama3 --stage prepare
job q2b_caa_quick $M.15_sycophancy_baselines --model llama3 --stage gen --quick
job q5a_ays_quick $M.16_sycophancy_benchmarks --model llama3 --quick
job q1_repe_const $M.14_act_target_ablation --model llama3 --conditions repe_const --subset all
job q3_ablations $M.14_act_target_ablation --model llama3 --conditions repe_act repe_shuffled repe_idonly --subset 650
job q4_baselines $M.15_sycophancy_baselines --model llama3 --stage gen --conditions caa antisyc_prompt
job q5_ays $M.16_sycophancy_benchmarks --model llama3
job q6a_smoke $M.17_cross_reader_rescore --reader mistral7b --quick
job q6_ministral $M.17_cross_reader_rescore --reader ministral
job q6_mistral7b $M.17_cross_reader_rescore --reader mistral7b
job q6_llama3 $M.17_cross_reader_rescore --reader llama3
echo "$(date +%T) QUEUE1 DONE" >> logs_revision/queue1.status
