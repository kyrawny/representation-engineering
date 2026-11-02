#!/bin/bash
# Queue 3 (P1): waits for queue 2, then three-turn dialogues, then incremental cross-reader rescoring of their outputs.
cd /c/Users/Kyra/Documents/Repos/representation-engineering
export PYTHONIOENCODING=utf-8
PY=/c/Users/Kyra/mambaforge-pypy3/envs/repeng/python.exe
M=examples.act_three.experiments.revision
until grep -q "QUEUE2 DONE" logs_revision/queue2.status 2>/dev/null; do sleep 30; done
job() { name=$1; shift; echo "$(date +%T) START $name" >> logs_revision/queue3.status; $PY -m "$@" > logs_revision/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> logs_revision/queue3.status; }
job q10a_multiturn_quick $M.18_multiturn_act --model llama3 --quick
job q10_multiturn $M.18_multiturn_act --model llama3
echo "$(date +%T) QUEUE3 DONE" >> logs_revision/queue3.status
