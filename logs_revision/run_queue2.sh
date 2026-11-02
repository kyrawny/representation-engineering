#!/bin/bash
# Queue 2: waits for queue 1, then local grading of are-you-sure.
cd /c/Users/Kyra/Documents/Repos/representation-engineering
export PYTHONIOENCODING=utf-8
PY=/c/Users/Kyra/mambaforge-pypy3/envs/repeng/python.exe
M=examples.act_three.experiments.revision
until grep -q "QUEUE1 DONE" logs_revision/queue1.status; do sleep 20; done
job() { name=$1; shift; echo "$(date +%T) START $name" >> logs_revision/queue2.status; $PY -m "$@" > logs_revision/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> logs_revision/queue2.status; }
job q7_localgrade $M.19_grade_are_you_sure_local
echo "$(date +%T) QUEUE2 DONE" >> logs_revision/queue2.status
