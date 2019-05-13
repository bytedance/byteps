import os
import json

role = os.environ['ARNOLD_ROLE']
ps = os.environ['ARNOLD_SERVER_HOSTS'].split(',')
worker = os.environ['ARNOLD_WORKER_HOSTS'].split(',')
task_index = int(os.environ['ARNOLD_ID'])

async = int(os.getenv("ASYNC", '0'))

if not async:

    if role == 'worker':
        role = 'master'

    if role == 'master':
        cmd_line_flags = ' '.join([
            "--master=grpc://%s" % (ps[task_index]),
            "--ps_replicas=%d" % len(ps),
            "--worker_replicas=%d" % len(worker),
            "--worker_id=%d" % task_index,
            "--worker_job='/job:master'",
            "--schedule=train",
            "--sync" if len(worker) == 1 else "",
        ])
    else:
        cmd_line_flags = " ".join([
            "--master=grpc://%s" % (ps[task_index]),
            "--schedule=run_std_server",
        ])

else:
    chief = worker[0]
    if role == 'worker':
        if task_index == 0:
            cmd_line_flags = ' '.join([
                "--master=grpc://%s" % (worker[task_index]),
                "--ps_replicas=%d" % len(ps),
                "--worker_replicas=%d" % len(worker),
                "--worker_id=%d" % task_index,
                "--worker_job=/job:chief",
                "--schedule=train",
            ])
        else:
            cmd_line_flags = ' '.join([
                "--master=grpc://%s" % (worker[task_index]),
                "--ps_replicas=%d" % len(ps),
                "--worker_replicas=%d" % len(worker),
                "--worker_id=%d" % task_index,
                "--worker_job=/job:worker",
                "--schedule=train",
            ])
    else:
        cmd_line_flags = " ".join([
            "--master=grpc://%s" % (chief),
            "--schedule=run_std_server",
        ])


print('%s' % cmd_line_flags)
