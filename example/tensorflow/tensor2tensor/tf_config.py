import os
import json

role = os.environ['ARNOLD_ROLE']
ps = os.environ['ARNOLD_SERVER_HOSTS'].split(',')
worker = os.environ['ARNOLD_WORKER_HOSTS'].split(',')
task_index = os.environ['ARNOLD_ID']
async = int(os.getenv("ASYNC", '0'))

if not async:
    if role == 'worker':
        role = 'master'

    cluster = {'ps': ps, 'master': worker}
    config = json.dumps({
        'cluster': cluster,
        'task': {
            'type': role,
            'index': task_index,
        },
        'environment': 'cloud'
    })

if async:
    if len(worker) < 2:
        cluster = {'ps': ps, 'chief': worker}
        if role == 'worker':
            role = 'chief'
        config = json.dumps({
            'cluster': cluster,
            'task': {
                'type': role,
                'index': task_index,
            },
            'environment': 'cloud'
        })
    else:
        cluster = {'ps': ps, 'chief': [worker[0]], 'worker': worker[1:]}
        if role == 'worker' and int(task_index) == 0:
            role = 'chief'
            config = json.dumps({
                'cluster': cluster,
                'task': {
                    'type': role,
                    'index': task_index,
                },
                'environment': 'cloud'
            })
        else:
            if role == 'worker':
                task_index = str(int(task_index) - 1)
                config = json.dumps({
                    'cluster': cluster,
                    'task': {
                        'type': role,
                        'index': task_index,
                    },
                    'environment': 'cloud'
                })
            else:
                config = json.dumps({
                    'cluster': cluster,
                    'task': {
                        'type': role,
                        'index': task_index,
                    },
                    'environment': 'cloud'
                })



print('%s' % config)
