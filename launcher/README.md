### How to use distributed launcher

Create two host files: `worker_hosts` and `server_hosts`, put your lists of hosts inside (one IP:port per line).

For example, we want `10.0.0.1:12345` to be the scheduler, `10.0.0.2` and `10.0.0.3` to be the workers, `10.0.0.4` and `10.0.0.5` to be the servers.

Then `worker_hosts` should be:
```
10.0.0.2
10.0.0.3
```

And `server_hosts` should be:
```
10.0.0.4
10.0.0.5
```

Finally, start the distributed ssh launcher by:

```
python dist_launcher.py --worker-hostfile worker_hosts --server-hostfile server_hosts \
        --scheduler-ip 10.0.0.1 --scheduler-port 12345 \
        --username root --env ENV1:1 --env ENV2:2 \
        'echo this is $DMLC_ROLE; python byteps/launcher/launch.py YOUR_COMMAND'
```

The script will automatically help you setup the necessary [environment variables](/docs/env.md) and launch BytePS processes.