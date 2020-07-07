# Run BytePS on Kubernetes
This guide will walk you through running BytePS on Kubernetes cluster. We will use MXNet Operator to launch a BytePS job. Please do not get confused with its name, you can still run TensorFlow and PyTorch with BytePS using MXNet Operator.

## Installing MXNet Operator

### Deploy MXJob CRD and Apache MXNet Operator
```
git clone https://github.com/jasonliu747/mxnet-operator.git
cd mxnet-operator
kubectl create -k manifests/overlays/v1beta1/
```

### Verify that MXJob CRD and Apache MXNet Operator are installed
Check that the Apache MXNet custom resource is installed via:
```
kubectl get crd
```
The output should include `mxjobs.kubeflow.org` like the following:
```
NAME                                           AGE
...
mxjobs.kubeflow.org                            4d
...
```
Check that the Apache MXNet operator is running via:
```
kubectl get pods | grep mxnet-operator
```
The output should include mxnet-operaror-xxx like the following:
```
NAME                             READY   STATUS    RESTARTS   AGE
mxnet-operator-d466b46bc-xbqvs   1/1     Running   0          4m37s
```

## Creating a BytePS training job
Let's say you have two workers, and each one with 8 GPUs. For better performance we use two servers as well. You don't have to worry about `DMLC_WORKER_ID`, MXNet Operator will help you inject to each worker accordingly.
```
kubectl create -f examples/v1beta1/byteps_dist_gpu_v1beta1.yaml
```

## Monitoring your training job

```
$ kubectl get pods | grep byteps-mxnet-job
NAME                                          READY   STATUS              RESTARTS   AGE
byteps-mxnet-job-scheduler-0                  0/1     ContainerCreating   0          11s
byteps-mxnet-job-server-0                     0/1     ContainerCreating   0          11s
byteps-mxnet-job-server-1                     0/1     ContainerCreating   0          11s
byteps-mxnet-job-worker-0                     0/1     ContainerCreating   0          10s
byteps-mxnet-job-worker-1                     0/1     ContainerCreating   0          10s
```
And get the sample output for an example job by typing `kubectl get -o yaml mxjob byteps-mxnet-job`
```
apiVersion: kubeflow.org/v1beta1
kind: MXJob
metadata:
  annotations:
  creationTimestamp: "2020-07-06T10:05:13Z"
  generation: 5
  name: byteps-mxnet-job
  namespace: default
  resourceVersion: "130227952"
  selfLink: /apis/kubeflow.org/v1beta1/namespaces/default/mxjobs/byteps-mxnet-job
  uid: 2f1eddff-bf70-11ea-9352-00163e0a61a0
spec:
  cleanPodPolicy: All
  jobMode: MXTrain
  mxReplicaSpecs:
    Scheduler:
      replicas: 1
      restartPolicy: Never
      template:
        metadata:
          creationTimestamp: null
        spec:
          containers:
          - command:
            - bpslaunch
            image: bytepsimage/mxnet
            name: mxnet
            ports:
            - containerPort: 9091
              name: mxjob-port
            resources: {}
    Server:
      replicas: 2
      restartPolicy: Never
      template:
        metadata:
          creationTimestamp: null
        spec:
          containers:
          - command:
            - bpslaunch
            image: bytepsimage/mxnet
            name: mxnet
            ports:
            - containerPort: 9091
              name: mxjob-port
            resources: {}
    Worker:
      replicas: 2
      restartPolicy: Never
      template:
        metadata:
          creationTimestamp: null
        spec:
          containers:
          - args:
            - python3
            - /usr/local/byteps/example/mxnet/train_imagenet_byteps.py
            - --benchmark
            - "1"
            - --batch-size=32
            command:
            - bpslaunch
            image: bytepsimage/mxnet
            name: mxnet
            ports:
            - containerPort: 9091
              name: mxjob-port
            resources:
              limits:
                nvidia.com/gpu: "8"
            volumeMounts:
            - mountPath: /dev/shm
              name: dshm
          volumes:
          - emptyDir:
              medium: Memory
            name: dshm
status:
  conditions:
  - lastTransitionTime: "2020-07-06T10:05:13Z"
    lastUpdateTime: "2020-07-06T10:05:13Z"
    message: MXJob byteps-mxnet-job is created.
    reason: MXJobCreated
    status: "True"
    type: Created
  - lastTransitionTime: "2020-07-06T10:05:13Z"
    lastUpdateTime: "2020-07-06T10:05:35Z"
    message: MXJob byteps-mxnet-job is running.
    reason: MXJobRunning
    status: "True"
    type: Running
  mxReplicaStatuses:
    Scheduler:
      active: 1
    Server:
      active: 2
    Worker: {}
  startTime: "2020-07-06T10:05:13Z"
```