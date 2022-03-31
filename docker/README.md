# Prebuilt Images

Belows are prebuilt docker images, and their associated commands to build. These prebuilt images might not be up-to-date.
You may need to manually build them to get the latest functionalities of BytePS using the dockerfile.

| Docker image | How to build |
| --- | --- |
| bytepsimage/tensorflow       | docker build -t bytepsimage/tensorflow . -f Dockerfile --build-arg FRAMEWORK=tensorflow |
| bytepsimage/pytorch          | docker build -t bytepsimage/pytorch . -f Dockerfile --build-arg FRAMEWORK=pytorch |
| bytepsimage/mxnet            | docker build -t bytepsimage/mxnet . -f Dockerfile --build-arg FRAMEWORK=mxnet |
