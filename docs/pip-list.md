# BytePS pip list

Choose a pip source according to the framework and CUDA version you already have. Find YOUR_WHEEL_URL from the below lists, and then: 

```
wget -O byteps-0.1.0-cp27-none-any.whl YOUR_WHEEL_URL
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps byteps-0.1.0-cp27-none-any.whl
```

Note: All of the following are only valid for workers. You should refer to [this](/docker/Dockerfile.server) for server/scheduler images.  



TensorFlow

| Version | CUDA | URL |
| --- | --- | --- |
| 1.12.0 | 9.0 | https://test-files.pythonhosted.org/packages/f3/ef/888e2a92eeb1c96081442c9f39614956d9f016dc05c0d5d0020b5e1a37f0/byteps_tensorflow1.12.0_cu90-0.1.0-cp27-none-any.whl |
| 1.14.0 | 9.0 | https://test-files.pythonhosted.org/packages/84/02/50eb38bae4097aa43253c1ef644a86425529546422272167a7f506cf1354/byteps_tensorflow1.14.0_cu90_v1-0.1.0-cp27-none-any.whl |
| 1.14.0 | 10.0 | https://test-files.pythonhosted.org/packages/cb/ef/6baf3b3d4c69f8a31a2cd50437fb7b81acd14da8fd31644bc8d21c4c850e/byteps_tensorflow1.14.0_cu100-0.1.0-cp27-none-any.whl |



PyTorch

| Version | CUDA | URL |
| --- | --- | --- |
| 1.0.1 | 9.0 | https://test-files.pythonhosted.org/packages/ff/ae/37f5ca6597081127da9f52e486d66f33806dfe61da0bbc5e04d97f818c39/byteps_pytorch1.0.1_cu90-0.1.0-cp27-none-any.whl |
| 1.0.1 | 10.0 | https://test-files.pythonhosted.org/packages/a6/7f/05dd08c83df6fb9143c29183b60c01dea16c1d81f45b70f745646c167537/byteps_pytorch1.0.1_cu100-0.1.0-cp27-none-any.whl |
| 1.1.0 | 9.0 | https://test-files.pythonhosted.org/packages/db/6f/c99266a52e71d4df875fdf3ff3fa073b98424ea0a7182a0237b1930d34be/byteps_pytorch1.1.0_cu90-0.1.0-cp27-none-any.whl |
| 1.1.0 | 10.0 | https://test-files.pythonhosted.org/packages/cf/03/1b26a3bb259d7cf1f7d4d0ad731c6a3eeae2e4f4b1273d7a344dc90b300a/byteps_pytorch1.1.0_cu100-0.1.0-cp27-none-any.whl |


MXNet

| Version | CUDA | URL |
| --- | --- | --- |
| 1.4.1 | 9.0 | https://test-files.pythonhosted.org/packages/11/3c/0abba947c2d212ea205801108e81d8445d08686bbf91a72d3dc8249dd266/byteps_mxnet1.4.1_cu90-0.1.0-cp27-none-any.whl |
| 1.4.1 | 10.0 | https://test-files.pythonhosted.org/packages/3b/c5/c9545305cac2669f90819e33c748c169a7ae6daf454326a21879b3376fff/byteps_mxnet1.4.1_cu100-0.1.0-cp27-none-any.whl |


