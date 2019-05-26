How to use

build by `make test` in the root directory, then run

```bash
find test_* -type f -executable -exec ./repeat.sh 4 ./local.sh 2 2 ./{} \;
```
