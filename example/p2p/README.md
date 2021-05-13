Steps to run tests locally: 

1. Update the root IP and PORT in common.sh
2. `bash scheduler.sh`
3. `bash server.sh 0`
4. `bash server.sh 1`
5. `bash worker.sh 0 0`, first argument for rank, second argument for cpu test
6. `bash worker.sh 1 0`, first argument for rank, second argument for cpu test