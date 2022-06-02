# Pipeline Server training

Here is the implementation of the server version of pipeline parallelism with compression algorithm.

# Reproduce

```
python3 client_run.py --ifconfig <network card> --url <server ip adress>
```

## Todo

1. Finish the rest compression algorithms
2. give a clear API to  test each part of time-consuming (training time, compression time, send&recv time)

