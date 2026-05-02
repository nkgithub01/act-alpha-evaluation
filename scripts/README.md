## Run docker environment:

```bash
run_docker.sh --compile
```

## next build backend

```bash
build_backend.sh QKV
```

## Finally, run the backend to compile a kernel

```bash
./act-backends/QKV --input attention.hlo --output asm/compiled_qkv.py
````