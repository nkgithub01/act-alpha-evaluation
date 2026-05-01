# First run docker:
run_docker.sh --compile

# next build backend

build_backend.sh QKV

./act-backends/QKV --input attention.hlo --output asm/compiled_qkv.py
