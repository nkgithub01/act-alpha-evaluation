backend=$1
input=$2

# parse the path given in input and use only the last item in the path
input=$(basename "$input")
INPUT_DIR="/workspace/kernels/"
in_file_name="${input%.hlo}"

mkdir -p asm
/workspace/act-backends/${backend} --input ${INPUT_DIR}/${input} --output /workspace/asm/compiled_${in_file_name}_${backend}.py