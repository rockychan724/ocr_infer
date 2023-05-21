# build
cmake -S . -B build
cmake --build build

# run
./build/detect_infer

# eval
python3 ../py_infer/eval_detect.py ../../../testdata/e2e/gt inference_output/preds
