# build
cmake -S . -B build
cmake --build build

# run
./build/detect_infer

# eval
python3 eval_detect.py 
