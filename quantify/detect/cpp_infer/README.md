# build
cmake -S . -B build
cmake --build build

# run
cd build
./detect_infer

# eval
python3 eval_detect.py 
