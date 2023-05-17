# build
cmake -S . -B build
cmake --build build

# run
cd build && ./rec_infer

# eval
cd .. && python3 eval_rec.py 
