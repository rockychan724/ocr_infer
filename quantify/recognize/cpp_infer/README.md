# build
cmake -S . -B build
cmake --build build

# run
./build/rec_infer

# eval
python3 eval_rec.py 
