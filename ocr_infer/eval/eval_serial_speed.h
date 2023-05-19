#ifndef OCR_INFER_EVAL_EVAL_SERIAL_ACC_H_
#define OCR_INFER_EVAL_EVAL_SERIAL_ACC_H_

#include <filesystem>
#include <unordered_set>

#include "ocr_infer/engines/serial_engine.h"
#include "ocr_infer/util/image_util.h"
#include "ocr_infer/util/timer.h"

namespace fs = std::filesystem;

class EvalSerialSpeed : public SerialEngine {
 public:
  virtual int Run(const std::string &image_dir) override {
    images_.clear();
    std::vector<std::string> names;

    // Read images
    LOG(INFO) << "Begin to read images.";
    ReadImages(image_dir, names, images_);
    int count = images_.size();
    LOG(INFO) << "There are " << count << " images";

    int det_batch_num = std::ceil(double(count) / detect_batch_size_);
    double tick_start, tick_end;
    tick_start = Timer::GetMillisecond();
    int run_times = 5;
    for (int _ = 0; _ < run_times; _++) {
      int begin_index = 0;
      for (int i = 0; i < det_batch_num; i++) {
        std::shared_ptr<DetInput> det_in = std::make_shared<DetInput>();
        int end_index = begin_index + detect_batch_size_ >= count
                            ? count
                            : begin_index + detect_batch_size_;
        for (int j = begin_index; j < end_index; j++) {
          det_in->names.emplace_back(names[j]);
          det_in->images.emplace_back(images_[names[j]]);
        }
        begin_index = end_index;

        std::shared_ptr<MatchOutput> match_out =
            serial_e2e_pipeline_->Run(det_in);
      }
    }
    tick_end = Timer::GetMillisecond();

    double diff = tick_end - tick_start;
    double average_time = diff / (count * run_times);
    double fps = 1.0e3 / average_time;
    std::stringstream ss;
    ss << "Test frames = " << count * run_times << "\n"
       << "Total time = " << diff / 1.0e3 << " s\n"
       << "Average time per image = " << average_time << " ms/image\n"
       << "FPS = " << fps << "\n";
    std::cout << "\n" << ss.str() << "\n";

    // save speed info to file
    fs::path save_file = output_dir_ / "speed.txt";
    std::ofstream ofs(save_file);
    ofs << ss.rdbuf();
    ofs.close();
    return 0;
  }
};

#endif  // OCR_INFER_EVAL_EVAL_SERIAL_ACC_H_
