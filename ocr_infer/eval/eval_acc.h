#ifndef OCR_INFER_EVAL_EVAL_ACC_H_
#define OCR_INFER_EVAL_EVAL_ACC_H_

#include <filesystem>
#include <fstream>
#include <unordered_set>

#include "glog/logging.h"
#include "ocr_infer/engines/parallel_engine.h"
#include "ocr_infer/util/image_util.h"

namespace fs = std::filesystem;

class EvalAcc : public ParallelEngine {
 protected:
  virtual void Consume() override {
    det_output_dir_ = output_dir_ / "det_output";
    rec_output_dir_ = output_dir_ / "rec_output";
    CHECK(fs::create_directories(det_output_dir_));
    CHECK(fs::create_directories(rec_output_dir_));

    while (!stop_consume_) {
      auto match_result = receiver_->pop();
      Print(match_result, true, false);
    }
  }

  virtual void Print(const std::shared_ptr<MatchOutput> &match_result,
                     bool draw_detect_box = false,
                     bool execute_callback_func = false) override {
    for (int i = 0; i < match_result->names.size(); i++) {
      std::string name = match_result->names[i];
      fs::path det_output_path = det_output_dir_ / (name + ".jpg");
      fs::path rec_output_path = rec_output_dir_ / (name + ".txt");
      std::stringstream ss;
      int boxnum = match_result->boxnum[i];
      std::cout << name << " has " << boxnum << " boxes:" << std::endl;
      cv::Mat img = images_[name].clone();
      for (int j = 0; j < boxnum; j++) {
        std::string text = match_result->multitext[i][j];
        cv::RotatedRect box = match_result->multiboxes[i][j];
        cv::Point2f vertices2f[4];
        box.points(vertices2f);
        for (int k = 0; k < 4; k++) {
          ss << int(vertices2f[k].x) << "," << int(vertices2f[k].y) << ",";
        }
        std::cout << "\t" << text << std::endl;
        ss << text << std::endl;
        if (draw_detect_box) {
          DrawDetectBox(img, box, vertices2f, j);
        }
      }
      std::cout << "*** hit id = " << match_result->hitid[i] << std::endl;

      if (saved_file_.find(name) == saved_file_.end()) {
        saved_file_.insert(name);

        // write recognize result
        std::ofstream ofs(rec_output_path.c_str());
        if (!ofs.is_open()) {
          std::cout << "Can't open output file! Please check file path "
                    << rec_output_path << std::endl;
          continue;
        }
        ofs << ss.rdbuf();
        ofs.close();

        // draw detect box
        if (draw_detect_box) {
          cv::imwrite(det_output_path, img);
        }
      }

      if (execute_callback_func) {
        callback_func_(ss.str(), img, other_);
      }
    }
  }

 private:
  fs::path det_output_dir_;
  fs::path rec_output_dir_;

  std::unordered_set<std::string> saved_file_;
};

#endif  // OCR_INFER_EVAL_EVAL_ACC_H_
