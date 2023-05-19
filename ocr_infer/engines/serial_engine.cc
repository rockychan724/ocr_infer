#include "ocr_infer/engines/serial_engine.h"

#include <cmath>

#include "glog/logging.h"
#include "ocr_infer/util/config_util.h"
#include "ocr_infer/util/image_util.h"
#include "ocr_infer/util/init.h"
#include "ocr_infer/util/timer.h"

int SerialEngine::Init(const std::string& config_file,
                       CallbackFunc callback_func, void* other) {
  std::unordered_map<std::string, std::string> config;
  CHECK(ReadConfig(config_file, "configuration", config))
      << "Read config file failed!";

  callback_func_ = callback_func;
  other_ = other;
  output_dir_ = Query(config, "output_dir");

  InitDirectory("ocr_infer", output_dir_);

  detect_batch_size_ = std::stoi(Query(config, "detect_batch_size"));

  serial_e2e_pipeline_ = std::make_unique<SerialE2ePipeline>(config);

  return 0;
}

int SerialEngine::Run(const std::string& image_dir) {
  images_.clear();
  std::vector<std::string> names;

  // Read images
  LOG(INFO) << "Begin to read images.";
  ReadImages(image_dir, names, images_);
  int count = images_.size();
  LOG(INFO) << "There are " << count << " images";

  int det_batch_num = std::ceil(double(count) / detect_batch_size_);
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

    std::shared_ptr<MatchOutput> match_out = serial_e2e_pipeline_->Run(det_in);

    Print(match_out, true, true);
  }

  return 0;
}

// std::string SerialEngine::Run(const std::shared_ptr<Input>& in) {
//   std::shared_ptr<DetInput> det_in = std::make_shared<DetInput>();
//   det_in->names = in->names;
//   det_in->images = in->images;

//   std::shared_ptr<MatchOutput> match_result =
//   serial_e2e_pipeline_->Run(det_in);

//   return Print(match_result);
// }

void SerialEngine::Print(const std::shared_ptr<MatchOutput>& match_result,
                         bool draw_detect_box, bool execute_callback_func) {
  for (int i = 0; i < match_result->names.size(); i++) {
    std::string name = match_result->names[i];
    std::stringstream ss;
    int boxnum = match_result->boxnum[i];
    cv::Mat img = images_[name].clone();
    for (int j = 0; j < boxnum; j++) {
      std::string text = match_result->multitext[i][j];
      cv::RotatedRect box = match_result->multiboxes[i][j];
      cv::Point2f vertices2f[4];
      box.points(vertices2f);
      for (int k = 0; k < 4; k++) {
        ss << int(vertices2f[k].x) << "," << int(vertices2f[k].y) << ",";
      }
      ss << text << std::endl;
      if (draw_detect_box) {
        DrawDetectBox(img, box, vertices2f);
      }
    }

    if (execute_callback_func) {
      callback_func_(ss.str(), img, other_);
    }
  }
}
