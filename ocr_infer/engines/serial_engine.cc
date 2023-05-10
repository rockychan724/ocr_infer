#include "ocr_infer/engines/serial_engine.h"

#include <cmath>

#include "glog/logging.h"
#include "ocr_infer/core/util/config_util.h"
#include "ocr_infer/util/image_util.h"
#include "ocr_infer/util/init.h"
#include "ocr_infer/util/read_config.h"
#include "ocr_infer/util/timer.h"

int SerialEngine::Init(const std::string& config_file,
                       CallbackFunc callback_func, void* other) {
  std::unordered_map<std::string, std::string> config;
  CHECK(read_config(config_file, "configuration", config))
      << "Read \"config.ini\" failed!";

  detect_batch_size_ = std::stoi(Query(config, "detect_batch_size"));

  serial_e2e_pipeline_ = std::make_unique<SerialE2ePipeline>(config);

  callback_func_ = callback_func;
  other_ = other;

  return InitLog("ocr_infer");
}

int SerialEngine::Run(const std::string& image_dir) {
  std::vector<cv::Mat> images;
  std::vector<std::string> names;
  size_t count = ReadImages(image_dir, images, names);

  int det_batch_num = std::ceil(double(count) / detect_batch_size_);
  int begin_index = 0;
  for (int i = 0; i < det_batch_num; i++) {
    std::shared_ptr<DetInput> det_in = std::make_shared<DetInput>();
    int end_index = begin_index + detect_batch_size_ >= count
                        ? count
                        : begin_index + detect_batch_size_;
    det_in->names.assign(names.begin() + begin_index,
                         names.begin() + end_index);
    det_in->images.assign(images.begin() + begin_index,
                          images.begin() + end_index);
    begin_index = end_index;

    std::shared_ptr<MatchOutput> match_out = serial_e2e_pipeline_->Run(det_in);

    Print(match_out, true);
  }

  return 0;
}

std::string SerialEngine::Run(const std::shared_ptr<Input>& in) {
  std::shared_ptr<DetInput> det_in = std::make_shared<DetInput>();
  det_in->names = in->names;
  det_in->images = in->images;

  std::shared_ptr<MatchOutput> match_result = serial_e2e_pipeline_->Run(det_in);

  return Print(match_result);
}

std::string SerialEngine::Print(
    const std::shared_ptr<MatchOutput>& match_result,
    bool execute_callback_func) {
  std::string out;
  for (auto it = match_result->name2boxnum.begin();
       it != match_result->name2boxnum.end(); it++) {
    std::string name = it->first;
    std::stringstream ss;
    // std::cout << name << " has " << it->second << " CiTiaos:" << std::endl;
    size_t text_num = match_result->name2text[name].size();
    for (size_t i = 0; i < text_num; i++) {
      std::string text = match_result->name2text[name][i];
      cv::RotatedRect box = match_result->name2boxes[name][i];
      cv::Point2f vertices2f[4];
      box.points(vertices2f);
      cv::Point root_points[1][4];
      for (int j = 0; j < 4; ++j) {
        ss << int(vertices2f[j].x) << "," << int(vertices2f[j].y) << ",";
      }
      // std::cout << "\t" << text << std::endl;
      ss << text << std::endl;
    }
    // std::cout << "*** hit id = " << match_result->name2hitid[name]
    //           << std::endl;
    // std::vector<string> hit_content =
    //     extern_interface.find_sensi_word(match_result->name2hitid[name]);
    // for (auto h = hit_content.begin(); h != hit_content.end(); h++) {
    //   std::cout << "\t" << *h << std::endl;
    // }

    out += ss.str();
    if (execute_callback_func) {
      callback_func_(ss.str(), other_);
    }
  }
  return out;
}
