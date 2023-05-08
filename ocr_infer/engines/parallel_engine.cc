#include "ocr_infer/engines/parallel_engine.h"

#include "glog/logging.h"
#include "ocr_infer/core/pipeline/pipeline.h"
#include "ocr_infer/core/util/config_util.h"
#include "ocr_infer/util/image_util.h"
#include "ocr_infer/util/read_config.h"
#include "ocr_infer/util/timer.h"

int ParallelEngine::Init(const std::string& config_file, void* callback_func) {
  std::unordered_map<std::string, std::string> config;
  CHECK(read_config(config_file, "configuration", config))
      << "Read \"config.ini\" failed!";

  detect_batch_size_ = std::stoi(Query(config, "detect_batch_size"));

  auto pipeline_io = PipelineFactory::BuildE2e(config);
  sender_ = pipeline_io.first;
  receiver_ = pipeline_io.second;

  // callback_func_ = static_cast<func*>(callback_func);
  callback_func_ = (void (*)(const std::string&))callback_func;

  consumer_ = std::make_shared<std::thread>([this]() { GatherResult(); });

  return 0;
}

int ParallelEngine::Run(const std::string& image_dir) {
  std::vector<cv::Mat> images;
  std::vector<std::string> names;
  size_t count = ReadImages(image_dir, images, names);
  int id = 0;
  double tick_fake_start = Timer::GetMillisecond();
  double tick_start, tick_end;
  // int period = 10000;
  int start_point = 500, end_point = 1500, test_num = 2000;
  while (1) {
    auto input = std::make_shared<DetInput>();
    for (int j = 0; j < detect_batch_size_; j++) {
      input->images.emplace_back(images[id % count]);
      input->names.emplace_back(names[id % count]);
      id++;
    }
    sender_->push(input);
    // if (id % 160 == 0) {
    //   tick_end = Timer::GetMillisecond();
    //   double diff = tick_end - tick_fake_start;
    //   double rough_average_time = diff / (id - 0);
    //   double rough_fps = 1.0e3 / rough_average_time;
    //   cout << "id = " << id << ", rough_fps = " << rough_fps << endl;
    // }
    if (id >= start_point && id < (start_point + detect_batch_size_)) {
      tick_start = Timer::GetMillisecond();
      start_point = id;
    } else if (id >= end_point && id < (end_point + detect_batch_size_)) {
      tick_end = Timer::GetMillisecond();
      double diff = tick_end - tick_start;
      double average_time = diff / (id - start_point);
      double fps = 1.0e3 / average_time;
      printf(
          "\nTest frames = %d\nTotal time = %lf s\nAverage time per image = "
          "%lf ms\nFPS = "
          "%lf\n\n",
          (id - start_point), diff / 1.0e3, average_time, fps);
      ofstream ofs("./speed.txt");
      ofs << "Test frames = " << (id - start_point)
          << "\nTotal time = " << diff / 1.0e3
          << " s\nAverage time per image = " << average_time
          << " ms\nFPS = " << fps << "\n";
      ofs.close();
    } else if (id >= test_num) {
      break;
    }
  }
  printf("Waiting 10 seconds for pipeline synchronization.\n");
  fflush(stdout);
  for (int i = 0; i < 20; i++) {
    std::cout << "Run over!\n";
    sleep(1);
  }
  return 0;
}

int ParallelEngine::Run(const Input& in) { return 0; }

void ParallelEngine::GatherResult() {
  while (true) {
    auto match_result = receiver_->pop();

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

      // (*callback_func_)(ss.str());
      callback_func_(ss.str());
    }
  }
}
