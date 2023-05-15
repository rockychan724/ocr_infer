#include "ocr_infer/engines/parallel_engine.h"

#include "glog/logging.h"
#include "ocr_infer/core/pipeline/pipeline.h"
#include "ocr_infer/core/util/config_util.h"
#include "ocr_infer/util/image_util.h"
#include "ocr_infer/util/init.h"
#include "ocr_infer/util/read_config.h"
#include "ocr_infer/util/syscall.h"
#include "ocr_infer/util/timer.h"

ParallelEngine::ParallelEngine() : consumer_(nullptr) { stop_consume_ = false; }

ParallelEngine::~ParallelEngine() {
  stop_consume_ = true;
  if (consumer_ && consumer_->joinable()) {
    consumer_->detach();
  }
}

int ParallelEngine::Init(const std::string& config_file,
                         CallbackFunc callback_func, void* other,
                         const std::string& output_dir) {
  callback_func_ = callback_func;
  other_ = other;
  output_dir_ = output_dir;

  std::unordered_map<std::string, std::string> config;
  CHECK(read_config(config_file, "configuration", config))
      << "Read \"config.ini\" failed!";

  // check directory
  auto check_path = [](std::string& path) {
    // check directory postfix
    if (path.back() != '/') {
      path += "/";
    }
    // mkdir
    if (Access(path.c_str(), 0) == 0) {
      std::string cmd = "rm -r " + path;
      system(cmd.c_str());
    }
    CHECK(Mkdir(path.c_str(), 0777) == 0)
        << "Can't create directory " << path << " !\n";
  };
  check_path(output_dir_);

  auto pipeline_io = PipelineFactory::BuildE2e(config);
  sender_ = pipeline_io.first;
  receiver_ = pipeline_io.second;

  detect_batch_size_ = std::stoi(Query(config, "detect_batch_size"));

  consumer_ = std::make_unique<std::thread>([this]() { Consume(); });

  return InitLog("ocr_infer");
}

int ParallelEngine::Run(const std::string& image_dir, int start_point,
                        int end_point, int test_num) {
  images_.clear();
  std::vector<std::string> names;

  // Read images
  LOG(INFO) << "Begin to read images.";
  ReadImages(image_dir, names, images_);
  int count = images_.size();
  LOG(INFO) << "There are " << count << " images";

  int id = 0;
  double tick_fake_start = Timer::GetMillisecond();  // debug
  double tick_start, tick_end;
  while (1) {
    auto input = std::make_shared<DetInput>();
    for (int j = 0; j < detect_batch_size_; j++) {
      std::string name = names[id % count];
      input->names.emplace_back(name);
      input->images.emplace_back(images_[name]);
      id++;
    }
    sender_->push(input);
    // debug
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
      std::stringstream ss;
      ss << "Test frames = " << (id - start_point) << "\n"
         << "Total time = " << diff / 1.0e3 << " s\n"
         << "Average time per image = " << average_time << " ms/image\n"
         << "FPS = " << fps << "\n";
      std::cout << "\n" << ss.str() << "\n";

      // save speed info to file
      ofstream ofs(output_dir_ + "speed.txt");
      ofs << ss.rdbuf();
      ofs.close();
    } else if (id >= test_num) {
      break;
    }
  }
  printf("Waiting 20 seconds for pipeline synchronization.\n");
  fflush(stdout);
  for (int i = 0; i < 20; i++) {
    std::cout << "Run over!\n";
    sleep(1);
  }
  return 0;
}

void ParallelEngine::Consume() {
  while (!stop_consume_) {
    auto match_result = receiver_->pop();
    Print(match_result, true, true);
  }
}

void ParallelEngine::Print(const std::shared_ptr<MatchOutput>& match_result,
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
