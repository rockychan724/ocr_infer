#ifndef OCR_INFER_TEST_TEST_SPEED_H_
#define OCR_INFER_TEST_TEST_SPEED_H_

#include <algorithm>
#include <random>

#include "ocr_infer/core/pipeline/pipeline.h"
#include "ocr_infer/test/util/read_config.h"
#include "ocr_infer/test/util/timer.h"

class TestSpeed {
 public:
  TestSpeed(const std::string &, int) {
    std::unordered_map<std::string, std::string> config;
    if (!read_config("/home/chenlei/Documents/cnc/ocr_infer/ocr_infer/test/config_cnc.ini", config,
                     "configuration")) {
      printf("read config.ini failed\n");
      exit(1);
    }

    auto it = config.find("detect_batch_size");
    CHECK(it != config.end()) << "Can't find \""
                              << "detect_batch_size"
                              << "\" in config!";
    detect_batch_size_ = std::stoi(it->second);

    auto pipeline_io = PipelineFactory::BuildE2e(config);
    sender_ = pipeline_io.first, receiver_ = pipeline_io.second;

    consumer_ = std::make_shared<std::thread>([this]() { Consume(); });
  }

  void Run() {
    std::vector<cv::Mat> images;
    std::vector<std::string> names;
    size_t count = ReadImages("/home/chenlei/Documents/cnc/testdata/image", images, names);
    int id = 0;
    double tick_fake_start = Timer::GetMillisecond();
    double tick_start, tick_end;
    // int period = 10000;
    int start_point = 5000, end_point = 10000, test_num = 10000;
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
            "\nTest frames = %d\nTotal time = %lf s\nAverage time per image = %lf ms\nFPS = "
            "%lf\n\n",
            (id - start_point), diff / 1.0e3, average_time, fps);
        ofstream ofs("./speed.txt");
        ofs << "Test frames = " << (id - start_point) << "\nTotal time = " << diff / 1.0e3
            << " s\nAverage time per image = " << average_time << " ms\nFPS = " << fps << "\n";
        ofs.close();
      } else if (id >= test_num) {
        break;
      }
    }
    printf("Waiting 10 seconds for pipeline synchronization.\n");
    fflush(stdout);
    sleep(10);
    std::exit(0);
  }

 private:
  std::shared_ptr<QueueSender<DetInput>> sender_;
  std::shared_ptr<QueueReceiver<MatchOutput>> receiver_;
  std::shared_ptr<std::thread> consumer_;

  int detect_batch_size_;

  void Consume() {
    while (1) {
      receiver_->pop();
    }
  }

  size_t ReadImages(const string &images_path, std::vector<cv::Mat> &images,
                    std::vector<std::string> &names) {
    LOG(INFO) << "Begin reading images.";
    std::vector<cv::String> files;
    cv::glob(images_path + "/*.jpg", files, false);
    std::shuffle(files.begin(), files.end(), std::default_random_engine(9));
    size_t count = files.size();
    for (size_t i = 0; i < count; i++) {
      cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);
      images.emplace_back(img);
    }
    names = GetFileName(files);
    printf("\nThere are %lu images\n\n", count);
    return count;
  }

  std::vector<std::string> GetFileName(const std::vector<std::string> &files) {
    std::vector<std::string> names;
    for (const auto &file : files) {
      int index1 = file.find_last_of("/");
      int index2 = file.find_last_of(".");
      names.emplace_back(file.substr(index1 + 1, index2 - index1 - 1));
    }
    return names;
  }
};

#endif  // OCR_INFER_TEST_TEST_SPEED_H_
