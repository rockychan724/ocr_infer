#ifndef OCR_INFER_TEST_TEST_SPEED_H_
#define OCR_INFER_TEST_TEST_SPEED_H_

#include <algorithm>
#include <random>
#include <thread>

#include "ocr_infer/core/pipeline/pipeline.h"
#include "ocr_infer/test/util/read_config.h"
#include "ocr_infer/test/util/syscall.h"
#include "ocr_infer/test/util/timer.h"

class TestSpeed {
 public:
  TestSpeed(const std::string &config_file, int opt) {
    std::unordered_map<std::string, std::string> config;
    if (!read_config(config_file, "configuration", config)) {
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

    consumer_ = std::make_shared<std::thread>([this]() { ConsumeAndMatch(); });
  }

  void Run(const std::string &test_data_dir) {
    std::vector<cv::Mat> images;
    std::vector<std::string> names;
    size_t count = ReadImages(test_data_dir, images, names);
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
    std::exit(0);
  }

 private:
  std::shared_ptr<QueueSender<DetInput>> sender_;
  std::shared_ptr<QueueReceiver<MatchOutput>> receiver_;
  std::shared_ptr<std::thread> consumer_;

  std::unordered_map<std::string, int> saved_num_;

  int detect_batch_size_;

  void OnlyConsume() {
    while (1) {
      receiver_->pop();
    }
  }

  void ConsumeAndMatch() {
    std::string output_dir = "/home/chenlei/Documents/cnc/rec_output/";
    if (Access(output_dir.c_str(), 0) == 0) {
      std::string cmd = "rm -r " + output_dir;
      system(cmd.c_str());
    }
    if (Mkdir(output_dir.c_str(), 0777) == -1) {
      printf("Cannot make \"%s\"!\n", output_dir.c_str());
      exit(1);
    }
    while (1) {
      auto res = receiver_->pop();
      MatAndPrintResultProcess(res);
    }
  }

  // print intermediate results
  void MatAndPrintResultProcess(std::shared_ptr<MatchOutput> res) {
    for (auto it = res->name2boxnum.begin(); it != res->name2boxnum.end();
         it++) {
      std::string name = it->first;
      std::string file =
          "/home/chenlei/Documents/cnc/rec_output/" + name + ".txt";
      std::stringstream ss;
      std::cout << name << " has " << it->second << " CiTiaos:" << std::endl;
      size_t result_num = res->name2text[name].size();
      for (size_t i = 0; i < result_num; i++) {
        std::string text = res->name2text[name][i];
        cv::RotatedRect box = res->name2boxes[name][i];
        cv::Point2f vertices2f[4];
        box.points(vertices2f);
        cv::Point root_points[1][4];
        for (int j = 0; j < 4; ++j) {
          ss << int(vertices2f[j].x) << "," << int(vertices2f[j].y) << ",";
        }
        std::cout << "\t" << text << std::endl;
        ss << text << std::endl;
      }
      std::cout << "*** hit id = " << res->name2hitid[name] << std::endl;
      // std::vector<string> hit_content =
      //     extern_interface.find_sensi_word(res->name2hitid[name]);
      // for (auto h = hit_content.begin(); h != hit_content.end(); h++) {
      //   std::cout << "\t" << *h << std::endl;
      // }

      // 完整地保存识别结果，方便准确测试
      // TODO: 考虑一下是否新增收集结果的节点
      if (saved_num_.find(name) == saved_num_.end()) {
        std::ofstream ofs(file.c_str());
        if (!ofs.is_open()) {
          std::cout << "Can't open output file! Please check file path."
                    << std::endl;
          continue;
        }
        ofs << ss.rdbuf();
        ofs.close();
        saved_num_.insert({name, res->name2text[name].size()});
        if (res->name2text[name].size() < it->second) {
          std::cout << "****** " << name << ", " << res->name2text[name].size() << ", "
                  << it->second << ", " << ss.str() << std::endl;
        }
      } else if (saved_num_[name] < it->second) {
        std::ofstream ofs(file.c_str(), ios::app);
        ofs << ss.rdbuf();
        ofs.close();
        saved_num_[name] += res->name2text[name].size();
        std::cout << "****** " << name << ", " << res->name2text[name].size() << ", "
                  << it->second << ", " << ss.str() << std::endl;
      }
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
