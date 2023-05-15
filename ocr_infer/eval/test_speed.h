#ifndef OCR_INFER_EVAL_TEST_SPEED_H_
#define OCR_INFER_EVAL_TEST_SPEED_H_

#include <algorithm>
#include <thread>
#include <unordered_set>

#include "ocr_infer/core/pipeline/pipeline.h"
#include "ocr_infer/core/util/config_util.h"
#include "ocr_infer/util/image_util.h"
#include "ocr_infer/util/read_config.h"
#include "ocr_infer/util/syscall.h"
#include "ocr_infer/util/timer.h"

class TestSpeed {
 public:
  TestSpeed(const std::string &config_file, int opt) {
    std::unordered_map<std::string, std::string> config;
    if (!read_config(config_file, "configuration", config)) {
      printf("read config.ini failed\n");
      exit(1);
    }

    detect_batch_size_ = std::stoi(Query(config, "detect_batch_size"));

    auto pipeline_io = PipelineFactory::BuildE2e(config);
    sender_ = pipeline_io.first, receiver_ = pipeline_io.second;

    consumer_ = std::make_shared<std::thread>([this]() { ConsumeAndMatch(); });
  }

  void Run(const std::string &test_data_dir) {
    saved_file_.clear();
    images_.clear();
    std::vector<std::string> names;

    // Read images
    LOG(INFO) << "Begin to read images.";
    ReadImages(test_data_dir, names, images_);
    int count = images_.size();
    LOG(INFO) << "There are " << count << " images";

    int id = 0;
    double tick_fake_start = Timer::GetMillisecond();
    double tick_start, tick_end;
    int start_point = 4000, end_point = 9000, test_num = 10000;
    while (1) {
      auto input = std::make_shared<DetInput>();
      for (int j = 0; j < detect_batch_size_; j++) {
        std::string name = names[id % count];
        input->names.emplace_back(name);
        input->images.emplace_back(images_[name]);
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
        std::stringstream ss;
        ss << "Test frames = " << (id - start_point) << "\n"
           << "Total time = " << diff / 1.0e3 << " s\n"
           << "Average time per image = " << average_time << " ms/image\n"
           << "FPS = " << fps << "\n";
        std::cout << "\n" << ss.str() << "\n";
        ofstream ofs("./speed.txt");
        ofs << ss.rdbuf();
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

  std::unordered_map<std::string, cv::Mat> images_;
  std::unordered_set<std::string> saved_file_;

  int detect_batch_size_;

  void OnlyConsume() {
    while (1) {
      receiver_->pop();
    }
  }

  void ConsumeAndMatch() {
    // TODO: add check directory util.
    auto check_dir = [](const std::string &dir) {
      if (Access(dir.c_str(), 0) == 0) {
        std::string cmd = "rm -r " + dir;
        system(cmd.c_str());
      }
      CHECK(Mkdir(dir.c_str(), 0777) == 0)
          << "Can't create directory " << dir << " !\n";
    };
    std::string det_output_dir = "/home/chenlei/Documents/cnc/det_output/";
    std::string rec_output_dir = "/home/chenlei/Documents/cnc/rec_output/";
    check_dir(det_output_dir);
    check_dir(rec_output_dir);

    while (1) {
      auto res = receiver_->pop();
      MatAndPrintResultProcess(res);
    }
  }

  // print intermediate results
  void MatAndPrintResultProcess(std::shared_ptr<MatchOutput> res) {
    for (int i = 0; i < res->names.size(); i++) {
      std::string name = res->names[i];
      std::string det_output_path =
          "/home/chenlei/Documents/cnc/det_output/" + name + ".jpg";
      std::string rec_output_path =
          "/home/chenlei/Documents/cnc/rec_output/" + name + ".txt";
      std::stringstream ss;

      int boxnum = res->boxnum[i];
      std::cout << name << " has " << boxnum << " CiTiaos:" << std::endl;
      cv::Mat img = images_[name].clone();
      for (int j = 0; j < boxnum; j++) {
        std::string text = res->multitext[i][j];
        cv::RotatedRect box = res->multiboxes[i][j];
        cv::Point2f vertices2f[4];
        box.points(vertices2f);
        for (int k = 0; k < 4; k++) {
          ss << int(vertices2f[k].x) << "," << int(vertices2f[k].y) << ",";
        }
        std::cout << "\t" << text << std::endl;
        ss << text << std::endl;

        if (saved_file_.find(name) == saved_file_.end()) {
          DrawDetectBox(img, box, vertices2f);
          cv::imwrite(det_output_path, img);
        }
      }
      std::cout << "*** hit id = " << res->hitid[i] << std::endl;

      if (saved_file_.find(name) == saved_file_.end()) {
        saved_file_.insert(name);
        std::ofstream ofs(rec_output_path.c_str());
        if (!ofs.is_open()) {
          std::cout << "Can't open output file! Please check file path."
                    << std::endl;
          continue;
        }
        ofs << ss.rdbuf();
        ofs.close();
      }
    }
  }
};

#endif  // OCR_INFER_EVAL_TEST_SPEED_H_
