#ifndef OCR_INFER_ENGINES_PARALLEL_ENGINE_H_
#define OCR_INFER_ENGINES_PARALLEL_ENGINE_H_

#include <memory>
#include <thread>

#include "ocr_infer/api/data_type.h"
#include "ocr_infer/core/common/data_structure.h"
#include "ocr_infer/core/common/transmission.h"

class ParallelEngine {
 public:
  ParallelEngine();

  ~ParallelEngine();

  /**
   * @brief
   *
   * @param config_file
   * @param callback_func
   * @param other
   * @param output_dir
   * @return int
   */
  virtual int Init(const std::string &config_file,
                   CallbackFunc callback_func = nullptr, void *other = nullptr,
                   const std::string &output_dir = "./output");

  /**
   * @brief
   *
   * @param image_dir
   * @param start_point
   * @param end_point
   * @param test_num
   * @return int
   */
  virtual int Run(const std::string &image_dir, int start_point = 500,
                  int end_point = 1500, int test_num = 2000);

 protected:
  std::shared_ptr<QueueSender<DetInput>> sender_;
  std::shared_ptr<QueueReceiver<MatchOutput>> receiver_;
  std::unique_ptr<std::thread> consumer_;
  bool stop_consume_;

  int detect_batch_size_;

  std::unordered_map<std::string, cv::Mat> images_;

  CallbackFunc callback_func_;
  void *other_;
  std::string output_dir_;

  virtual void Consume();

  virtual void Print(const std::shared_ptr<MatchOutput> &match_result,
                            bool draw_detect_box = false,
                            bool execute_callback_func = false);
};

#endif  // OCR_INFER_ENGINES_PARALLEL_ENGINE_H_
