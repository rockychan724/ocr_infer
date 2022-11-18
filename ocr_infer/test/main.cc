#include <iostream>

#include "ocr_infer/test/test_speed.h"

int main() {
  int a = 10;
  std::cout << "run" << std::endl;

  TestSpeed test("", 0);
  test.Run();

  return 0;
}
