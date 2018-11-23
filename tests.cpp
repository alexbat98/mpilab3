//
// Created by Александр Баташев on 23/11/2018.
//
#include <iostream>

void merge(const double* data, const int* sizes, const int* displacements, int p, int n, double* buffer) {

  int *pointers = new int[p];

  for (int i = 0; i < p; i++) {
    pointers[i] = displacements[i];
  }

  for (int i = 0; i < n; i++) {
    double min = 1000000000000;
    int idx = 0;
    for (int k = 0; k < p; k++) {
      if (min > data[pointers[k]] && pointers[k] < displacements[k] + sizes[k]) {
        idx = k;
        min = data[pointers[k]];
      }
    }
    pointers[idx]++;
    buffer[i] = min;
  }

}

int main() {

  double data[] = {1, 3, 5, 2, 4, 6, 8, 9};

  int sizes[] = {3, 3, 2};
  int disp[] = {0, 3, 6};

  double buf[8];

  merge(data, sizes, disp, 3, 8, buf);

  for (double i : buf) {
    std::cout << i << " ";
  }

  std::cout << std::endl;

  return 0;
}