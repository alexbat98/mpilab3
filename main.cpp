#include <iostream>
#include <cmath>
#include <vector>
#include <queue>
#include <MPI.h>
#include <random>
#include <iomanip>

void radix_sort(double *array, size_t size) {

  std::vector<std::queue<uint64_t> > rad(65536);

  auto *data = reinterpret_cast<uint64_t*>(array);

  uint64_t mask = 65535;

  for (size_t r = 0; r < 4; r++) {
    for (size_t i = 0; i < size; i++) {
      uint64_t c = (data[i] & mask) >> (r * 16);
      rad[c].push(data[i]);
    }

    mask = mask << 16;

    size_t idx = 0;

    for (size_t i = 0; i < 65536; i++) {
      while (!rad[i].empty()) {
        data[idx++] = rad[i].front();
        rad[i].pop();
      }
    }
  }


}

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

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);

  int procId, procCount;
  double startTime, endTime;
  double scatterStartTime, scatterEndTime;
  double gatherStartTime, gatherEndTime;
  double mergeStartTime, mergeEndTime;
  double sortStartTime, sortEndTime;

  MPI_Comm_rank(MPI_COMM_WORLD, &procId);
  MPI_Comm_size(MPI_COMM_WORLD, &procCount);

  const int n = std::stoi(std::string(argv[1]));

  double *data = nullptr;

  if (procId == 0) {
    data = new double[n];
    std::default_random_engine generator(static_cast<unsigned int>(time(nullptr)));
    std::uniform_real_distribution<double> distribution(1, 100000000000);

    for (size_t i = 0; i < n; i++) {
      data[i] = distribution(generator);
    }
  }

  int partSize = n / procCount;
  int tail = n % procCount;

  int *sizes = procId == 0 ? new int[procCount] : nullptr;
  int *displacements = procId == 0 ? new int[procCount] : nullptr;

  if (procId == 0) {
    for (int i = 0; i < procCount - 1; i++) {
      sizes[i] = partSize;
      displacements[i] = i * partSize;
    }
    sizes[procCount - 1] = (partSize + tail);
    displacements[procCount - 1] = n - (partSize + tail);
  }

  startTime = MPI_Wtime();

  int receiveCount =
      procId == procCount - 1 ? (partSize + tail) : partSize;

  auto *receiveBuffer = new double[receiveCount];

  scatterStartTime = MPI_Wtime();
  MPI_Scatterv(data, sizes, displacements, MPI_DOUBLE, receiveBuffer, receiveCount, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
  scatterEndTime = MPI_Wtime();

  sortStartTime = MPI_Wtime();
  radix_sort(receiveBuffer, static_cast<size_t>(receiveCount));
  sortEndTime = MPI_Wtime();

  gatherStartTime = MPI_Wtime();
  MPI_Gatherv(receiveBuffer, receiveCount, MPI_DOUBLE, data, sizes, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  gatherEndTime = MPI_Wtime();

  double* res = nullptr;

  if (procId == 0) {
    res = new double[n];

    mergeStartTime = MPI_Wtime();
    merge(data, sizes, displacements, procCount, n, res);
    mergeEndTime = MPI_Wtime();
  }

  endTime = MPI_Wtime();

  if (procId == 0) {
    std::cout << endTime - startTime << std::endl;
    std::cout << scatterEndTime - scatterStartTime << std::endl;
    std::cout << gatherEndTime - gatherStartTime << std::endl;
    std::cout << mergeEndTime - mergeStartTime << std::endl;
    std::cout << sortEndTime - sortStartTime << std::endl;
    std::cout << partSize << std::endl;
  }

  MPI_Finalize();

  return 0;
}