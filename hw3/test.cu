#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int exit_code = 0;
     
     // check that the correct number of command line arguments were given
     if (argc != 2)
     {
         std::cerr << "usage: " << argv[0] << " <int>" << std::endl
                   << "Must supply the number of times to double the input file!" 
                   << std::endl;
         return 1;
     }
 
     // convert argument to integer
     int number_of_doubles = std::atoi(argv[1]);
     if (number_of_doubles < 0)
     {
         std::cerr << "usage: " << argv[0] << " <int>" << std::endl
                   << "Number of time times must be at least zero." 
                   << std::endl;
         return 1;
     }
 
     cudaFree(0); //initialize cuda context to avoid including cost in timings later
 
     // Warm-up each of the kernels to avoid including overhead in timing.
     // If the kernels are written correctly, then they should
     // never make a bad memory access, even though we are passing in NULL
     // pointers since we are also passing in a size of 0
     shift_char <<<1, 1>>>(nullptr, nullptr, 0, 0);
     shift_int  <<<1, 1>>>(nullptr, nullptr, 0, 0);
     shift_int2 <<<1, 1>>>(nullptr, nullptr, 0, 0);
 
     // First load the text
     std::ifstream ifs(MOBY_DICK, std::ios::binary);
     if (!ifs) 
     {
         std::cerr << "Couldn't open " << MOBY_DICK << "!" << std::endl;
         return 1;
     }
 
     std::vector<unsigned char> text;
 
     // get file length by seeking to end of file and getting offset
     // we then re-seek back to the beginning of the file
     ifs.seekg(0, std::ios::end);
     int length = ifs.tellg();
     ifs.seekg(0, std::ios::beg);
 
     // read in the text
     text.resize(length);
     ifs.read((char *) &text[0], length);
     ifs.close();
 
     // make number_of_doubles copies of the text
     // sizes_to_test should hold [1, 2, ...., number_of_doubles] * text.size()
     // text should hould number_of_doubles copies of text
     std::vector<uint> sizes_to_test;
     sizes_to_test.push_back(text.size());
 
     for (int i = 0; i < number_of_doubles; ++i) 
     {
         text.insert(text.end(), text.begin(), text.end());
         sizes_to_test.push_back(text.size());
     }
 
     // allocate host arrays
     std::vector<unsigned char> text_gpu(text.size());
     std::vector<unsigned char> text_host(text.size());
 
     // Compute the size of the arrays in bytes for memory allocation.
     // We need enough padding so that the uint2 access won't be out of bounds.
     const int num_bytes_alloc = (text.size() + 7) * sizeof(unsigned char);
 
     // pointers to device arrays
     unsigned char *device_input_array  = nullptr;
     unsigned char *device_output_array = nullptr;
 
     // cudaMalloc device arrays
     cudaMalloc((void **) &device_input_array,  num_bytes_alloc);
     cudaMalloc((void **) &device_output_array, num_bytes_alloc);
 
     // set the padding to 0 to avoid overflow.
     cudaMemset(device_input_array + text.size(), 0, num_bytes_alloc - text.size());
 
     // if either memory allocation failed, report an error message
     if (!device_input_array || !device_output_array) 
     {
         std::cerr << "Couldn't allocate memory!" << std::endl;
         return 1;
     }
 
     // generate random shift in interval [1, 25]
     unsigned char shift_amount = (rand() % 25) + 1;
 
     // Size of text in bytes. This is the largest size that was allocated.
     const int num_bytes = text.size() * sizeof(unsigned char);
 
     // copy input to GPU
     {
         event_pair timer;
         start_timer(&timer);
         cudaMemcpy(device_input_array, &text[0], text.size(), cudaMemcpyHostToDevice);
         check_launch("copy to gpu");
 
         double elapsed_time_h2d = stop_timer(&timer);
         std::cout << "Host -> Device transfer bandwidth " 
                   << num_bytes / (elapsed_time_h2d / 1000.) / 1E9 
                   << std::endl << std::endl;
     }
 
     // generate reference output
     {
         event_pair timer;
         start_timer(&timer);
         host_shift(text, text_host, shift_amount);
         double elapsed_time_host = stop_timer(&timer);
         std::cout << "Host (reference) solution bandwidth GB/sec: " 
                   << 2 * num_bytes / (elapsed_time_host / 1000.) / 1E9 
                   << std::endl << std::endl;
     }
 
     // CUDA block size
     std::cout << std::setw(45) << "Device Bandwidth GB/sec" << std::endl;
 
     std::cout << std::setw(70) << std::setfill('-') << " " 
               << std::endl << std::setfill(' ');
 
     std::cout << std::setw(15) << " " << std::setw(15) << "char" 
               << std::setw(15) << "uint" << std::setw(15) 
               << "uint2" << std::endl;
               
     std::cout << std::setw(15) << "Problem Size MB" << std::endl;

     for (int i=0; i<100; i++) {
       std::cout<<text_host[i];
     }

     return 0;
}

