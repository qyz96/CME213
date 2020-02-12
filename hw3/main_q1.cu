/* This is machine problem 1, part 1, shift problem
 *
 * The problem is to take in a string (a vector of characters) and a shift amount,
 * and add that number to each element of
 * the string, effectively "shifting" each element in the
 * string.
 *
 * We do this in three different ways:
 * 1. With a cuda kernel loading chars and outputting chars for each thread
 * 2. With a cuda kernel, casting the character pointer to an int so that
 *    we load and store 4 bytes each time instead of 1 which gives us better coalescing
 *    and uses the memory effectively to achieve higher bandwidth
 * 3. Same spiel except with a uint2, so that we load 8 bytes each time
 *
 */

 #include <algorithm>
 #include <cstdlib>
 #include <iostream>
 #include <iomanip>
 #include <ctime>
 #include <fstream>
 #include <vector>
 
 #include "util.cuh"
 #include "shift.cuh"
 
 
 constexpr const char *MOBY_DICK = "mobydick.txt";
 constexpr const int CUDA_BLOCK_SIZE = 256;
 
 void host_shift(std::vector<unsigned char> &input_array, 
                 std::vector<unsigned char> &output_array,
                 unsigned char shift_amount) 
 {
     std::transform(input_array.begin(), input_array.end(), output_array.begin(),
         [&shift_amount](unsigned char& element) {
             return element + shift_amount;
         }
     );
 }
 
 void checkResults(std::vector<unsigned char> &text_host, 
                   unsigned char *device_output_array,
                   unsigned int num_entries, 
                   const char *type) 
 {
     // allocate space on host for gpu results
     std::vector<unsigned char> text_from_gpu(num_entries);
 
     // download and inspect the result on the host:
     cudaMemcpy(&text_from_gpu[0], device_output_array, num_entries,
                cudaMemcpyDeviceToHost);
     check_launch("copy from gpu");
 
     // check CUDA output versus reference output
     int error = 0;
 
     for (unsigned int i = 0; i < num_entries; i++) 
     {
         if (text_host[i] == text_from_gpu[i])
             continue;
 
         ++error;
         std::cerr << "Mismatch at pos: " << i << std::endl
                   << "Expected " << static_cast<int>(text_host[i])
                   << " and got " << static_cast<int>(text_from_gpu[i])
                   << std::endl;
 
         if (error > 10) 
         {
             std::cerr << std::endl << "Too many errors, quitting..." << std::endl;
             break;
         }
     }
 
     if (error) 
     {
         std::cerr << "\nError(s) in " << type << " kernel!" << std::endl;
         exit(1);
     }
 }
 
 int main(int argc, char** argv) 
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
 
     // Loop through all the problem sizes and generate timing / bandwidth information for each
     // and also check correctness
     for (const uint size_to_test : sizes_to_test) 
     {
         // generate GPU char output
         double elapsed_time_char = doGPUShiftChar(device_input_array,
                                    device_output_array, shift_amount, size_to_test, CUDA_BLOCK_SIZE);
         checkResults(text_host, device_output_array, size_to_test, "char");
 
         // make sure we don't falsely say the next kernel is correct because we've left the correct answer sitting in memory
         cudaMemset(device_output_array, 0, size_to_test);
 
         // generate GPU uint output
         double elapsed_time_uint = doGPUShiftUInt(device_input_array,
                                    device_output_array, shift_amount, size_to_test, CUDA_BLOCK_SIZE);
         checkResults(text_host, device_output_array, size_to_test, "uint");
 
         // make sure we don't falsely say the next kernel is correct because we've left the correct answer sitting in memory
         cudaMemset(device_output_array, 0, size_to_test);
 
         // generate GPU uint2 output
         double elapsed_time_uint2 = doGPUShiftUInt2(device_input_array,
                                     device_output_array, shift_amount, size_to_test, CUDA_BLOCK_SIZE);
         checkResults(text_host, device_output_array, size_to_test, "uint2");
 
         // make sure we don't falsely say the next kernel is correct because we've left the correct answer sitting in memory
         cudaMemset(device_output_array, 0, size_to_test);
 
         std::cout << std::setw(15) << size_to_test / 1E6 << " " 
                   << std::setw(15) << 2 * size_to_test / (elapsed_time_char / 1000.) / 1E9 
                   << std::setw(15) << 2 * size_to_test / (elapsed_time_uint / 1000.) / 1E9
                   << std::setw(15) << 2 * size_to_test / (elapsed_time_uint2 / 1000.) / 1E9
                   << std::endl;
     }
 
     // deallocate memory
     cudaFree(device_input_array);
     cudaFree(device_output_array);
 
     return exit_code;
 }
 