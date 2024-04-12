add_executable(tests cfft_forward.cpp rfft_forward.cpp cfft_backward.cpp rfft_backward.cpp twiddles.cpp main.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain SplitRadixFft::SplitRadixFft)
