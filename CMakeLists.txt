# #info
MESSAGE(STATUS "operation system arch is ${CMAKE_SYSTEM}")


# #version limit
cmake_minimum_required (VERSION 3.5)


# #c++ 11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


# #option
add_definitions(-fPIC)


# #third party
find_package(Threads)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)


# #include
include_directories(include)
include_directories(/usr/include)
include_directories(/usr/local/include)
include_directories(/usr/local/include/opencv4)

if(EXISTS "/usr/local/cuda-10.2/targets/aarch64-linux")
    include_directories(/usr/local/cuda-10.2/targets/aarch64-linux/include)
else()
    include_directories(/usr/local/cuda-10.2/targets/x86_64-linux/include)
endif()
if(EXISTS "/usr/include/aarch64-linux-gnu")
    include_directories(/usr/include/aarch64-linux-gnu)
else()
    include_directories(/usr/include/x86_64-linux-gnu)
endif()
include_directories(/usr/src/tensorrt/samples/common)


# #libs
link_directories(/usr/lib)
link_directories(/usr/local/lib)
if(EXISTS "/usr/local/cuda-10.2/targets/aarch64-linux/lib")
    link_directories(/usr/local/cuda-10.2/targets/aarch64-linux/lib)
else()
    link_directories(/usr/local/cuda-10.2/targets/x86_64-linux/lib)
endif()
if(EXISTS "/usr/lib/aarch64-linux-gnu")
    link_directories(/usr/lib/aarch64-linux-gnu)
else()
    link_directories(/usr/lib/x86_64-linux-gnu)
endif()


# #add target
add_library(qrcode src/qrcode.cpp)
add_library(detection src/detection.cpp)
add_library(camera src/camera.cpp) 
add_library(measure src/measure.cpp) 

add_library(logger  /usr/src/tensorrt/samples/common/logger.cpp)
add_library(ocr src/ocr.cpp)

add_library(core SHARED src/core.cpp)


# #make link
# workspace link
target_link_libraries(core detection)
target_link_libraries(core qrcode)
target_link_libraries(core measure)
target_link_libraries(core camera)

target_link_libraries(ocr ${CUDA_LIBRARIES})
target_link_libraries(ocr /usr/local/cuda-10.2/lib64/libcudart.so)
if(EXISTS "/usr/lib/aarch64-linux-gnu/libnvinfer.so")
    target_link_libraries(ocr /usr/lib/aarch64-linux-gnu/libnvinfer.so)
else()
    target_link_libraries(ocr /usr/lib/x86_64-linux-gnu/libnvinfer.so)
endif()
if(EXISTS "/usr/lib/aarch64-linux-gnu/libnvonnxparser.so")
    target_link_libraries(ocr /usr/lib/aarch64-linux-gnu/libnvonnxparser.so)
else()
    target_link_libraries(ocr /usr/lib/x86_64-linux-gnu/libnvonnxparser.so)
endif()
target_link_libraries(ocr logger)
target_link_libraries(core ocr)

# third
target_link_libraries(detection ${OpenCV_LIBS})
target_link_libraries(qrcode zbar)
target_link_libraries(camera ${CMAKE_THREAD_LIBS_INIT})

# copy py and configFile
file(COPY server.py DESTINATION ./)
file(COPY icons DESTINATION ./)
file(COPY frames DESTINATION ./)
file(COPY render_frames DESTINATION ./)
file(COPY ONNXmobile DESTINATION ./)
