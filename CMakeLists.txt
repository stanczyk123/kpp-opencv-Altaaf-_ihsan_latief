cmake_minimum_required(VERSION 3.10)
project(kpp_opencv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

# Program kalibrasi webcam (menghasilkan yaml/kalibrasi.yaml)
add_executable(calib src/calib.cpp)
target_link_libraries(calib ${OpenCV_LIBS})

# Program utama: multi-object 3D tracker & HUD
add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBS})