project(Roboman-SDM)
cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
include_directories("src/include")
aux_source_directory("src/include" SOURCES_FILE)
ADD_EXECUTABLE(Roboman-SDM-TEST src/test_model.cpp)
TARGET_LINK_LIBRARIES(Roboman-SDM-TEST ${OpenCV_LIBS})





