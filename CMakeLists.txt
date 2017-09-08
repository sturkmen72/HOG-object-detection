cmake_minimum_required(VERSION 2.8)

PROJECT (cpp-example-train_HOG)

find_package(OpenCV  REQUIRED)

ADD_EXECUTABLE (cpp-example-train_HOG train_HOG.cpp)

if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	target_link_libraries( cpp-example-train_HOG ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)