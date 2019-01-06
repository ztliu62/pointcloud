cmake_minimum_required(VERSION 3.13)
project(pointcloud)

set(CMAKE_CXX_STANDARD 11 )

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")

#find_package(PCL REQUIRED COMPONENT common io filters)
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

add_executable(pointcloud main.cpp)
target_link_libraries( pointcloud ${OpenCV_LIBS} ${PCL_LIBRARIES})
