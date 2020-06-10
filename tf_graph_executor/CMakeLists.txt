cmake_minimum_required(VERSION 2.8.3)
project(tf_graph_executor)

include_directories(/home/janw/anaconda3/envs/tf_gpu_python2/lib/python2.7/site-packages/tensorflow/include)

find_package(catkin_simple 0.1.0 REQUIRED)
catkin_simple()

add_definitions(-std=c++11)

cs_add_library(${PROJECT_NAME} 
  src/tf_graph_executor.cpp
)
target_link_libraries(${PROJECT_NAME})

#find_package(Boost REQUIRED COMPONENTS system thread)

#add_doxygen(REQUIRED)

cs_install()
cs_export()
