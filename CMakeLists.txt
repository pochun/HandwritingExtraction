cmake_minimum_required(VERSION 3.11)
project(HandwritingExtraction)

set(CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_definitions("/EHsc")
endif(MSVC)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options)

include_directories(${Boost_INCLUDE_DIRS})
#target_link_libraries(LIB ${OpenCV_LIBS})

add_executable(handwriting_extraction main.cpp)
target_link_libraries(handwriting_extraction ${Boost_LIBRARIES} ${OpenCV_LIBS})
