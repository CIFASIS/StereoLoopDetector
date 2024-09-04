cmake_minimum_required(VERSION 3.8...3.28)
project(StereoLoopDetector 
        VERSION 1.0
        DESCRIPTION "Stereo Loop Detector by Nicolas Soncini"
        LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
include(ExternalProject)
include(FetchContent)

option(BUILD_DemoStereo  "Build demo stereo application" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

## Options
add_compile_options(-Wall -Wextra -Wpedantic)

## Headers
set(HDRS
    include/StereoLoopDetector/DLoopDetector.h
    include/StereoLoopDetector/TemplatedLoopDetector.h
    include/StereoLoopDetector/StereoParameters.h)

## Dependencies
set(DEPENDENCY_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
set(DEPENDENCY_INSTALL_DIR
    ${DEPENDENCY_DIR}/install)
set(DEPENDENCY_CMAKE_ARGS
    -DCMAKE_BUILD_TYPE={CMAKE_BUILD_TYPE})

# OpenCV
if(DEFINED OpenCV_DIR)
    set(DEPENDENCY_CMAKE_ARGS
        ${DEPENDENCY_CMAKE_ARGS} 
        -DOpenCV_DIR=${OpenCV_DIR})
endif()
find_package(OpenCV 4.9.0 REQUIRED
    COMPONENTS sfm)
include_directories(${OpenCV_INCLUDE_DIRS})

# Boost (Needed for option parsing on the stereo demo, logging in the future)
find_package(Boost 1.74.0 REQUIRED 
    COMPONENTS program_options log)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

# Sophus
find_package(Sophus REQUIRED)
if(Sophus_FOUND)
    include_directories(${Sophus_INCLUDE_DIRS})
endif()

# Yaml-cpp (Needed for stereo param parsing on the stereo demo)
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.7.0)
FetchContent_GetProperties(yaml-cpp)
if(NOT yaml-cpp_POPULATED)
  message(STATUS "Fetching yaml-cpp...")
  FetchContent_Populate(yaml-cpp)
  add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
endif()

# TODO: reupload to our servers and replace with DownloadProject macro
# to ease the patch application, which is clunky right now
macro(GetDependency name other_dependency)
  find_package(${name} QUIET
    PATHS ${DEPENDENCY_INSTALL_DIR})
  if(${${name}_FOUND})
    message("${name} library found, using it from the system")
    include_directories(${${name}_INCLUDE_DIRS})
    add_custom_target(${name}_dep)
  else(${${name}_FOUND})
    message("${name} library not found in the system, it will be downloaded on build")
    option(DOWNLOAD_${name}_dependency "Download ${name} dependency" ON)
    if(${DOWNLOAD_${name}_dependency})
      if(NOT ${other_dependency})
        set(dependency ${other_dependency}_dep)
      endif()
      ExternalProject_Add(${name}
        PREFIX ${DEPENDENCY_DIR}
        GIT_REPOSITORY http://github.com/dorian3d/${name}
        GIT_TAG master
        INSTALL_DIR ${DEPENDENCY_INSTALL_DIR}
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> ${DEPENDENCY_CMAKE_ARGS}
        DEPENDS ${dependency})
      add_custom_target(${name}_dep ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR} DEPENDS ${name})
    else()
      message(SEND_ERROR "Please, activate DOWNLOAD_${name}_dependency option or download manually")
    endif(${DOWNLOAD_${name}_dependency})
  endif(${${name}_FOUND})
endmacro(GetDependency)

# DLib
GetDependency(DLib "" "")

# DBoW2
# TODO: incorporate extra optional step of patching to above macro
find_package(DBoW2 QUIET
    PATHS ${DEPENDENCY_INSTALL_DIR})
if(${DBoW2_FOUND})
  message("DBoW2 library found, using it from the system")
  include_directories(${DBoW2_INCLUDE_DIRS})
  add_custom_target(DBoW2_dep)
else(${DBoW2_FOUND})
message("DBoW2 library not found in the system, it will be downloaded on build")
  option(DOWNLOAD_DBoW2_dependency "Download DBoW2 dependency" ON)
  if(${DOWNLOAD_DBoW2_dependency})
    if(NOT ${other_dependency})
      set(dependency ${other_dependency}_dep)
    endif()
    ExternalProject_Add(DBoW2
        PREFIX ${DEPENDENCY_DIR}
        GIT_REPOSITORY http://github.com/dorian3d/DBoW2
        GIT_TAG master
        INSTALL_DIR ${DEPENDENCY_INSTALL_DIR}
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> ${DEPENDENCY_CMAKE_ARGS}
        DEPENDS DLib_dep
        PATCH_COMMAND git --git-dir= apply ${CMAKE_SOURCE_DIR}/resources/DBoW2_PR-43.patch ${CMAKE_SOURCE_DIR}/resources/DBoW2_PR-68.patch)
    add_custom_target(DBoW2_dep ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR} DEPENDS DBoW2)
  else()
    message(SEND_ERROR "Please, activate DOWNLOAD_DBoW2_dependency option or download manually")
  endif(${DOWNLOAD_DBoW2_dependency})
endif(${DBoW2_FOUND})
# GetDependency(DBoW2 DLib 'git --git-dir= apply ${CMAKE_SOURCE_DIR}/resources/DBoW2_PR-43.patch ${CMAKE_SOURCE_DIR}/resources/DBoW2_PR-68.patch')

add_custom_target(Dependencies 
    ${CMAKE_COMMAND} 
    ${CMAKE_SOURCE_DIR} 
    DEPENDS DBoW2_dep DLib_dep)

include_directories(include/StereoLoopDetector/)

## Applications / Binaries
if(BUILD_DemoStereo)
    add_executable(demo_stereo demo/demo_stereo.cpp
                demo/RowMatcher.cpp demo/RowMatcher.hpp
                demo/ORBextractor.cc demo/ORBextractor.h)
    add_dependencies(demo_stereo DLib_dep DBoW2_dep)
    target_link_libraries(demo_stereo 
        ${OpenCV_LIBS}
        ${DLib_LIBS} 
        ${DBoW2_LIBS}
        Boost::program_options
        yaml-cpp::yaml-cpp)
    target_include_directories(demo_stereo PUBLIC 
        ${DLib_INCLUDE_DIRS} 
        ${DBoW2_INCLUDE_DIRS})
    set_target_properties(demo_stereo PROPERTIES CXX_STANDARD 17)
endif(BUILD_DemoStereo)

configure_file(src/StereoLoopDetector.cmake.in
  "${PROJECT_BINARY_DIR}/StereoLoopDetector.cmake" @ONLY)

install(DIRECTORY include/StereoLoopDetector DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/StereoLoopDetector.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES "${PROJECT_BINARY_DIR}/StereoLoopDetector.cmake" 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/StereoLoopDetector/)
install(DIRECTORY ${DEPENDENCY_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL)
