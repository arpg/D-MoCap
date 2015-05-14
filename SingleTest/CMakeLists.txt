cmake_minimum_required(VERSION 2.8)
    project( SingleTest )

    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
        add_definitions(-DHAVE_CLANG)
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++0x")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-unused-variable -Wno-unused-parameter")

    message("CMAKE_C_FLAGS   : ${CMAKE_C_FLAGS}")
    message("CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")

    LIST( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules/" )

    find_package( HAL 0.1 REQUIRED )
    find_package( Kangaroo 0.1 REQUIRED )
    find_package( Pangolin 0.1 REQUIRED )
    find_package( SceneGraph 0.1 REQUIRED )
    find_package( CVars 0.1 REQUIRED )
    find_package( Calibu 0.1 REQUIRED )
    find_package( OpenCV2 REQUIRED )
    find_package( Sophus REQUIRED )
    find_package( TBB QUIET )

    find_package( miniglog QUIET )
    find_package( Protobuf REQUIRED )

    if(TBB_FOUND)
        add_definitions(-DHAVE_TBB)
    endif()
    
    if(CVars_FOUND)
        add_definitions(-DHAVE_CVARS)
    endif()

    include_directories( . )
    include_directories( ${Kangaroo_INCLUDE_DIRS} )
    include_directories( ${CVars_INCLUDE_DIRS} )
    include_directories( ${HAL_INCLUDE_DIRS} )
    include_directories( ${PbMsgs_INCLUDE_DIRS} )
    include_directories( ${SceneGraph_INCLUDE_DIRS} )
    include_directories( ${Pangolin_INCLUDE_DIRS} )
    include_directories( ${Calibu_INCLUDE_DIRS} )
    include_directories( ${OpenCV2_INCLUDE_DIR} )
    include_directories( ${TBB_INCLUDE_DIRS} )
    include_directories( ${Sophus_INCLUDE_DIRS} )
    include_directories( ${MINIGLOG_INCLUDE_DIRS} )
    include_directories( ${Node_INCLUDE_DIRS} )

    link_directories( ${HAL_LINK_DIRS} )
    link_directories( ${PbMsgs_LINK_DIRS} )
    link_directories( ${Pangolin_LIBRARY_DIRS} )

    add_executable( singletest main.cpp )

    target_link_libraries( singletest ${Kangaroo_LIBRARIES})
    target_link_libraries( singletest ${Node_LIBRARIES} )
    target_link_libraries( singletest ${HAL_LIBRARIES} )
    target_link_libraries( singletest ${PbMsgs_LIBRARIES} )
    target_link_libraries( singletest ${Pangolin_LIBRARIES} )
    target_link_libraries( singletest ${SceneGraph_LIBRARIES} )
    target_link_libraries( singletest ${CVars_LIBRARIES} )
    target_link_libraries( singletest ${Calibu_LIBRARIES} )
    target_link_libraries( singletest ${MINIGLOG_LIBRARIES} )
    target_link_libraries( singletest 
      ${OpenCV2_LIBRARIES} ${PROTOBUF_LIBRARIES} ${MINIGLOG_LIBRARIES})

    if( TBB_FOUND )
      target_link_libraries( singletest ${TBB_LIBRARIES} )
    endif()
