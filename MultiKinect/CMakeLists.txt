cmake_minimum_required(VERSION 2.8)
    project( MultiKinect )

    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
        add_definitions(-DHAVE_CLANG)
    endif()

    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
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
    find_package( CVARS 0.1 REQUIRED )
    find_package( Calibu 0.1 REQUIRED )
    find_package( OpenCV2 REQUIRED )
    find_package( Sophus REQUIRED )
    find_package( TBB QUIET )
    find_package( MINIGLOG QUIET )
    find_package( miniglog QUIET )
    find_package( Protobuf REQUIRED )
    find_package( Node QUIET )

    if(TBB_FOUND)
        add_definitions(-DHAVE_TBB)
    endif()

    include_directories( . )
    include_directories( ${Kangaroo_INCLUDE_DIRS} )
    include_directories( ${CVARS_INCLUDE_DIR} )
    include_directories( ${HAL_INCLUDE_DIRS} )
    include_directories( ${SceneGraph_INCLUDE_DIRS} )
    include_directories( ${Pangolin_INCLUDE_DIRS} )
    include_directories( ${Calibu_INCLUDE_DIRS} )
    include_directories( ${OpenCV2_INCLUDE_DIR} )
    include_directories( ${TBB_INCLUDE_DIRS} )
    include_directories( ${Sophus_INCLUDE_DIRS} )
    include_directories( ${MINIGLOG_INCLUDE_DIRS} )
    include_directories( ${Node_INCLUDE_DIRS} )

    link_directories( ${HAL_LINK_DIRS} )
    link_directories( ${Pangolin_LIBRARY_DIRS} )

    add_executable( MKServer server.cpp )

    target_link_libraries( MKServer ${Kangaroo_LIBRARIES})
    target_link_libraries( MKServer ${Node_LIBRARIES} )
    target_link_libraries( MKServer ${HAL_LIBRARIES})
    target_link_libraries( MKServer ${Pangolin_LIBRARIES})
    target_link_libraries( MKServer ${SceneGraph_LIBRARIES} )
    target_link_libraries( MKServer ${CVARS_LIBRARIES} )
    target_link_libraries( MKServer ${Calibu_LIBRARIES} )
    target_link_libraries( MKServer ${MINIGLOG_LIBRARIES} )
    target_link_libraries( MKServer
              ${OpenCV2_LIBRARIES}
                ${PROTOBUF_LIBRARIES}
                  ${MINIGLOG_LIBRARIES})

    if( TBB_FOUND )
      target_link_libraries( MKServer ${TBB_LIBRARIES} )
    endif()

    add_executable( MKClient client.cpp )
    target_link_libraries( MKClient ${Kangaroo_LIBRARIES})
    target_link_libraries( MKClient ${Node_LIBRARIES} )
    target_link_libraries( MKClient ${HAL_LIBRARIES})
    target_link_libraries( MKClient ${Pangolin_LIBRARIES})
    target_link_libraries( MKClient ${SceneGraph_LIBRARIES} )
    target_link_libraries( MKClient ${CVARS_LIBRARIES} )
    target_link_libraries( MKClient ${Calibu_LIBRARIES} )
    target_link_libraries( MKClient ${MINIGLOG_LIBRARIES} )
    target_link_libraries( MKClient
              ${OpenCV2_LIBRARIES}
                ${PROTOBUF_LIBRARIES}
                  ${MINIGLOG_LIBRARIES})

    if( TBB_FOUND )
      target_link_libraries( MKClient ${TBB_LIBRARIES} )
    endif()
