
if(${CMAKE_VERSION} VERSION_LESS "3.13.0")
    message("PCL example requires cmake 3.13")
else()
    enable_language(C) # For CMake 3.31 or newer (fix the import of PCL)

    find_package(PCL 1.6 QUIET)
    if(PCL_FOUND)
        set(CMAKE_INCLUDE_CURRENT_DIR ON)

        set(ponca_pclwrapper_SRCS
                main.cpp
                pcl_wrapper.cpp
                pcl_wrapper.hpp
                pcl_wrapper.h
        )
        add_executable(ponca_pclwrapper ${ponca_pclwrapper_SRCS})

        # Some version of PCL does link to Boost::thread but not propagate this information
        find_package(Boost QUIET COMPONENTS thread)
        if(Boost_FOUND)
            target_link_libraries(ponca_pclwrapper PUBLIC ${Boost_LIBRARIES})
        endif()
        ponca_handle_eigen_dependency(ponca_pclwrapper)

        add_custom_command( TARGET ponca_pclwrapper POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E copy
                                ${CMAKE_CURRENT_SOURCE_DIR}/bun_zipper.ply
                                $<TARGET_FILE_DIR:ponca_pclwrapper>
                            MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/bun_zipper.ply
                            BYPRODUCTS bun_zipper.ply
                            COMMENT "Copying ponca_pclwrapper dataset"
            )

        target_link_directories(ponca_pclwrapper PUBLIC ${PCL_LIBRARY_DIRS})
#        target_compile_definitions(ponca_pclwrapper PUBLIC ${PCL_DEFINITIONS})
        target_link_libraries(ponca_pclwrapper PUBLIC ${PCL_LIBRARIES})
        target_include_directories(ponca_pclwrapper PRIVATE ${PONCA_src_ROOT})
        target_include_directories(ponca_pclwrapper PUBLIC ${PCL_INCLUDE_DIRS})
        add_dependencies(ponca-examples ponca_pclwrapper)
    endif()
endif()
