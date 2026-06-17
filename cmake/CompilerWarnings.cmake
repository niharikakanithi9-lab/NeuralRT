function(neuralrt_set_warnings target_name)
    target_compile_options(${target_name} PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
        -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual
        -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion
    )
endfunction()