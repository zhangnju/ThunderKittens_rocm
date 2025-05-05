#include "kittens.cuh"
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

namespace ducks {
    namespace epilogue {
        struct identifier {};
    }
}

struct epilogue_load_input {
    using identifier = ducks::epilogue::identifier;
}

struct epilogue {
    using identifier = ducks::epilogue::identifier;
    
    template<typename T>
    T apply(const T &matmul_out) {
        return matmul_out;
    }
};