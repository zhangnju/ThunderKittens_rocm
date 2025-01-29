// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define TILEN_SWITCH(TILEN, ...)                                               \
  [&] {                                                                        \
    if (TILEN == 128) {                                                        \
      constexpr static int kTileN = 128;                                       \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 144) {                                                 \
      constexpr static int kTileN = 144;                                       \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 160) {                                                 \
      constexpr static int kTileN = 160;                                       \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 176) {                                                 \
      constexpr static int kTileN = 176;                                       \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 192) {                                                 \
      constexpr static int kTileN = 192;                                       \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 208) {                                                 \
      constexpr static int kTileN = 208;                                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define TILEN_SWITCH_32(TILEN, ...)                                            \
  [&] {                                                                        \
    if (TILEN == 128) {                                                        \
      constexpr static int kTileN = 128;                                       \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 160) {                                                 \
      constexpr static int kTileN = 160;                                       \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 192) {                                                 \
      constexpr static int kTileN = 192;                                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define TILEN_SWITCH_32_96_192(TILEN, ...)                                     \
  [&] {                                                                        \
    if (TILEN == 96) {                                                         \
      constexpr static int kTileN = 96;                                        \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 128) {                                                 \
      constexpr static int kTileN = 128;                                       \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 160) {                                                 \
      constexpr static int kTileN = 160;                                       \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 192) {                                                 \
      constexpr static int kTileN = 192;                                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define TILEN_SWITCH_M64_32(TILEN, ...)                                        \
  [&] {                                                                        \
    if (TILEN == 96) {                                                         \
      constexpr static int kTileN = 96;                                        \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 128) {                                                 \
      constexpr static int kTileN = 128;                                       \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 160) {                                                 \
      constexpr static int kTileN = 160;                                       \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 192) {                                                 \
      constexpr static int kTileN = 192;                                       \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 224) {                                                 \
      constexpr static int kTileN = 224;                                       \
      return __VA_ARGS__();                                                    \
    } else if (TILEN == 256) {                                                 \
      constexpr static int kTileN = 256;                                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define TILEM_SWITCH_128_256(TILEM, ...)                                       \
  [&] {                                                                        \
    if (TILEM == 128) {                                                        \
      constexpr static int kTileM = 128;                                       \
      return __VA_ARGS__();                                                    \
    } else if (TILEM == 256) {                                                 \
      constexpr static int kTileM = 256;                                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define TILEM_SWITCH(TILEM, ...)                                               \
  [&] {                                                                        \
    if (TILEM == 256) {                                                        \
      constexpr static int kTileM = 256;                                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define TILEMN_SWITCH_128_256(TILEM, TILEN, ...)                                  \
  [&] {                                                                           \
    if (TILEM == 128) {                                                           \
      constexpr static int kTileM = 128;                                          \
      if (TILEN == 96) {                                                          \
        constexpr static int kTileN = 96;                                         \
        return __VA_ARGS__();                                                     \
      } else if (TILEN == 128) {                                                  \
        constexpr static int kTileN = 128;                                        \
        return __VA_ARGS__();                                                     \
      } else if (TILEN == 160) {                                                  \
        constexpr static int kTileN = 160;                                        \
        return __VA_ARGS__();                                                     \
      } else if (TILEN == 192) {                                                  \
        constexpr static int kTileN = 192;                                        \
        return __VA_ARGS__();                                                     \
      } else if (TILEN == 224) {                                                  \
        constexpr static int kTileN = 224;                                        \
        return __VA_ARGS__();                                                     \
      } else if (TILEN == 256) {                                                  \
        constexpr static int kTileN = 256;                                        \
        return __VA_ARGS__();                                                     \
      }                                                                           \
    } else {                                                                      \
      STATIC_SWITCH_CHECK(false, "Invalid tile configuration Mb=" + std::to_string(TILEM) + " Nb=" + std::to_string(TILEN)); \
    }                                                                             \
  }()

#define CLUSTER_SWITCH(CLUSTERM, CLUSTERN, ...)                                \
  [&] {                                                                        \
    if (CLUSTERM == 1 && CLUSTERN == 2) {                                      \
      constexpr static int kClusterM = 1;                                      \
      constexpr static int kClusterN = 2;                                      \
      return __VA_ARGS__();                                                    \
    } else if (CLUSTERM == 2 && CLUSTERN == 1) {                               \
      constexpr static int kClusterM = 2;                                      \
      constexpr static int kClusterN = 1;                                      \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define CLUSTER_SWITCH_111221(CLUSTERM, CLUSTERN, ...)                         \
  [&] {                                                                        \
    if (CLUSTERM == 1 && CLUSTERN == 1) {                                      \
      constexpr static int kClusterM = 1;                                      \
      constexpr static int kClusterN = 1;                                      \
      return __VA_ARGS__();                                                    \
    } else if (CLUSTERM == 1 && CLUSTERN == 2) {                               \
      constexpr static int kClusterM = 1;                                      \
      constexpr static int kClusterN = 2;                                      \
      return __VA_ARGS__();                                                    \
    } else if (CLUSTERM == 2 && CLUSTERN == 1) {                               \
      constexpr static int kClusterM = 2;                                      \
      constexpr static int kClusterN = 1;                                      \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define TILEN_SWITCH_16_64_LOG2(TILEN, ...)         \
  [&] {                                             \
    if (TILEN == 16)                                \
    {                                               \
      constexpr static int kTileN = 16;             \
      return __VA_ARGS__();                         \
    }                                               \
    else if (TILEN == 32)                           \
    {                                               \
      constexpr static int kTileN = 32;             \
      return __VA_ARGS__();                         \
    }                                               \
    else if (TILEN == 64)                           \
    {                                               \
      constexpr static int kTileN = 64;             \
      return __VA_ARGS__();                         \
    }                                               \
    else if (TILEN == 128)                          \
    {                                               \
      constexpr static int kTileN = 128;            \
      return __VA_ARGS__();                         \
    }                                               \
    else                                            \
    {                                               \
      STATIC_SWITCH_CHECK(false, "Invalid tileN: " + std::to_string(TILEN)); \
    }                                               \
  }()

#define CLUSTERM_SWITCH_124(CLUSTERM, ...)                \
  [&] {                                                   \
    if (CLUSTERM == 1)                                    \
    {                                                     \
      constexpr static int kClusterM = 1;                 \
      constexpr static int kClusterN = 1;                 \
      return __VA_ARGS__();                               \
    }                                                     \
    else if (CLUSTERM == 2)                               \
    {                                                     \
      constexpr static int kClusterM = 2;                 \
      constexpr static int kClusterN = 1;                 \
      return __VA_ARGS__();                               \
    }                                                     \
    else if (CLUSTERM == 4)                               \
    {                                                     \
      constexpr static int kClusterM = 4;                 \
      constexpr static int kClusterN = 1;                 \
      return __VA_ARGS__();                               \
    }                                                     \
    else                                                  \
    {                                                     \
      STATIC_SWITCH_CHECK(false, "Invalid clusterM: " + std::to_string(CLUSTERM)); \
    }                                                     \
  }()

#define STATIC_SWITCH_CHECK(COND, ...) \
  do { \
    if (!(COND)) { \
      throw std::runtime_error(__VA_ARGS__); \
    } \
  } while (0)

// use TORCH_CHECK
