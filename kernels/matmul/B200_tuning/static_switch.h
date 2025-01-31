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

#define TILE_SWITCH_64_128(TILE, NAME, VAR_NAME, ...) \
  [&] { \
    if (TILE == 64) { \
      constexpr static int VAR_NAME = 64; \
      return __VA_ARGS__(); \
    } else if (TILE == 128) { \
      constexpr static int VAR_NAME = 128; \
      return __VA_ARGS__(); \
    } else { \
      STATIC_SWITCH_CHECK(false, "Invalid tile configuration " NAME "=" + std::to_string(TILE)); \
    } \
  }()

#define TILE_SWITCH_64_128_256(TILE, NAME, VAR_NAME, ...) \
  [&] { \
    if (TILE == 64) { \
      constexpr static int VAR_NAME = 64; \
      return __VA_ARGS__(); \
    } else if (TILE == 128) { \
      constexpr static int VAR_NAME = 128; \
      return __VA_ARGS__(); \
    } else if (TILE == 256) { \
      constexpr static int VAR_NAME = 256; \
      return __VA_ARGS__(); \
    } else { \
      STATIC_SWITCH_CHECK(false, "Invalid tile configuration " NAME "=" + std::to_string(TILE)); \
    } \
  }()

// ncta=2
#define TILEM_SWITCH(TILEM, ...) \
  TILE_SWITCH_64_128(TILEM, "Mb", kTileM, __VA_ARGS__)

#define TILEN_SWITCH(TILEN, ...) \
  TILE_SWITCH_64_128_256(TILEN, "Nb", kTileN, __VA_ARGS__)

#define TILEK_SWITCH(TILEK, ...) \
  TILE_SWITCH_64_128(TILEK, "Kb", kTileK, __VA_ARGS__)

#define STATIC_SWITCH_CHECK(COND, ...) \
  do { \
    if (!(COND)) { \
      throw std::runtime_error(__VA_ARGS__); \
    } \
  } while (0)

// use TORCH_CHECK
