# AXV Phantom examples

This repository contains runnable C++23 samples built on top of the AXV Phantom
SDK.

## Example

- `axvp_camera_demo` opens a USB camera, sends frames through the SDK pipeline,
  shows the anonymized output in a window, and prints JSON metadata for every
  processed frame.

## Prerequisites

- A local checkout of `axvphantom-sdk` next to this repository, or a custom
  path passed through `-DAXVP_SDK_DIR=/path/to/axvphantom-sdk`.
- The SDK data assets downloaded with `make install-data` in the SDK repo.
- OpenCV with `highgui` and `videoio` support.

## Build

```bash
cmake --preset debug
cmake --build --preset debug
```

If the SDK checkout is not next to this repository, point CMake at it:

```bash
cmake --preset debug -DAXVP_SDK_DIR=/absolute/path/to/axvphantom-sdk
cmake --build --preset debug
```

## Run

```bash
./build/debug/axvp_camera_demo --device 0
```

For an optimized build:

```bash
cmake --preset release
cmake --build --preset release
./build/release/axvp_camera_demo --device 0
```

Useful flags:

- `--width` and `--height` to request a capture resolution.
- `--model-dir` to override the SDK data directory.
- `--detector-model`, `--detector-sha256`, and `--detector-size` to override
  detector validation inputs.
- `--policy none|block|blur|block+blur` to tune the pipeline behavior.

The demo defaults to `AXVP_POLICY_BLUR_FALLBACK` and uses the sibling SDK's
`data/` directory for detector and face-landmark assets.

## CUDA

The demo does not have a separate CUDA-only processing pipeline yet. It still
runs through the SDK, and the SDK chooses its own backend internally.

If you want to launch the example on a CUDA-enabled workstation, keep the run
command the same and pin the NVIDIA device with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 ./build/release/axvp_camera_demo --device 0
```

If your local OpenCV build does not include CUDA support, the example still
works normally on the CPU path.
