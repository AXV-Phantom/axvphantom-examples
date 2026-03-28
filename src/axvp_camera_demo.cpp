#include "axvphantom/axvphantom.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <array>
#include <charconv>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifndef AXVP_EXAMPLE_DEFAULT_MODEL_DIR
#define AXVP_EXAMPLE_DEFAULT_MODEL_DIR "."
#endif

namespace {

struct Options final {
    int device = 0;
    int width = 1280;
    int height = 720;
    std::string model_dir = AXVP_EXAMPLE_DEFAULT_MODEL_DIR;
    std::optional<std::string> detector_model_path{};
    std::optional<std::string> detector_model_sha256{};
    std::optional<std::size_t> detector_model_size{};
    axvp_policy_t policy = AXVP_POLICY_BLUR_FALLBACK;
};

[[nodiscard]] bool parse_int(std::string_view text, int &value) noexcept {
    int parsed = 0;
    const auto *begin = text.data();
    const auto *end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, parsed);
    if (ec != std::errc{} || ptr != end) {
        return false;
    }

    value = parsed;
    return true;
}

[[nodiscard]] bool parse_size(std::string_view text,
                             std::size_t &value) noexcept {
    std::size_t parsed = 0;
    const auto *begin = text.data();
    const auto *end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, parsed);
    if (ec != std::errc{} || ptr != end) {
        return false;
    }

    value = parsed;
    return true;
}

[[nodiscard]] std::string to_hex(std::span<const std::uint8_t> bytes) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const std::uint8_t byte : bytes) {
        stream << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return stream.str();
}

[[nodiscard]] std::string verdict_to_string(axvp_liveness_verdict_t verdict) {
    switch (verdict) {
    case AXVP_LIVENESS_UNCERTAIN:
        return "UNCERTAIN";
    case AXVP_LIVENESS_SPOOF:
        return "SPOOF";
    case AXVP_LIVENESS_LIVE:
        return "LIVE";
    }

    return "UNKNOWN";
}

[[nodiscard]] std::string verdict_to_string(axvp::fb::LivenessVerdict verdict) {
    return verdict_to_string(static_cast<axvp_liveness_verdict_t>(verdict));
}

[[nodiscard]] std::string status_to_string(axvp_status_t status) {
    switch (status) {
    case AXVP_STATUS_OK:
        return "OK";
    case AXVP_STATUS_INVALID_ARGUMENT:
        return "INVALID_ARGUMENT";
    case AXVP_STATUS_INVALID_CONFIG:
        return "INVALID_CONFIG";
    case AXVP_STATUS_OUT_OF_MEMORY:
        return "OUT_OF_MEMORY";
    case AXVP_STATUS_NOT_INITIALIZED:
        return "NOT_INITIALIZED";
    case AXVP_STATUS_UNSUPPORTED:
        return "UNSUPPORTED";
    case AXVP_STATUS_INTERNAL_ERROR:
        return "INTERNAL_ERROR";
    case AXVP_STATUS_SECURITY_ERROR:
        return "SECURITY_ERROR";
    }

    return "UNKNOWN";
}

[[nodiscard]] std::string policy_to_string(axvp_policy_t policy) {
    if (policy == AXVP_POLICY_NONE) {
        return "none";
    }

    std::vector<std::string_view> parts;
    if ((policy & AXVP_POLICY_BLOCK_ON_FAIL) != AXVP_POLICY_NONE) {
        parts.emplace_back("block");
    }
    if ((policy & AXVP_POLICY_BLUR_FALLBACK) != AXVP_POLICY_NONE) {
        parts.emplace_back("blur");
    }

    std::ostringstream stream;
    for (std::size_t index = 0; index < parts.size(); ++index) {
        if (index > 0U) {
            stream << '+';
        }
        stream << parts[index];
    }
    return stream.str();
}

[[nodiscard]] std::optional<axvp_policy_t>
parse_policy(std::string_view text) noexcept {
    if (text == "none") {
        return AXVP_POLICY_NONE;
    }
    if (text == "block") {
        return AXVP_POLICY_BLOCK_ON_FAIL;
    }
    if (text == "blur") {
        return AXVP_POLICY_BLUR_FALLBACK;
    }
    if (text == "block+blur" || text == "blur+block") {
        return AXVP_POLICY_BLOCK_ON_FAIL | AXVP_POLICY_BLUR_FALLBACK;
    }

    return std::nullopt;
}

[[nodiscard]] std::string face_summary_label(const axvp::fb::FaceRecord &face) {
    const auto *face_id = face.face_id();
    const std::span<const std::uint8_t> id_span(
        face_id == nullptr ? nullptr : face_id->Data(),
        face_id == nullptr ? 0U : face_id->size());
    const std::string id_hex = id_span.empty() ? std::string{} : to_hex(id_span);
    std::ostringstream stream;
    stream << id_hex.substr(0U, 8U) << ' '
           << verdict_to_string(face.verdict()) << ' '
           << std::fixed << std::setprecision(2) << face.liveness_score();
    if (face.pulse_bpm() > 0U) {
        stream << ' ' << face.pulse_bpm() << " bpm";
    }
    return stream.str();
}

void draw_text_box(cv::Mat &image, const std::string &text, cv::Point origin,
                   const cv::Scalar &fg, const cv::Scalar &bg) {
    int baseline = 0;
    const double scale = 0.48;
    const int thickness = 1;
    const cv::Size text_size = cv::getTextSize(
        text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    const cv::Rect box(origin.x, origin.y - text_size.height - 10,
                       text_size.width + 12, text_size.height + 12);
    cv::rectangle(image, box, bg, cv::FILLED, cv::LINE_AA);
    cv::putText(image, text, {origin.x + 6, origin.y - 5},
                cv::FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv::LINE_AA);
}

void draw_metadata_overlay(cv::Mat &image, const axvp::MetadataView &metadata) {
    const std::string summary =
        "frame " + std::to_string(metadata.frame_id()) + " | faces " +
        std::to_string(metadata.faces_detected()) + "/" +
        std::to_string(metadata.faces_anonymized()) + " | complete " +
        (metadata.anonymization_complete() ? "yes" : "no") + " | latency " +
        std::to_string(metadata.processing_latency_us()) + " us";
    draw_text_box(image, summary, {16, 28}, {255, 255, 255},
                  {12, 12, 12});

    const auto faces = metadata.faces();
    for (std::size_t index = 0U; index < faces.size(); ++index) {
        const auto &face = faces[index];
        const auto bbox = face.bbox();
        const cv::Rect rect{
            static_cast<int>(std::lround(bbox.x())),
            static_cast<int>(std::lround(bbox.y())),
            std::max(1, static_cast<int>(std::lround(bbox.width()))),
            std::max(1, static_cast<int>(std::lround(bbox.height()))),
        };

        const cv::Scalar color =
            face.verdict() == axvp::fb::LivenessVerdict_LIVE
                ? cv::Scalar(64, 196, 96)
                : face.verdict() == axvp::fb::LivenessVerdict_SPOOF
                      ? cv::Scalar(64, 64, 224)
                      : cv::Scalar(64, 192, 224);

        cv::rectangle(image, rect, color, 2, cv::LINE_AA);
        draw_text_box(image, face_summary_label(face), rect.tl() + cv::Point(0, -4),
                      {255, 255, 255}, color);
    }
}

void log_metadata_json(const axvp::MetadataView &metadata) {
    std::ostringstream stream;
    stream << std::boolalpha;
    stream << '{'
           << "\"frame_id\":" << metadata.frame_id() << ','
           << "\"timestamp_ns\":" << metadata.timestamp_ns() << ','
           << "\"pipeline_version\":\"" << metadata.pipeline_version() << "\","
           << "\"faces_detected\":" << metadata.faces_detected() << ','
           << "\"faces_anonymized\":" << metadata.faces_anonymized() << ','
           << "\"anonymization_complete\":"
           << metadata.anonymization_complete() << ','
           << "\"processing_latency_us\":" << metadata.processing_latency_us()
           << ','
           << "\"faces\":[";

    const auto faces = metadata.faces();
    for (std::size_t index = 0U; index < faces.size(); ++index) {
        if (index > 0U) {
            stream << ',';
        }

        const auto &face = faces[index];
        const auto bbox = face.bbox();
        const auto *face_id = face.face_id();
        const std::span<const std::uint8_t> id_span(
            face_id == nullptr ? nullptr : face_id->Data(),
            face_id == nullptr ? 0U : face_id->size());

        stream << '{'
               << "\"index\":" << index << ','
               << "\"face_id\":\"" << to_hex(id_span) << "\","
               << "\"bbox\":{"
               << "\"x\":" << bbox.x() << ','
               << "\"y\":" << bbox.y() << ','
               << "\"w\":" << bbox.width() << ','
               << "\"h\":" << bbox.height()
               << "},"
               << "\"liveness_score\":" << face.liveness_score() << ','
               << "\"pulse_bpm\":" << face.pulse_bpm() << ','
               << "\"verdict\":\"" << verdict_to_string(face.verdict())
               << "\","
               << "\"pixels_wiped\":" << face.pixels_wiped()
               << '}';
    }

    stream << "]}";
    std::cout << stream.str() << '\n';
}

[[nodiscard]] std::filesystem::path default_model_dir() {
    return std::filesystem::path{AXVP_EXAMPLE_DEFAULT_MODEL_DIR};
}

void print_usage(const char *program) {
    std::cerr
        << "Usage: " << program
        << " [--device N] [--width W] [--height H] [--model-dir PATH]\n"
        << "                [--detector-model PATH] [--detector-sha256 HEX]\n"
        << "                [--detector-size BYTES] [--policy none|block|blur|block+blur]\n";
}

struct ParseOutcome final {
    std::optional<Options> options{};
    int exit_code = 0;
};

[[nodiscard]] ParseOutcome parse_args(int argc, char **argv) {
    Options options{};
    options.model_dir = default_model_dir().string();

    for (int index = 1; index < argc; ++index) {
        const std::string_view arg = argv[index];
        const auto next_value = [&](std::string_view name)
            -> std::optional<std::string_view> {
            if (index + 1 >= argc) {
                std::cerr << "missing value for " << name << '\n';
                return std::nullopt;
            }
            ++index;
            return std::string_view(argv[index]);
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return {.options = std::nullopt, .exit_code = 0};
        }

        if (arg == "--device") {
            const auto value = next_value(arg);
            if (!value.has_value() || !parse_int(*value, options.device)) {
                std::cerr << "invalid --device value\n";
                return {.options = std::nullopt, .exit_code = 1};
            }
            continue;
        }

        if (arg == "--width") {
            const auto value = next_value(arg);
            if (!value.has_value() || !parse_int(*value, options.width) ||
                options.width <= 0) {
                std::cerr << "invalid --width value\n";
                return {.options = std::nullopt, .exit_code = 1};
            }
            continue;
        }

        if (arg == "--height") {
            const auto value = next_value(arg);
            if (!value.has_value() || !parse_int(*value, options.height) ||
                options.height <= 0) {
                std::cerr << "invalid --height value\n";
                return {.options = std::nullopt, .exit_code = 1};
            }
            continue;
        }

        if (arg == "--model-dir") {
            const auto value = next_value(arg);
            if (!value.has_value()) {
                return {.options = std::nullopt, .exit_code = 1};
            }
            options.model_dir = std::string(*value);
            continue;
        }

        if (arg == "--detector-model") {
            const auto value = next_value(arg);
            if (!value.has_value()) {
                return {.options = std::nullopt, .exit_code = 1};
            }
            options.detector_model_path = std::string(*value);
            continue;
        }

        if (arg == "--detector-sha256") {
            const auto value = next_value(arg);
            if (!value.has_value()) {
                return {.options = std::nullopt, .exit_code = 1};
            }
            options.detector_model_sha256 = std::string(*value);
            continue;
        }

        if (arg == "--detector-size") {
            const auto value = next_value(arg);
            std::size_t parsed = 0U;
            if (!value.has_value() || !parse_size(*value, parsed)) {
                std::cerr << "invalid --detector-size value\n";
                return {.options = std::nullopt, .exit_code = 1};
            }
            options.detector_model_size = parsed;
            continue;
        }

        if (arg == "--policy") {
            const auto value = next_value(arg);
            if (!value.has_value()) {
                return {.options = std::nullopt, .exit_code = 1};
            }
            const auto parsed = parse_policy(*value);
            if (!parsed.has_value()) {
                std::cerr << "invalid --policy value\n";
                return {.options = std::nullopt, .exit_code = 1};
            }
            options.policy = *parsed;
            continue;
        }

        std::cerr << "unknown argument: " << arg << '\n';
        return {.options = std::nullopt, .exit_code = 1};
    }

    return {.options = std::move(options), .exit_code = 0};
}

[[nodiscard]] axvp_config_t make_config(const Options &options,
                                        const cv::Size &size) noexcept {
    axvp_config_t config{};
    config.size = static_cast<std::uint32_t>(sizeof(config));
    config.width = static_cast<std::uint32_t>(size.width);
    config.height = static_cast<std::uint32_t>(size.height);
    config.format = AXVP_FMT_BGR;
    config.policy = options.policy;
    config.device_index = static_cast<std::uint32_t>(options.device);
    config.rppg_window_frames = 24U;
    config.model_dir = options.model_dir.c_str();
    config.detector_model_path =
        options.detector_model_path.has_value()
            ? options.detector_model_path->c_str()
            : nullptr;
    config.detector_model_sha256 =
        options.detector_model_sha256.has_value()
            ? options.detector_model_sha256->c_str()
            : nullptr;
    config.detector_model_size = options.detector_model_size.value_or(0U);
    return config;
}

[[nodiscard]] cv::Mat make_output_view(const axvp::Result &result) {
    const axvp_result_t *native = result.native();
    return cv::Mat(static_cast<int>(native->frame.height),
                   static_cast<int>(native->frame.width), CV_8UC3,
                   const_cast<void *>(native->frame.data),
                   static_cast<std::size_t>(native->frame.stride));
}

} // namespace

int main(int argc, char **argv) {
    const auto parsed = parse_args(argc, argv);
    if (!parsed.options.has_value()) {
        return parsed.exit_code;
    }

    const Options &options = *parsed.options;

    cv::VideoCapture capture(options.device, cv::CAP_ANY);
    if (!capture.isOpened()) {
        std::cerr << "failed to open camera device " << options.device << '\n';
        return 2;
    }

    capture.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(options.width));
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(options.height));

    cv::Mat first_frame;
    if (!capture.read(first_frame) || first_frame.empty()) {
        std::cerr << "failed to read initial frame from camera\n";
        return 3;
    }

    const axvp_config_t config = make_config(options, first_frame.size());
    auto context = axvp::Context::create(config);
    if (!context.has_value()) {
        std::cerr << "failed to create SDK context: "
                  << status_to_string(context.error()) << '\n';
        return 4;
    }

    if (const axvp_status_t policy_status =
            context->set_policy(options.policy);
        policy_status != AXVP_STATUS_OK) {
        std::cerr << "failed to set policy: "
                  << status_to_string(policy_status) << '\n';
        return 5;
    }

    std::cout << "using camera device " << options.device << " at "
              << first_frame.cols << 'x' << first_frame.rows
              << ", policy=" << policy_to_string(options.policy) << '\n';

    constexpr std::string_view kWindowName = "AXV Phantom camera demo";
    cv::namedWindow(std::string(kWindowName), cv::WINDOW_NORMAL);
    cv::resizeWindow(std::string(kWindowName), first_frame.cols,
                     first_frame.rows);

    auto process_and_show = [&](cv::Mat frame) -> bool {
        axvp::Frame input(frame, static_cast<std::uint64_t>(
                                     std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now()
                                             .time_since_epoch())
                                         .count()));

        auto processed = context->process(input);
        if (!processed.has_value()) {
            std::cerr << "process_frame failed: "
                      << status_to_string(processed.error()) << '\n';
            return false;
        }

        auto metadata = axvp::MetadataView::create(*processed->native());
        if (!metadata.has_value()) {
            std::cerr << "metadata parse failed: "
                      << status_to_string(metadata.error()) << '\n';
            return false;
        }

        cv::Mat output = make_output_view(*processed).clone();
        draw_metadata_overlay(output, *metadata);
        log_metadata_json(*metadata);

        cv::imshow(std::string(kWindowName), output);
        return true;
    };

    if (!process_and_show(first_frame)) {
        return 6;
    }

    for (;;) {
        cv::Mat frame;
        if (!capture.read(frame) || frame.empty()) {
            std::cerr << "camera stream ended\n";
            break;
        }

        if (!process_and_show(frame)) {
            continue;
        }

        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
    }

    return 0;
}
