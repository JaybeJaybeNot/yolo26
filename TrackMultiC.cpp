#include <Python.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

std::filesystem::path root;
const std::vector<std::string> videos = {
    "[BLUR_TEST]RoboMasterTestVideo3.mp4",
    "RoboMasterTestVideo2.mp4",
};

class plate {
public:
    plate(double xValue, double yValue) : x(xValue), y(yValue) {}
    void move(double dx, double dy) {
        x += dx;
        y += dy;
    }

    double x;
    double y;
};

class robot {
public:
    robot(double xValue, double yValue) : x(xValue), y(yValue) {}
    void move(double dx, double dy) {
        x += dx;
        y += dy;
    }

    double x;
    double y;
    std::string color;
    plate* plates = nullptr;
};

class Track {
public:
    Track(double x1, double y1, double x2, double y2, int marginValue = 80)
        : x((x1 + x2) / 2.0),
          y((y1 + y2) / 2.0),
          width(x2 - x1),
          height(y2 - y1),
          margin(marginValue),
          missing_frames(0) {}

    void update(double x1, double y1, double x2, double y2) {
        x = (x1 + x2) / 2.0;
        y = (y1 + y2) / 2.0;
        width = x2 - x1;
        height = y2 - y1;
        missing_frames = 0;
    }

    std::array<int, 4> get_roi(int frame_width, int frame_height) const {
        return {
            std::max(0, static_cast<int>(x - width / 2.0 - margin)),
            std::max(0, static_cast<int>(y - height / 2.0 - margin)),
            std::min(frame_width, static_cast<int>(x + width / 2.0 + margin)),
            std::min(frame_height, static_cast<int>(y + height / 2.0 + margin)),
        };
    }

    double x;
    double y;
    double width;
    double height;
    int margin;
    int missing_frames;
};

std::vector<std::tuple<std::string, cv::Rect>> color_detection(
    const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    std::vector<std::tuple<std::string, cv::Scalar, cv::Scalar>> color_ranges = {
        {"red", cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255)},
        {"red", cv::Scalar(170, 100, 100), cv::Scalar(180, 255, 255)},
        {"blue", cv::Scalar(100, 100, 100), cv::Scalar(130, 255, 255)},
    };

    std::vector<std::tuple<std::string, cv::Rect>> detections;
    for (const std::string& color : {std::string("red"), std::string("blue")}) {
        cv::Mat mask = cv::Mat::zeros(hsv.size(), CV_8UC1);
        for (const auto& range : color_ranges) {
            if (std::get<0>(range) == color) {
                cv::Mat range_mask;
                cv::inRange(
                    hsv, std::get<1>(range), std::get<2>(range), range_mask);
                cv::bitwise_or(mask, range_mask, mask);
            }
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(
            mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& contour : contours) {
            if (cv::contourArea(contour) > 100.0) {
                detections.emplace_back(color, cv::boundingRect(contour));
            }
        }
    }
    return detections;
}

std::vector<std::pair<std::string, cv::Point>> grouping(
    const cv::Mat& image,
    const std::vector<std::array<float, 4>>& plate_boxes,
    std::string enemyColor,
    std::string allyColor,
    double maxPixDistance) {
    std::transform(
        enemyColor.begin(), enemyColor.end(), enemyColor.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(
        allyColor.begin(), allyColor.end(), allyColor.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if ((enemyColor != "red" && enemyColor != "blue") ||
        (allyColor != "red" && allyColor != "blue")) {
        throw std::invalid_argument(
            "enemyColor and allyColor must be 'red' or 'blue'");
    }
    if (enemyColor == allyColor) {
        throw std::invalid_argument(
            "enemyColor and allyColor must be different");
    }
    if (maxPixDistance < 0.0) {
        throw std::invalid_argument("maxPixDistance must be non-negative");
    }
    if (image.empty() || plate_boxes.empty()) {
        return {};
    }

    auto color_boxes = color_detection(image);
    color_boxes.erase(
        std::remove_if(
            color_boxes.begin(), color_boxes.end(),
            [&](const auto& entry) {
                return std::get<0>(entry) != enemyColor &&
                       std::get<0>(entry) != allyColor;
            }),
        color_boxes.end());
    if (color_boxes.empty()) {
        return {};
    }

    using Candidate =
        std::tuple<double, std::size_t, std::size_t, std::string, cv::Point>;
    std::vector<Candidate> candidates;
    for (std::size_t plate_index = 0; plate_index < plate_boxes.size();
         ++plate_index) {
        const auto& plate_box = plate_boxes[plate_index];
        const cv::Point center(
            static_cast<int>((plate_box[0] + plate_box[2]) / 2.0F),
            static_cast<int>((plate_box[1] + plate_box[3]) / 2.0F));

        for (std::size_t region_index = 0; region_index < color_boxes.size();
             ++region_index) {
            const std::string& color = std::get<0>(color_boxes[region_index]);
            const cv::Rect& region = std::get<1>(color_boxes[region_index]);
            const double dx = std::max({
                static_cast<double>(plate_box[0] - (region.x + region.width)),
                static_cast<double>(region.x - plate_box[2]),
                0.0});
            const double dy = std::max({
                static_cast<double>(plate_box[1] - (region.y + region.height)),
                static_cast<double>(region.y - plate_box[3]),
                0.0});
            const double distance = std::sqrt(dx * dx + dy * dy);
            if (distance <= maxPixDistance) {
                candidates.emplace_back(
                    distance, plate_index, region_index,
                    color == enemyColor ? "enemy" : "ally", center);
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& left,
                                                        const Candidate& right) {
        return std::tie(
                   std::get<0>(left), std::get<1>(left), std::get<2>(left)) <
               std::tie(
                   std::get<0>(right), std::get<1>(right), std::get<2>(right));
    });
    std::vector<bool> matched_plates(plate_boxes.size(), false);
    std::vector<bool> matched_regions(color_boxes.size(), false);
    std::vector<std::pair<std::string, cv::Point>> matches;
    for (const auto& candidate : candidates) {
        const std::size_t plate_index = std::get<1>(candidate);
        const std::size_t region_index = std::get<2>(candidate);
        if (matched_plates[plate_index] || matched_regions[region_index]) {
            continue;
        }
        matched_plates[plate_index] = true;
        matched_regions[region_index] = true;
        matches.emplace_back(std::get<3>(candidate), std::get<4>(candidate));
    }
    return matches;
}

std::size_t track_video(
    PyObject* model,
    std::filesystem::path video_path,
    const std::string& enemyColor,
    const std::string& allyColor,
    double maxPixDistance) {
    if (!video_path.is_absolute()) {
        video_path = root / video_path;
    }
    std::size_t frame_count = 0;
    const auto start_time = std::chrono::steady_clock::now();

    std::mutex jobs_mutex;
    std::condition_variable jobs_condition;
    std::deque<std::function<void()>> jobs;
    bool stop_worker = false;
    std::thread worker([&] {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(jobs_mutex);
                jobs_condition.wait(
                    lock, [&] { return stop_worker || !jobs.empty(); });
                if (stop_worker && jobs.empty()) {
                    return;
                }
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            job();
        }
    });

    using Matches = std::vector<std::pair<std::string, cv::Point>>;
    std::deque<std::pair<std::size_t, std::future<Matches>>> pending;
    PyObject* predict = PyObject_GetAttrString(model, "predict");
    PyObject* arguments = PyTuple_New(0);
    PyObject* keywords = PyDict_New();
    PyDict_SetItemString(
        keywords, "source", PyUnicode_FromString(video_path.string().c_str()));
    PyDict_SetItemString(keywords, "imgsz", PyLong_FromLong(224));
    PyDict_SetItemString(keywords, "conf", PyFloat_FromDouble(0.1));
    const std::string tracker_path = (root / "byteTrackConfig.yaml").string();
    PyDict_SetItemString(
        keywords, "tracker", PyUnicode_FromString(tracker_path.c_str()));
    PyDict_SetItemString(keywords, "save", PyBool_FromLong(0));
    PyDict_SetItemString(keywords, "device", PyLong_FromLong(0));
    PyDict_SetItemString(keywords, "stream", PyBool_FromLong(1));
    PyObject* results = PyObject_Call(predict, arguments, keywords);
    Py_DECREF(predict);
    Py_DECREF(arguments);
    Py_DECREF(keywords);
    if (results == nullptr) {
        {
            std::lock_guard<std::mutex> lock(jobs_mutex);
            stop_worker = true;
        }
        jobs_condition.notify_one();
        worker.join();
        throw std::runtime_error("Ultralytics model.predict failed");
    }

    PyObject* iterator = PyObject_GetIter(results);
    if (iterator == nullptr) {
        Py_DECREF(results);
        {
            std::lock_guard<std::mutex> lock(jobs_mutex);
            stop_worker = true;
        }
        jobs_condition.notify_one();
        worker.join();
        throw std::runtime_error("Could not iterate over YOLO results");
    }

    try {
        PyObject* result = nullptr;
        while ((result = PyIter_Next(iterator)) != nullptr) {
            ++frame_count;

            PyObject* boxes = PyObject_GetAttrString(result, "boxes");
            PyObject* xyxy = PyObject_GetAttrString(boxes, "xyxy");
            PyObject* cpu_boxes = PyObject_CallMethod(xyxy, "cpu", nullptr);
            PyObject* numpy_boxes =
                PyObject_CallMethod(cpu_boxes, "numpy", nullptr);
            PyObject* rows = PySequence_Fast(
                numpy_boxes, "YOLO boxes must be a sequence of rows");
            std::vector<std::array<float, 4>> plate_boxes;
            const Py_ssize_t row_count = PySequence_Fast_GET_SIZE(rows);
            for (Py_ssize_t row_index = 0; row_index < row_count; ++row_index) {
                PyObject* row = PySequence_Fast(
                    PySequence_Fast_GET_ITEM(rows, row_index),
                    "YOLO box row must be a sequence");
                std::array<float, 4> box{};
                for (Py_ssize_t coordinate = 0; coordinate < 4; ++coordinate) {
                    box[static_cast<std::size_t>(coordinate)] =
                        static_cast<float>(PyFloat_AsDouble(
                            PySequence_Fast_GET_ITEM(row, coordinate)));
                }
                Py_DECREF(row);
                plate_boxes.push_back(box);
            }
            Py_DECREF(rows);
            Py_DECREF(numpy_boxes);
            Py_DECREF(cpu_boxes);
            Py_DECREF(xyxy);
            Py_DECREF(boxes);

            PyObject* original_image =
                PyObject_GetAttrString(result, "orig_img");
            Py_buffer image_buffer{};
            if (PyObject_GetBuffer(
                    original_image, &image_buffer,
                    PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) != 0) {
                Py_DECREF(original_image);
                Py_DECREF(result);
                throw std::runtime_error(
                    "Could not access YOLO's original frame");
            }
            if (image_buffer.ndim != 3 || image_buffer.shape[2] != 3) {
                PyBuffer_Release(&image_buffer);
                Py_DECREF(original_image);
                Py_DECREF(result);
                throw std::runtime_error(
                    "YOLO's original frame must have shape (height, width, 3)");
            }
            cv::Mat frame(
                static_cast<int>(image_buffer.shape[0]),
                static_cast<int>(image_buffer.shape[1]),
                CV_8UC3,
                image_buffer.buf,
                static_cast<std::size_t>(image_buffer.strides[0]));
            cv::Mat frame_copy = frame.clone();
            PyBuffer_Release(&image_buffer);
            Py_DECREF(original_image);
            Py_DECREF(result);

            auto task = std::make_shared<std::packaged_task<Matches()>>(
                [frame_copy = std::move(frame_copy),
                 plate_boxes = std::move(plate_boxes),
                 enemyColor,
                 allyColor,
                 maxPixDistance]() {
                    return grouping(
                        frame_copy, plate_boxes, enemyColor, allyColor,
                        maxPixDistance);
                });
            std::future<Matches> future = task->get_future();
            {
                std::lock_guard<std::mutex> lock(jobs_mutex);
                jobs.emplace_back([task] { (*task)(); });
            }
            jobs_condition.notify_one();
            pending.emplace_back(frame_count, std::move(future));

            if (pending.size() > 2) {
                const std::size_t frame_id = pending.front().first;
                Matches matches = pending.front().second.get();
                pending.pop_front();
                for (const auto& match : matches) {
                    std::cout << "Frame: " << frame_id << " | Team: "
                              << match.first << " | Center: (" << match.second.x
                              << ", " << match.second.y << ")\n";
                }
            }
        }
        if (PyErr_Occurred()) {
            PyErr_Print();
            throw std::runtime_error("Error while reading YOLO results");
        }

        while (!pending.empty()) {
            const std::size_t frame_id = pending.front().first;
            Matches matches = pending.front().second.get();
            pending.pop_front();
            for (const auto& match : matches) {
                std::cout << "Frame: " << frame_id << " | Team: " << match.first
                          << " | Center: (" << match.second.x << ", "
                          << match.second.y << ")\n";
            }
        }
    } catch (...) {
        Py_DECREF(iterator);
        Py_DECREF(results);
        {
            std::lock_guard<std::mutex> lock(jobs_mutex);
            stop_worker = true;
        }
        jobs_condition.notify_one();
        worker.join();
        throw;
    }

    Py_DECREF(iterator);
    Py_DECREF(results);
    {
        std::lock_guard<std::mutex> lock(jobs_mutex);
        stop_worker = true;
    }
    jobs_condition.notify_one();
    worker.join();

    const double elapsed =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time)
            .count();
    if (elapsed > 0.0) {
        std::cout << "Tracking speed for " << video_path.filename().string()
                  << ": " << static_cast<double>(frame_count) / elapsed
                  << " FPS (" << frame_count << " frames over " << elapsed
                  << "s)\n";
    }
    return frame_count;
}

}  // namespace

int main(int, char* argv[]) {
    root = std::filesystem::absolute(argv[0]).parent_path();
    const std::filesystem::path model_path =
        root / "runs/detect/[LARGE]Robomasters.v6i.yolo26_EPOCH100/weights/best.engine";
    const std::string enemyColor = "red";
    const std::string allyColor = "blue";
    constexpr double maxPixDistance = 80.0;

    Py_Initialize();
    if (!Py_IsInitialized()) {
        std::cerr << "Could not initialize Python for Ultralytics YOLO\n";
        return 1;
    }

    PyObject* ultralytics = PyImport_ImportModule("ultralytics");
    if (ultralytics == nullptr) {
        PyErr_Print();
        Py_FinalizeEx();
        return 1;
    }
    PyObject* yolo_class = PyObject_GetAttrString(ultralytics, "YOLO");
    Py_DECREF(ultralytics);
    PyObject* model_argument =
        PyUnicode_FromString(model_path.string().c_str());
    PyObject* model_arguments = PyTuple_Pack(1, model_argument);
    Py_DECREF(model_argument);
    PyObject* model = PyObject_CallObject(yolo_class, model_arguments);
    Py_DECREF(model_arguments);
    Py_DECREF(yolo_class);
    if (model == nullptr) {
        PyErr_Print();
        Py_FinalizeEx();
        return 1;
    }

    try {
        for (const std::string& video : videos) {
            track_video(
                model, video, enemyColor, allyColor, maxPixDistance);
        }
    } catch (const std::exception& error) {
        if (PyErr_Occurred()) {
            PyErr_Print();
        }
        std::cerr << "TrackMultiC: " << error.what() << '\n';
        Py_DECREF(model);
        Py_FinalizeEx();
        return 1;
    }

    Py_DECREF(model);
    return Py_FinalizeEx() < 0 ? 1 : 0;
}
