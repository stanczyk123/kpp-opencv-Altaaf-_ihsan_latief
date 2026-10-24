#include "../include/tracker.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

class TrackerApp
{
public:
    TrackerApp(const CameraParams &cam, std::vector<ObjectTracker> objects)
        : cam(cam), objects(std::move(objects))
    {
    }

    int run(cv::VideoCapture &cap)
    {
        setupWindows();

        cv::Mat frame;
        auto prev = std::chrono::steady_clock::now();
        double fps = 0.0;

        while (cap.isOpened())
        {
            cap >> frame;
            if (frame.empty())
                break;

            // FPS (dihaluskan agar angkanya tidak loncat-loncat)
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - prev).count();
            prev = now;
            if (dt > 0.0)
                fps = (fps == 0.0) ? 1.0 / dt : 0.9 * fps + 0.1 * (1.0 / dt);

            // Konversi BGR -> HLS (sekali per frame, dipakai oleh semua objek)
            cv::Mat hlsImage;
            cv::cvtColor(frame, hlsImage, cv::COLOR_BGR2HLS);

            // Deteksi 3 objek
            std::vector<Detection> detections;
            for (auto &obj : objects)
                detections.push_back(obj.detect(hlsImage, params, cam));

            // HUD
            cv::Mat hud = frame.clone();
            for (size_t i = 0; i < objects.size(); i++)
                if (detections[i].found)
                    drawHUD(hud, objects[i], detections[i]);

            cv::putText(hud, "FPS: " + formatNumber(fps, 1), cv::Point(10, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(hud, "[s] simpan  [q] keluar", cv::Point(10, hud.rows - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            cv::imshow("Tracker HUD", hud);
            showMasks();

            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27)
                break;
            if (key == 's')
                saveSnapshot(hud, detections);
        }

        return 0;
    }

private:
    CameraParams cam;
    std::vector<ObjectTracker> objects; // tidak boleh di-resize setelah trackbar dibuat
    SharedParams params;

    void setupWindows()
    {
        // Trackbar threshold HLS per objek
        for (auto &obj : objects)
            obj.createTrackbars();

        // Trackbar parameter bersama
        const std::string win = "Parameter";
        cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("Min area", win, &params.minArea, 20000);
        cv::createTrackbar("Erode", win, &params.erodeIter, 15);
        cv::createTrackbar("Dilate", win, &params.dilateIter, 20);
        cv::createTrackbar("Dekat (cm)", win, &params.warnDistCm, 200);
    }

    // Tampilkan mask ketiga objek berdampingan dalam satu jendela
    void showMasks()
    {
        std::vector<cv::Mat> tiles;
        for (const auto &obj : objects)
        {
            const cv::Mat &m = obj.getMask();
            if (m.empty())
                continue;
            cv::Mat small;
            cv::resize(m, small, cv::Size(), 0.5, 0.5);
            cv::putText(small, obj.name, cv::Point(5, 18), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(180), 1);
            tiles.push_back(small);
        }
        if (!tiles.empty())
        {
            cv::Mat combined;
            cv::hconcat(tiles, combined);
            cv::imshow("Mask", combined);
        }
    }

    void drawHUD(cv::Mat &img, const ObjectTracker &obj, const Detection &det)
    {
        bool warning = det.Z < params.warnDistCm;

        // Bounding box + centroid
        cv::rectangle(img, det.box, obj.drawColor, warning ? 4 : 2);
        cv::circle(img, cv::Point((int)det.centroid.x, (int)det.centroid.y), 4,
                   cv::Scalar(255, 255, 255), -1);
        cv::circle(img, cv::Point((int)det.centroid.x, (int)det.centroid.y), 5,
                   cv::Scalar(0, 0, 0), 1);

        // Label: nama, jarak Z (cm), koordinat spasial (X, Y, Z)
        std::string line1 = obj.name + " | Z: " + formatNumber(det.Z, 1) + " cm" +
                            (warning ? "  ! DEKAT" : "");
        std::string line2 = "(X,Y,Z) = (" + formatNumber(det.X, 1) + ", " + formatNumber(det.Y, 1) +
                            ", " + formatNumber(det.Z, 1) + ") cm";

        const int font = cv::FONT_HERSHEY_SIMPLEX;
        const double scale = 0.5;
        int base = 0;
        cv::Size s1 = cv::getTextSize(line1, font, scale, 1, &base);
        cv::Size s2 = cv::getTextSize(line2, font, scale, 1, &base);
        int w = std::max(s1.width, s2.width) + 8;
        int h = s1.height + s2.height + 14;

        // Taruh label di atas bounding box; jika tidak muat, taruh di dalam box bagian atas
        int x0 = std::min(std::max(det.box.x, 0), std::max(0, img.cols - w));
        int y0 = det.box.y - h;
        if (y0 < 0)
            y0 = det.box.y;

        cv::rectangle(img, cv::Rect(x0, y0, w, h), obj.drawColor, cv::FILLED);

        // Warna teks kontras terhadap latar label
        cv::Scalar textColor = (cv::mean(obj.drawColor)[0] > 150) ? cv::Scalar(0, 0, 0)
                                                                  : cv::Scalar(255, 255, 255);
        cv::putText(img, line1, cv::Point(x0 + 4, y0 + s1.height + 3), font, scale, textColor, 1);
        cv::putText(img, line2, cv::Point(x0 + 4, y0 + s1.height + s2.height + 9), font, scale,
                    textColor, 1);
    }

    // Simpan screenshot + tambahkan baris log koordinat 3D ke docs/log_koordinat.csv
    void saveSnapshot(const cv::Mat &hud, const std::vector<Detection> &detections)
    {
        std::filesystem::create_directories("docs");

        std::time_t t = std::time(nullptr);
        std::tm tm = *std::localtime(&t);

        char stamp[32];
        std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm);
        char readable[32];
        std::strftime(readable, sizeof(readable), "%Y-%m-%d %H:%M:%S", &tm);

        // Screenshot
        std::string imgPath = std::string("docs/screenshot_") + stamp + ".png";
        if (cv::imwrite(imgPath, hud))
            std::cout << "Screenshot tersimpan: " << imgPath << std::endl;
        else
            std::cerr << "Gagal menyimpan screenshot ke " << imgPath << std::endl;

        // Log CSV (append). Tulis header jika file baru.
        const std::string csvPath = "docs/log_koordinat.csv";
        bool newFile = !std::filesystem::exists(csvPath);
        std::ofstream csv(csvPath, std::ios::app);
        if (!csv.is_open())
        {
            std::cerr << "Gagal membuka " << csvPath << std::endl;
            return;
        }
        if (newFile)
            csv << "waktu,objek,x_pixel,y_pixel,w_pixel,X_cm,Y_cm,Z_cm\n";

        csv << std::fixed << std::setprecision(2);
        int rows = 0;
        for (size_t i = 0; i < objects.size(); i++)
        {
            const Detection &d = detections[i];
            if (!d.found)
                continue;
            csv << readable << "," << objects[i].name << "," << d.centroid.x << "," << d.centroid.y
                << "," << d.box.width << "," << d.X << "," << d.Y << "," << d.Z << "\n";
            rows++;
        }
        std::cout << "Log koordinat 3D: " << rows << " objek ditambahkan ke " << csvPath << std::endl;
    }

    static std::string formatNumber(double v, int precision)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(precision) << v;
        return oss.str();
    }
};

int main(int argc, char **argv)
{
    // 1. Baca parameter kalibrasi dari YAML (tidak ada nilai fx, fy yang di-hardcode)
    const std::string yamlPath = (argc > 1) ? argv[1] : "yaml/kalibrasi.yaml";

    CameraParams cam;
    if (!cam.load(yamlPath))
    {
        std::cerr << "Jalankan ./calib terlebih dahulu, lalu jalankan ./main dari folder utama proyek."
                  << std::endl;
        return -1;
    }
    std::cout << "Kalibrasi dimuat dari " << yamlPath << ": fx=" << cam.fx << " fy=" << cam.fy
              << " cx=" << cam.cx << " cy=" << cam.cy << std::endl;

    // 2. Definisi 3 objek fisik.
    //    >>> GANTI W_real (cm) dengan hasil pengukuran barang kalian sendiri, dan setel
    //    >>> rentang HLS awal sesuai hasil pencatatan (bisa disetel live lewat trackbar).
    //    Urutan: nama, W_real (cm), warna kotak (BGR), H low/high, L low/high, S low/high
    //    Merah: H low > H high artinya rentang Hue melewati 179 -> 0.
    std::vector<ObjectTracker> objects;
    objects.emplace_back("Botol Biru", 6.5, cv::Scalar(255, 128, 0), 100, 130, 40, 200, 80, 255);
    objects.emplace_back("Pouch Merah", 12.0, cv::Scalar(0, 0, 255), 170, 10, 40, 200, 100, 255);
    objects.emplace_back("Bola Kuning", 7.0, cv::Scalar(0, 255, 255), 20, 35, 60, 210, 100, 255);

    // 3. Buka webcam dengan resolusi yang sama seperti saat kalibrasi
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Webcam tidak bisa dibuka!" << std::endl;
        return -1;
    }
    if (cam.width > 0 && cam.height > 0)
    {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, cam.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, cam.height);

        int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        if (w != cam.width || h != cam.height)
            std::cerr << "PERINGATAN: resolusi webcam " << w << "x" << h
                      << " berbeda dari saat kalibrasi " << cam.width << "x" << cam.height
                      << ". Estimasi Z, X, Y akan meleset." << std::endl;
    }

    // 4. Jalankan aplikasi
    TrackerApp app(cam, std::move(objects));
    return app.run(cap);
}