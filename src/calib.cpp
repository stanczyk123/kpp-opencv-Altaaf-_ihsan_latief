#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static const char *OUTPUT_PATH = "yaml/kalibrasi.yaml";
static const int MIN_FRAMES = 50;

static bool saveYaml(const cv::Mat &K, const cv::Mat &D, const cv::Size &imageSize,
                     double reprojectionError, int framesUsed)
{
    cv::FileStorage fs(OUTPUT_PATH, cv::FileStorage::WRITE);
    if (!fs.isOpened())
    {
        std::cerr << "Gagal menulis " << OUTPUT_PATH
                  << " (jalankan dari folder utama proyek dan pastikan folder yaml/ ada)" << std::endl;
        return false;
    }

    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "frames_used" << framesUsed;
    fs << "camera_matrix" << K;
    fs << "fx" << K.at<double>(0, 0);
    fs << "fy" << K.at<double>(1, 1);
    fs << "cx" << K.at<double>(0, 2);
    fs << "cy" << K.at<double>(1, 2);
    fs << "dist_coeffs" << D; // [k1 k2 p1 p2 k3]
    fs << "reprojection_error" << reprojectionError;
    fs.release();
    return true;
}

static bool runCalibration(const std::vector<std::vector<cv::Point2f>> &imagePoints,
                           const cv::Size &boardSize, float squareSize, const cv::Size &imageSize)
{
    // Titik 3D chessboard di dunia nyata (bidang Z = 0), urutan baris demi baris
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < boardSize.height; i++)
        for (int j = 0; j < boardSize.width; j++)
            objp.push_back(cv::Point3f(j * squareSize, i * squareSize, 0.0f));

    std::vector<std::vector<cv::Point3f>> objectPoints(imagePoints.size(), objp);

    // Optimasi Levenberg-Marquardt: K, D, dan pose (R_i, t_i) tiap gambar
    cv::Mat K, D;
    std::vector<cv::Mat> rvecs, tvecs;
    double reprojectionError = cv::calibrateCamera(
        objectPoints, imagePoints, imageSize, K, D, rvecs, tvecs);

    std::cout << "\n=== HASIL KALIBRASI ===" << std::endl;
    std::cout << "Jumlah frame      : " << imagePoints.size() << std::endl;
    std::cout << "Reprojection error: " << reprojectionError << " px (RMS)" << std::endl;
    std::cout << "Camera matrix K   :\n" << K << std::endl;
    std::cout << "fx = " << K.at<double>(0, 0) << ", fy = " << K.at<double>(1, 1)
              << ", cx = " << K.at<double>(0, 2) << ", cy = " << K.at<double>(1, 2) << std::endl;
    std::cout << "Distortion D [k1 k2 p1 p2 k3]:\n" << D << std::endl;

    if (!saveYaml(K, D, imageSize, reprojectionError, (int)imagePoints.size()))
        return false;

    std::cout << "Tersimpan ke " << OUTPUT_PATH << std::endl;
    return true;
}

int main(int argc, char **argv)
{
    int cols = 8, rows = 6;
    float squareSize = 23.0f; // mm

    if (argc >= 3)
    {
        cols = std::atoi(argv[1]);
        rows = std::atoi(argv[2]);
    }
    if (argc >= 4)
        squareSize = (float)std::atof(argv[3]);

    if (cols < 2 || rows < 2)
    {
        std::cerr << "Ukuran chessboard tidak valid." << std::endl;
        return -1;
    }
    cv::Size boardSize(cols, rows);

    cv::VideoCapture cap(2);
    if (!cap.isOpened())
    {
        std::cerr << "Webcam tidak bisa dibuka!" << std::endl;
        return -1;
    }
    // Resolusi kalibrasi harus sama dengan resolusi yang dipakai program utama
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    std::cout << "Chessboard: " << cols << "x" << rows << " sudut dalam, kotak " << squareSize << " mm\n"
              << "[s] simpan frame | [c] kalibrasi (min " << MIN_FRAMES << " frame) | [q] keluar\n"
              << "Variasikan posisi, jarak, dan kemiringan papan saat mengambil frame." << std::endl;

    std::vector<std::vector<cv::Point2f>> imagePoints;
    cv::Size imageSize;
    cv::Mat frame;

    while (cap.isOpened())
    {
        cap >> frame;
        if (frame.empty())
            break;

        imageSize = frame.size();

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(
            gray, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);

        if (found)
        {
            // Haluskan posisi sudut sampai akurasi sub-piksel
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
        }

        cv::Mat display = frame.clone();
        if (found)
            cv::drawChessboardCorners(display, boardSize, corners, found);

        std::string info = "Frame: " + std::to_string(imagePoints.size()) + "/" + std::to_string(MIN_FRAMES) +
                           (found ? "  [terdeteksi - tekan s]" : "  [tidak terdeteksi]");
        cv::putText(display, info, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

        cv::imshow("Kalibrasi Webcam", display);

        int key = cv::waitKey(25) & 0xFF;

        if (key == 'q')
            break;

        if (key == 's')
        {
            if (found)
            {
                imagePoints.push_back(corners);
                std::cout << "Frame ke-" << imagePoints.size() << " disimpan." << std::endl;
            }
            else
            {
                std::cout << "Chessboard tidak terdeteksi, frame tidak disimpan." << std::endl;
            }
        }

        if (key == 'c')
        {
            if ((int)imagePoints.size() < MIN_FRAMES)
            {
                std::cout << "Butuh minimal " << MIN_FRAMES << " frame, baru ada "
                          << imagePoints.size() << "." << std::endl;
            }
            else
            {
                runCalibration(imagePoints, boardSize, squareSize, imageSize);
                break;
            }
        }
    }

    return 0;
}