#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include "ldmarkmodel.h"

#define EYES_WINDOW_SIZE 36
#define BLOCK_SIZE 12
#define WINDOW_SIZE 24
#define CAMERA_SOURCE 0
#define IR_CAMERA_SOURCE 2
#define RESIZE_DIM 32            // 리사이즈 크기
#define BRIGHTNESS_THRESHOLD 0.06

using namespace std;
using namespace cv;

// 함수 선언
void make_lbp_img(Mat& ref, Mat& result);
float face_verification(Mat& Image, int * ref_histogram);
bool is_eye_opened(const float* magnitude_hog, const int* orientation_hog, int width, int height);
cv::Mat rotateImage(cv::Mat& src, double angle, cv::Point2f center);
double calculateAngle(Point p1, Point p2);
cv::Mat increaseContrast(const cv::Mat& src, double alpha = 3.5, int beta = 0);
void crop_eye_region(Mat& Image, Mat& output_image, Point left_point, Point right_point);
bool is_both_eye_opened(ldmarkmodel& modelt, Mat& Image);
void safe_verification_total_task(int * ref_histogram, float * result_score);
bool is_bright_frame(const Mat& frame);
void displayMenu(void);
void calculate_vector(Mat* image, float* magnitude_hog, int* orientation_hog);
void calculate_histogram(Mat& Image, int* output_histogram);
void register_face(Mat* myface);


int main() {
    bool stop_flag = true;

    int mode;

    Mat ref;
    int ref_histogram[68 * 256] = { 0 };

    float result_score = 0.0;

    while(stop_flag){
        displayMenu();

        scanf("%d", &mode);

        switch(mode){
            case 0:                    
                register_face(&ref);
                calculate_histogram(ref, ref_histogram);
                waitKey(1000);
                printf("얼굴이 성공적으로 등록되었습니다.!\n");
                break;
            case 1:
                imshow("face",ref);
                waitKey(100);
                break;
            case 2:
                safe_verification_total_task(ref_histogram, &result_score);
                printf("score : %.2f\n", result_score);
                break;
            default:
                printf("프로그램 종료...\n");
                stop_flag = false;
        }
    }


    return 0;
}


void displayMenu(void) {
    printf("====================================\n");
    printf("      📸 얼굴 인증 시스템 메뉴      \n");
    printf("====================================\n");
    printf("  0 : 얼굴 등록\n");
    printf("  1 : 등록된 사진 확인\n");
    printf("  2 : 얼굴 인증 실행\n");
    printf("====================================\n");
    printf("원하는 모드를 선택하세요 (0-2): ");
}


float face_verification(Mat& Image, int * ref_histogram){

    cv::Mat current_shape;

    // 얼굴 랜드마크 모델 로드
    ldmarkmodel modelt;
    std::string modelFilePath = "roboman-landmark-model.bin";
    while (!load_ldmarkmodel(modelFilePath, modelt)) {
        std::cout << "Failed to load model. Enter model file path: ";
        std::cin >> modelFilePath;
    }

    Mat gray_img;
    cvtColor(Image, gray_img, COLOR_BGR2GRAY); // camera image color to gray

    modelt.track(Image, current_shape);
    int numLandmarks = current_shape.cols / 2;

    Mat lbp_img_result;
    make_lbp_img(gray_img, lbp_img_result);

    int lbp_histogram[68 * 256] = { 0 };

    for (int j = 0; j < numLandmarks; j++) {
        int x = static_cast<int>(current_shape.at<float>(j));
        int y = static_cast<int>(current_shape.at<float>(j + numLandmarks));

        int local_histogram[256] = { 0 };

        //4. crop lbp image by window(size 16x16)
        for (int wy = y - WINDOW_SIZE / 2; wy < y + WINDOW_SIZE / 2; wy++) {
            for (int wx = x - WINDOW_SIZE / 2; wx < x + WINDOW_SIZE / 2; wx++) {

                //5. local histogram by window(lbp image)
                local_histogram[lbp_img_result.at<uchar>(wy, wx)]++;
            }
        }

        for (int i = 0; i < 256; i++) {
            //6. copy to total histogram
            lbp_histogram[j * 256 + i] = local_histogram[i];
        }
    }

    //7. caculate score by cosine simmilarity
    float nomi = 0, refMag = 0, tarMag = 0;
    for (int i = 0; i < 256 * 68; i++) {
        nomi += ref_histogram[i] * lbp_histogram[i];
        refMag += ref_histogram[i] * ref_histogram[i];
        tarMag += lbp_histogram[i] * lbp_histogram[i];
    }
    float score = (nomi / (sqrt(refMag) * sqrt(tarMag)));

    // 8. set threshhold 0.78
    if (score > 0.78) {
        for (int j = 0; j < numLandmarks; j++) {
            int x = static_cast<int>(current_shape.at<float>(j));
            int y = static_cast<int>(current_shape.at<float>(j + numLandmarks));
            //create rectangles at positions of the windows.
            //the retangle is Green Color -> my face
            rectangle(Image, cv::Point(x - WINDOW_SIZE / 2, y - WINDOW_SIZE / 2), cv::Point(x + WINDOW_SIZE / 2, y + WINDOW_SIZE / 2), Scalar(0, 255, 0), 2, LINE_8);      
            //printf("score : %.2f\n", score);
        }
    }
    else {
        for (int j = 0; j < numLandmarks; j++) {
            int x = static_cast<int>(current_shape.at<float>(j));
            int y = static_cast<int>(current_shape.at<float>(j + numLandmarks));
            //create rectangles at positions of the windows.
            //the retangle is Red Color -> not my face
            rectangle(Image, cv::Point(x - WINDOW_SIZE / 2, y - WINDOW_SIZE / 2), cv::Point(x + WINDOW_SIZE / 2, y + WINDOW_SIZE / 2), Scalar(0, 0, 255), 2, LINE_8);
        }
    }

    return score;
}

// 수평/수직 성분 계산 함수
bool is_eye_opened(const float* magnitude_hog, const int* orientation_hog, int width, int height) {
    float horizontal_sum = 0.0, vertical_sum = 0.0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int bin = orientation_hog[y * width + x];
            float mag = magnitude_hog[y * width + x];

            // 수평 성분: 방향이 약 0도 또는 180도 근처
            if (bin == 0 || bin == 8 || bin == 7 || bin == 1) {
                horizontal_sum += mag*0.8;
            }
            // 수직 성분: 방향이 약 90도 근처
            else if (bin == 4 || bin == 5) {
                vertical_sum += mag;
            }
        }
    }

    // 수평/수직 성분 비교
    if (horizontal_sum > vertical_sum) {
        return false;  // 눈 감김
    } else {
        return true;  // 눈 뜸
    }
}

// 이미지 회전 함수
cv::Mat rotateImage(cv::Mat& src, double angle, cv::Point2f center) {
    cv::Mat rotMat = getRotationMatrix2D(center, angle, 1.0);
    cv::Mat rotatedImg;
    warpAffine(src, rotatedImg, rotMat, src.size(), INTER_LINEAR, BORDER_REFLECT);
    return rotatedImg;
}

// 대비 증가 함수
cv::Mat increaseContrast(const cv::Mat& src, double alpha, int beta) {
    cv::Mat dst;
    src.convertTo(dst, -1, alpha, beta);
    return dst;
}

// 두 점 사이의 각도 계산 함수
double calculateAngle(Point p1, Point p2) {
    double deltaY = p2.y - p1.y;
    double deltaX = p2.x - p1.x;
    return atan2(deltaY, deltaX);  // atan2는 두 점 사이의 기울기(각도)를 계산
}

void crop_eye_region(Mat& Image, Mat& output_image, Point left_point, Point right_point){

    // 왼쪽 눈 처리
    double EyeAngle = calculateAngle(left_point, right_point) * (180.0 / CV_PI);
    int EyeWidth = abs(right_point.x - left_point.x);
    int EyeHeight = abs(right_point.y - left_point.y);

    // 왼쪽 눈 처리
    Rect EyeROI(
        max(0, min(left_point.x, right_point.x) - 5),
        max(0, ((left_point.y + right_point.y) / 2) - 18),
        min(EyeWidth + 20, Image.cols - max(0, min(left_point.x, right_point.x) - 5)),
        min(36, Image.rows - max(0, ((left_point.y + right_point.y) / 2) - 18))
    );

    if (EyeROI.width <= 0 || EyeROI.height <= 0 ||
        EyeROI.x < 0 || EyeROI.y < 0 ||
        EyeROI.x + EyeROI.width > Image.cols ||
        EyeROI.y + EyeROI.height > Image.rows) {
        std::cerr << "Invalid leftEyeROI dimensions or position." << std::endl;
        return;
    }

    output_image = Image(EyeROI);

    if (output_image.empty()) {
        std::cerr << "eye region is empty." << std::endl;
        return;
    }

    Mat resizedEye, rotatedEye;

    cv::resize(output_image, resizedEye, Size(EYES_WINDOW_SIZE, EYES_WINDOW_SIZE));

    rotatedEye = rotateImage(resizedEye, EyeAngle, Point2f(resizedEye.cols / 2, resizedEye.rows / 2));

    output_image = increaseContrast(rotatedEye);
}

bool is_both_eye_opened(ldmarkmodel& modelt, Mat& Image){

    Mat current_shape; 

    Mat LeftEyeImage, RightEyeImage;
    Point leftEyeLeft, leftEyeRight, rightEyeLeft, rightEyeRight; 

    if (Image.empty()) {
        std::cerr << "Failed to capture frame." << std::endl;
        return false;
    }

    modelt.track(Image, current_shape);

    int numLandmarks = current_shape.cols / 2;
    for (int j = 0; j < numLandmarks; j++) {
        int x = current_shape.at<float>(j);
        int y = current_shape.at<float>(j + numLandmarks);

        if (j == 36) { // 왼쪽 눈 왼쪽 점
            leftEyeLeft = Point(x, y);
        }
        if (j == 39) { // 왼쪽 눈 오른쪽 점
            leftEyeRight = Point(x, y);
        }
        if (j == 42) { // 오른쪽 눈 왼쪽 점
            rightEyeLeft = Point(x, y);
        }
        if (j == 45) { // 오른쪽 눈 오른쪽 점
            rightEyeRight = Point(x, y);
        }
    }

    crop_eye_region(Image, LeftEyeImage, leftEyeLeft, leftEyeRight);
    crop_eye_region(Image, RightEyeImage, rightEyeLeft, rightEyeRight);
    
    float left_mag[EYES_WINDOW_SIZE*EYES_WINDOW_SIZE] = {0};
    int left_ori[EYES_WINDOW_SIZE*EYES_WINDOW_SIZE] = { 0 };

    float right_mag[EYES_WINDOW_SIZE*EYES_WINDOW_SIZE] = { 0};
    int right_ori[EYES_WINDOW_SIZE*EYES_WINDOW_SIZE] = { 0 };

    float left_hog[225] = { 0 };
    float right_hog[225] = { 0 };

    cvtColor(LeftEyeImage, LeftEyeImage, CV_BGR2GRAY);
    cvtColor(RightEyeImage, RightEyeImage, CV_BGR2GRAY);

    calculate_vector(&LeftEyeImage, left_mag, left_ori);
    calculate_vector(&RightEyeImage, right_mag, right_ori);

    bool rightEyeState = is_eye_opened(right_mag, right_ori, EYES_WINDOW_SIZE, EYES_WINDOW_SIZE);
    bool leftEyeState = is_eye_opened(left_mag, left_ori, EYES_WINDOW_SIZE, EYES_WINDOW_SIZE);

    // 결과 출력
    cout << "Right Eye: " << rightEyeState << endl;
    cout << "Left Eye: " << leftEyeState << endl;

    // 결과 보여주기
    cv::imshow("Left Eye (Contrast Increased)", LeftEyeImage);
    cv::imshow("Right Eye (Contrast Increased)", RightEyeImage);
    cv::imshow("Original Image", Image);

    return (rightEyeState && leftEyeState);    

}

void safe_verification_total_task(int * ref_histogram, float * result_score){

    *result_score = 0.0;

    // 얼굴 랜드마크 모델 로드
    ldmarkmodel modelt;
    std::string modelFilePath = "roboman-landmark-model.bin";
    while (!load_ldmarkmodel(modelFilePath, modelt)) {
        std::cout << "Failed to load model. Enter model file path: ";
        std::cin >> modelFilePath;
    }

    cv::VideoCapture mCamera(CAMERA_SOURCE);
    if (!mCamera.isOpened()) {
        std::cout << "Camera opening failed..." << std::endl;
        return;
    }

    cv::VideoCapture IRCamera(IR_CAMERA_SOURCE);
    int eye_open_count = 0;

    cv::Mat Image;
    Mat IRframe;
    cv::Mat current_shape;
    Point leftEyeLeft, leftEyeRight, rightEyeLeft, rightEyeRight;

    unsigned int frame_count=0;

    while (frame_count <= 30) {
        mCamera >> Image;
        IRCamera >> IRframe;

        if(is_both_eye_opened(modelt, Image)) {
            eye_open_count++;
        }
        else {
            eye_open_count = 0;
        }
    
        while(eye_open_count >= 10){

            if(!is_bright_frame(IRframe)) {
                waitKey(10);
                continue;
            }

            *result_score = face_verification(IRframe, ref_histogram);  
            waitKey(100);            
            cv::imshow("IR Image", IRframe);
	    waitKey(100);           

            if(*result_score > 0.78) {
                printf("인증되었습니다.\n");
                mCamera.release();
                IRCamera.release();
                cv::destroyAllWindows();
                return;
            }

            break;
        }

        frame_count++;

        printf("제한시간 : %u\n",frame_count);

        if (cv::waitKey(10) == 27) {
            break;  // 'ESC' 키를 눌러 종료
        }
    }
    printf("인증실패!!\n");
    mCamera.release();
    IRCamera.release();
    cv::destroyAllWindows();

    return;
}


void register_face(Mat* myface) {
    VideoCapture capture(IR_CAMERA_SOURCE);
    if (!capture.isOpened()) {
        std::cout << "Failed to open camera." << std::endl;
        return;
    }
        
    for(int i = 0; i < 25; i++){

        capture >> *myface;

        imshow("test",*myface);
        printf("적외선 카메라 안정화중...\n");
        waitKey(100);
    }

    printf("안정화 완료! 지금부터 얼굴을 등록합니다.\n");

    while(1){
        capture >> *myface;
        if(is_bright_frame(*myface) == true) {
            break;
        }
        waitKey(100);
    }
}

void calculate_histogram(Mat& Image, int* output_histogram) {

    //1. load model for finding landmark of face 
    ldmarkmodel modelt;
    std::string modelFilePath = "roboman-landmark-model.bin";
    while (!load_ldmarkmodel(modelFilePath, modelt)) {
        std::cout << "Failed to load model. Enter model file path: ";
        std::cin >> modelFilePath;
    }

    Mat gray_img;
    cvtColor(Image, gray_img, COLOR_BGR2GRAY); //input image color to gray
    Mat current_shape;
    modelt.track(Image, current_shape);

    int numLandmarks = current_shape.cols / 2;

    //2. make total lbp image for windowing
    Mat lbp_img_result;
    make_lbp_img(gray_img, lbp_img_result);

    for (int j = 0; j < numLandmarks; j++) {
        int x = static_cast<int>(current_shape.at<float>(j));
        int y = static_cast<int>(current_shape.at<float>(j + numLandmarks));

        int local_histogram[256] = { 0 };
        //3. make local histogram at positions of windows
        for (int wy = y - WINDOW_SIZE / 2; wy < y + WINDOW_SIZE / 2; wy++) {
            for (int wx = x - WINDOW_SIZE / 2; wx < x + WINDOW_SIZE / 2; wx++) {
                local_histogram[lbp_img_result.at<uchar>(wy, wx)]++;
            }
        }
        //4. copy to total histogram 
        for (int i = 0; i < 256; i++) {
            output_histogram[j * 256 + i] = local_histogram[i];
        }
    }
}

void make_lbp_img(Mat& ref, Mat& result) {
    result = Mat(ref.size(), CV_8UC1, Scalar(0));
    for (int y = 1; y < ref.rows - 1; y++) {
        for (int x = 1; x < ref.cols - 1; x++) {

            int magnitude[8] = 
            {
                ref.at<uchar>(y - 1, x), 
                ref.at<uchar>(y - 1, x + 1),
                ref.at<uchar>(y, x + 1), 
                ref.at<uchar>(y + 1, x + 1),
                ref.at<uchar>(y + 1, x), 
                ref.at<uchar>(y + 1, x - 1),
                ref.at<uchar>(y, x - 1), 
                ref.at<uchar>(y - 1, x - 1)
            };

            int current_magnitude = ref.at<uchar>(y, x);
            int LBP = 0;

            for (int i = 0; i < 8; i++) {
                LBP = (LBP << 1) | (current_magnitude <= magnitude[i]);
                //make lpb image by bit calculation
            }

            result.at<uchar>(y, x) = LBP;
        }
    }
}


// Frame 유효성 확인 (리사이즈 및 밝기 평균 적용)
bool is_bright_frame(const Mat& frame) {
    Mat resized_frame;
    resize(frame, resized_frame, Size(RESIZE_DIM, RESIZE_DIM));

    int width = resized_frame.cols;
    int height = resized_frame.rows;

    float sum = 0.0; 

    int max_sum = RESIZE_DIM * RESIZE_DIM;

    for(int y = 0; y < height; y++){
        for(int x = 0; x < width ; x++){
            sum += (resized_frame.at<uchar>(y,x)/255.0);
        }
    }

    //if((sum/max_sum)>0.5) return false;

    return (sum/max_sum) > BRIGHTNESS_THRESHOLD;  // 밝기 임계값 확인
}

// 이미지의 모든 픽셀에 대한 그레이디언트 벡터 계산
void calculate_vector(Mat* image, float* magnitude_hog, int* orientation_hog) {
    int height = image->rows, width = image->cols;

    int x, y, xx, yy;

    int mask_y[] =
    { -1, -1, -1,
       0,  0,  0,
       1,  1,  1 };

    int mask_x[] =
    { -1,  0,  1,
      -1,  0,  1,
      -1,  0,  1 };

    int fx = 0, fy = 0;

    float dir = 0.0;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {

            fx = fy = 0;

            for (yy = y - 1; yy <= y + 1; yy++) {
                for (xx = x - 1; xx <= x + 1; xx++) {

                    if (yy < 0 || yy >= height || xx < 0 || xx >= width) continue;

                    fy += image->at<uchar>(yy, xx) * mask_y[(yy - y + 1) * 3 + (xx - x + 1)];
                    fx += image->at<uchar>(yy, xx) * mask_x[(yy - y + 1) * 3 + (xx - x + 1)];
                }
            }

            dir = atan2(fy, fx) * (180 / CV_PI) + 90;

            if (dir > 180) dir -= 180;
            if (dir < 0) dir += 180;

            int bin = (int)(dir / 20);

            magnitude_hog[y * width + x] = sqrt(fx * fx + fy * fy);
            orientation_hog[y * width + x] = bin;
        }
    }
}

