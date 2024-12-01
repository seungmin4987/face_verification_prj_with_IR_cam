#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include "ldmarkmodel.h"

#define EYES_WINDOW_SIZE 36 //눈에 대한 윈도우 크기
#define WINDOW_SIZE 24 // 얼굴 랜드마크 에서의 윈도우 크기 
#define CAMERA_SOURCE 0 // 일반 웹켐 번호
#define IR_CAMERA_SOURCE 2 // IR카메라 번호
#define RESIZE_DIM 32 // 리사이즈 크기
#define BRIGHTNESS_THRESHOLD 0.06 
//32x32 윈도우의 평균밝기가 6% 이상이면, 유효한 IR 카메라 데이터임 

#define ANGLE_BIN 9 //orientation bin은 9개로 분류
#define ANGLE_BIN_NOM (180 / ANGLE_BIN)

using namespace std;
using namespace cv;

// 함수 선언
void displayMenu(void);
void register_face(Mat* myface); //얼굴 최초 등록

void safe_verification_total_task(int * ref_histogram, float * result_score);
bool is_bright_frame(const Mat& frame);
void make_lbp_img(Mat& ref, Mat& result);
float face_verification(Mat& Image, int * ref_histogram); //얼굴인증 태스크 관련 로직

void crop_eye_region(Mat& Image, Mat& output_image, Point left_point, Point right_point);
void increaseContrast(const Mat& src, Mat & dst,double alpha = 3.5);
double calculateAngle(Point p1, Point p2);
void rotateImage(Mat& src, double angle, Point center);
bool is_eye_opened(const float* magnitude_hog, const int* orientation_hog, int width, int height);
bool is_both_eye_opened(ldmarkmodel& modelt, Mat& Image); //두눈이 모두 떴는지 확인하는 로직

void calculate_vector(Mat* image, float* magnitude_hog, int* orientation_hog); //그래디언트 벡터 계산
void calculate_histogram(Mat& Image, int* output_histogram); //LBP 이미지를 이용한 히스토그램 계산

void append_histogram_to_csv(float *mag, int *ori, int window_size, const char *filename) ; //csv파일로 히스토그램 내보내기

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

void append_histogram_to_csv(float *mag, int *ori, int window_size, const char *filename) {
    FILE *file = fopen(filename, "w"); // "w" 모드로 파일 덮어 쓰기
    if (!file) {
        perror("Failed to open file");
        return;
    }

    // 파일이 비어 있는 경우 헤더를 작성
    fseek(file, 0, SEEK_END);
    if (ftell(file) == 0) { // 파일 크기가 0인지 확인
        fprintf(file, "Idx,Mag,Ori\n"); // CSV 헤더 작성
    }

    // 데이터를 CSV 형식으로 저장
    for (int y = 0; y < window_size; y++) {
        for (int x = 0; x < window_size; x++) {
            int idx = y * window_size + x;
            fprintf(file, "%d,%.3f,%d\n", idx, mag[idx], ori[idx]);
        }
    }

    fclose(file);
}



float face_verification(Mat& Image, int * ref_histogram){

    cv::Mat current_shape;

    // 얼굴 랜드마크 모델 로드
    ldmarkmodel modelt;
    std::string modelFilePath = "roboman-landmark-model.bin";
    while (!load_ldmarkmodel(modelFilePath, modelt)) {
        std::cout << "모델 로드 실패. 모델 파일 경로를 입력하세요: ";
        std::cin >> modelFilePath;
    }

    Mat gray_img;
    cvtColor(Image, gray_img, COLOR_BGR2GRAY); // 카메라 이미지를 컬러에서 그레이스케일로 변환

    modelt.track(Image, current_shape);
    int numLandmarks = current_shape.cols / 2; // 랜드마크 개수 계산

    Mat lbp_img_result;
    make_lbp_img(gray_img, lbp_img_result); // LBP 이미지 생성

    int lbp_histogram[68 * 256] = { 0 }; // LBP 히스토그램 초기화

    for (int j = 0; j < numLandmarks; j++) {
        int x = static_cast<int>(current_shape.at<float>(j)); // 현재 랜드마크의 x 좌표
        int y = static_cast<int>(current_shape.at<float>(j + numLandmarks)); // 현재 랜드마크의 y 좌표

        int local_histogram[256] = { 0 }; // 윈도우 내 지역 히스토그램 초기화

        // LBP 이미지에서 윈도우(크기 16x16)로 영역 자르기
        for (int wy = y - WINDOW_SIZE / 2; wy < y + WINDOW_SIZE / 2; wy++) {
            for (int wx = x - WINDOW_SIZE / 2; wx < x + WINDOW_SIZE / 2; wx++) {

                // 윈도우 내 지역 히스토그램 생성
                local_histogram[lbp_img_result.at<uchar>(wy, wx)]++;
            }
        }

        for (int i = 0; i < 256; i++) {
            // 전체 히스토그램에 복사
            lbp_histogram[j * 256 + i] = local_histogram[i];
        }
    }

    // 코사인 유사도를 계산하여 점수 산출
    float nomi = 0, refMag = 0, tarMag = 0;
    for (int i = 0; i < 256 * 68; i++) {
        nomi += ref_histogram[i] * lbp_histogram[i];
        refMag += ref_histogram[i] * ref_histogram[i];
        tarMag += lbp_histogram[i] * lbp_histogram[i];
    }
    float score = (nomi / (sqrt(refMag) * sqrt(tarMag)));

    // 임계값 0.78로 판별
    if (score > 0.78) {
        for (int j = 0; j < numLandmarks; j++) {
            int x = static_cast<int>(current_shape.at<float>(j));
            int y = static_cast<int>(current_shape.at<float>(j + numLandmarks));
            // 윈도우 위치에 초록색 사각형 생성 -> 나의 얼굴로 판별
            rectangle(Image, cv::Point(x - WINDOW_SIZE / 2, y - WINDOW_SIZE / 2), cv::Point(x + WINDOW_SIZE / 2, y + WINDOW_SIZE / 2), Scalar(0, 255, 0), 2, LINE_8);      
        }
    }
    else {
        for (int j = 0; j < numLandmarks; j++) {
            int x = static_cast<int>(current_shape.at<float>(j));
            int y = static_cast<int>(current_shape.at<float>(j + numLandmarks));
            // 윈도우 위치에 빨간색 사각형 생성 -> 내 얼굴이 아님
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
            int bin = orientation_hog[y*width+x];
            float mag = magnitude_hog[y*width+x];

                //0 : 0~20
                //1 : 20~40
                //2 : 40~60
                //3 : 60~80
                //4 : 80~100
                //5 : 100~120
                //6 : 120~140
                //7 : 140~160
                //8 : 160~180

            // 수평 성분: 방향이 약 0도 또는 180도 근처
            if (bin == 0 || bin == 1 || bin == 8) {
                horizontal_sum += mag;
            }
            // 수직 성분: 방향이 약 90도 근처
            else if (bin==3 || bin == 4 || bin == 5) {
                vertical_sum += mag * 0.78;
            }
        }
    }

    // 수평/수직 성분 비교
    if (horizontal_sum > vertical_sum) {
        return false;  // 눈 감김
    } else {
        return true;  // 눈 뜸
    }

    return true;
}

// bool is_eye_opened(const float* magnitude_hog, const int* orientation_hog, int width, int height) {
//     float horizontal_sum = 0.0, vertical_sum = 0.0;
//     float mag_sum = 0.0;

//     // 전체 크기 계산 (정규화를 위한 합산)
//     for (int y = 0; y < height; y++) {
//         for (int x = 0; x < width; x++) {
//             mag_sum += magnitude_hog[y * width + x];
//         }
//     }

//     if (mag_sum == 0.0f) {
//         // Magnitude가 0이면 눈 감김으로 간주 (무효 상태 처리)
//         return false;
//     }

//     // 가중치 설정
//     const float horizontal_weight = 0.7f;  // 수평 성분 가중치
//     const float vertical_weight = 1.0f;  // 수직 성분 가중치

//     // 수평 및 수직 성분 계산
//     for (int y = 0; y < height; y++) {
//         for (int x = 0; x < width; x++) {
//             int bin = orientation_hog[y * width + x];
//             float mag = magnitude_hog[y * width + x] / mag_sum;  // 정규화된 Magnitude

//             // 수평 성분: 방향이 약 0도 또는 180도 근처
//             if (bin == 0 || bin == 1 || bin == 8) {
//                 horizontal_sum += mag * horizontal_weight;
//             }
//             // 수직 성분: 방향이 약 90도 근처
//             else if (bin == 4 ) {
//                 vertical_sum += mag * vertical_weight;
//             }
//         }
//     }

//     // 수평/수직 성분 비교
//     if (horizontal_sum > vertical_sum) {
//         return false;  // 눈 감김
//     } else {
//         return true;  // 눈 뜸
//     }
// } // 정규화 방식인데, 생각보다 성능이 좋지 않아 제외.


void rotateImage(Mat& src, Mat& dst, double angle, Point center) {
    int width = src.cols;
    int height = src.rows;

    // 라디안으로 변환
    double radians = angle * CV_PI / 180.0;

    // 회전 행렬 요소 계산
    double cos_theta = std::cos(radians);
    double sin_theta = std::sin(radians);

    // 출력 이미지 초기화
    dst = cv::Mat::zeros(height, width, src.type());

    // 역방향 매핑을 사용한 회전
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 회전 후 좌표를 원본 이미지 좌표로 매핑
            double srcX = cos_theta * (x - center.x) + sin_theta * (y - center.y) + center.x;
            double srcY = -sin_theta * (x - center.x) + cos_theta * (y - center.y) + center.y;

            // 원본 좌표가 유효한 범위 내에 있는지 확인
            if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                // 근접 이웃 보간 사용
                int ix = (int)srcX;
                int iy = (int)srcY;

                // 출력 픽셀 값 설정
                dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(iy, ix);
            }
            else{
                dst.at<cv::Vec3b>(y, x) = Vec3b(36, 36, 36);
            }
        }
    }
}

void increaseContrast(const Mat& src, Mat& dst, double alpha) {
    // 입력 이미지 크기와 채널 확인
    int width = src.cols;
    int height = src.rows;

    // 입력 이미지를 그레이스케일로 변환
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 출력 이미지 초기화
    dst = cv::Mat::zeros(height, width, CV_8UC1);

    // 대비 조정: new_pixel = alpha * old_pixel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int mag = gray.at<uchar>(y, x);

            // 대비 조정
            int new_mag = (int)(alpha * mag);

            // saturation (0~255)
            new_mag = (new_mag > 255) ? 255 : (new_mag < 0 ? 0 : new_mag);

            // 결과 저장
            dst.at<uchar>(y, x) = new_mag;
        }
    }
}

// 두 점 사이의 각도 계산 함수
double calculateAngle(Point p1, Point p2) {
    double deltaY = p2.y - p1.y;
    double deltaX = p2.x - p1.x;
    return atan2(deltaY, deltaX);  // 두 점 사이의 기울기(각도)를 계산
}

void crop_eye_region(Mat& Image, Mat& output_image, Point left_point, Point right_point){

    // 왼쪽 점과 오른쪽 점 좌표를 이용해 기울기를 계산
    double EyeAngle = calculateAngle(left_point, right_point) * (180.0 / CV_PI); //라디안-> degree
    int EyeWidth = abs(right_point.x - left_point.x); // 두 점 사이 가로 거리
    int EyeHeight = abs(right_point.y - left_point.y); // 두 점 사이 세로 거리

    // 점들에 기반한 ROI 설정
    Rect EyeROI(
        max(0, min(left_point.x, right_point.x) - 5), // 시작 x 좌표 (여유 공간 5 픽셀 추가)
        max(0, ((left_point.y + right_point.y) / 2) - 18), // 시작 y 좌표 (중간 지점에서 위로 18 픽셀 이동)
        min(EyeWidth + 20, Image.cols - max(0, min(left_point.x, right_point.x) - 5)), // 너비 (20 픽셀 추가)
        min(36, Image.rows - max(0, ((left_point.y + right_point.y) / 2) - 18)) // 높이 (최대 36 픽셀 제한)
    );

    // ROI가 유효한지 확인
    if (EyeROI.width <= 0 || EyeROI.height <= 0 || 
        EyeROI.x < 0 || EyeROI.y < 0 || 
        EyeROI.x + EyeROI.width > Image.cols || 
        EyeROI.y + EyeROI.height > Image.rows) { 
        printf("ROI 설정 오류\n");
        return;
    }

    // ****ROI로 이미지를 자름*****
    output_image = Image(EyeROI);

    // 결과 이미지가 비어 있는 경우 처리
    if (output_image.empty()) {
        printf("오류!!\n");
        return;
    }

    Mat resizedEye, rotatedEye;

    // 잘라낸 이미지를 편의를 위해 고정된 크기로 조정
    cv::resize(output_image, resizedEye, Size(EYES_WINDOW_SIZE, EYES_WINDOW_SIZE));

    // 이미지를 회전해서, 눈이 항상 수평을 유지하게 끔 설정
    rotateImage(resizedEye, rotatedEye, -EyeAngle, Point(resizedEye.cols / 2, resizedEye.rows / 2));

    // 밝은 곳은 밝게, 어두운 곳은 어둡게 유지하기 -> 빈익빈 부익부
    increaseContrast(rotatedEye, output_image);
}


bool is_both_eye_opened(ldmarkmodel& modelt, Mat& Image){

    Mat current_shape; // 현재 얼굴 랜드마크 좌표

    Mat LeftEyeImage, RightEyeImage; // 왼쪽 눈과 오른쪽 눈 이미지
    Point leftEyeLeft, leftEyeRight, rightEyeLeft, rightEyeRight; // 눈의 좌표 (왼쪽 점, 오른쪽 점)

    float left_mag[EYES_WINDOW_SIZE*EYES_WINDOW_SIZE] = {0}; // 왼쪽 크기 
    int left_ori[EYES_WINDOW_SIZE*EYES_WINDOW_SIZE] = { 0 }; // 왼쪽 방향 

    float right_mag[EYES_WINDOW_SIZE*EYES_WINDOW_SIZE] = { 0 }; // 오른쪽 크기
    int right_ori[EYES_WINDOW_SIZE*EYES_WINDOW_SIZE] = { 0 }; // 오른쪽 방향

    // 얼굴 랜드마크 추적
    modelt.track(Image, current_shape);

    // 랜드마크 개수 계산
    int numLandmarks = current_shape.cols / 2;

    // 랜드마크 좌표에서 눈의 왼쪽/오른쪽 점 추출
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

    // 각각 눈 영역 잘라내기
    crop_eye_region(Image, LeftEyeImage, leftEyeLeft, leftEyeRight);
    crop_eye_region(Image, RightEyeImage, rightEyeLeft, rightEyeRight);

    // 눈 벡터 계산
    calculate_vector(&LeftEyeImage, left_mag, left_ori);
    calculate_vector(&RightEyeImage, right_mag, right_ori);

    //append_histogram_to_csv(left_mag,left_ori,EYES_WINDOW_SIZE,"histogram_right.csv");
    //append_histogram_to_csv(right_mag,right_ori,EYES_WINDOW_SIZE,"histogram_left.csv");
    //히스토그램 파일이 필요하면 주석해제

    // 각각의 눈 열림 상태 확인
    bool rightEyeState = is_eye_opened(right_mag, right_ori, EYES_WINDOW_SIZE, EYES_WINDOW_SIZE);
    bool leftEyeState = is_eye_opened(left_mag, left_ori, EYES_WINDOW_SIZE, EYES_WINDOW_SIZE);

    // 결과 출력
    printf("Right Eye: %d\n", rightEyeState);
    printf("Left Eye: %d\n", leftEyeState);

    // 결과 이미지 표시
    cv::imshow("Left Eye (Contrast Increased)", LeftEyeImage); // 왼쪽 눈 이미지 출력
    cv::imshow("Right Eye (Contrast Increased)", RightEyeImage); // 오른쪽 눈 이미지 출력
    cv::imshow("Original Image", Image); // 원본 이미지 출력

    // 두 눈이 모두 열려 있는지 여부 반환
    return (rightEyeState && leftEyeState);    
}


void safe_verification_total_task(int * ref_histogram, float * result_score){

    *result_score = 0.0; // 결과 점수 초기화

    // 얼굴 랜드마크 모델 로드
    ldmarkmodel modelt;
    std::string modelFilePath = "roboman-landmark-model.bin";
    while (!load_ldmarkmodel(modelFilePath, modelt)) { // 모델 로드 실패 시 경로 재입력 요청
        std::cout << "모델 로드 실패. 모델 파일 경로를 입력하세요: ";
        std::cin >> modelFilePath;
    }

    // 카메라 열기
    cv::VideoCapture mCamera(CAMERA_SOURCE); // 메인 카메라
    if (!mCamera.isOpened()) { // 카메라 열기에 실패한 경우
        std::cout << "카메라 열기 실패..." << std::endl;
        return;
    }

    cv::VideoCapture IRCamera(IR_CAMERA_SOURCE); // IR 카메라
    if (!IRCamera.isOpened()) { // 카메라 열기에 실패한 경우
        std::cout << "카메라 열기 실패..." << std::endl;
        return;
    }

    int eye_open_count = 0; // 눈이 열린 상태를 유지한 횟수

    cv::Mat Image; // 메인 카메라 프레임
    Mat IRframe; // IR 카메라 프레임

    cv::Mat current_shape;
    Point leftEyeLeft, leftEyeRight, rightEyeLeft, rightEyeRight;
    //눈의 특징점

    unsigned int frame_count = 0; // 프레임 카운트

    while (frame_count <= 30) { // 최대 30 프레임 동안 반복
        mCamera >> Image; // 메인 카메라 프레임
        IRCamera >> IRframe; // IR 카메라 프레임

        if (is_both_eye_opened(modelt, Image)) { // 양쪽 눈이 열려 있는지 확인
            eye_open_count++; // 눈 열림 카운트 증가
        }
        else {
            eye_open_count = 0; // 카운트 초기화
        }

        // 눈 열림 상태가 8 프레임 연속 유지되면 인증 시도
        while (eye_open_count >= 8) {

            if (!is_bright_frame(IRframe)) { // IR 프레임 밝기 확인
                waitKey(10);
                continue; // 유효한 IR카메라 데이터가 나올때 까지 반복
            }

            *result_score = face_verification(IRframe, ref_histogram); // 얼굴 인증 수행
            waitKey(100);            
            cv::imshow("IR Image", IRframe); 
            waitKey(100);

            if (*result_score > 0.78) { // 인증 성공 여부 확인
                printf("인증되었습니다.\n");
                mCamera.release(); // 카메라 자원 해제
                IRCamera.release(); // IR 카메라 자원 해제
                cv::destroyAllWindows(); // 모든 창 닫기
                return;
            }

            break;
        }

        frame_count++; // 프레임 카운트 증가

        printf("제한시간 : %u\n", frame_count); // 제한 시간 출력

        if (cv::waitKey(15) == 27) { // 'ESC' 키를 눌러 종료
            break;
        }
    }

    // 인증 실패 메시지 출력
    printf("인증실패!!\n");
    mCamera.release(); // 카메라 자원 해제
    IRCamera.release(); // IR 카메라 자원 해제
    cv::destroyAllWindows(); // 모든 창 닫기

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

    // 1. 얼굴 랜드마크를 찾기 위한 모델 로드
    ldmarkmodel modelt;
    std::string modelFilePath = "roboman-landmark-model.bin";
    while (!load_ldmarkmodel(modelFilePath, modelt)) { // 모델 로드 실패 시 경로 재입력 요청
        std::cout << "모델 로드 실패. 모델 파일 경로를 입력하세요: ";
        std::cin >> modelFilePath;
    }

    Mat gray_img;
    cvtColor(Image, gray_img, COLOR_BGR2GRAY); // 입력 이미지를 그레이스케일로 변환
    Mat current_shape;
    modelt.track(Image, current_shape); // 얼굴 랜드마크 추적

    int numLandmarks = current_shape.cols / 2; // 랜드마크 개수 계산

    // 2. 윈도우 처리를 위한 전체 LBP 이미지 생성
    Mat lbp_img_result;
    make_lbp_img(gray_img, lbp_img_result); // LBP 이미지 생성

    // 랜드마크 위치를 기반으로 히스토그램 계산
    for (int j = 0; j < numLandmarks; j++) {
        int x = static_cast<int>(current_shape.at<float>(j)); // 현재 랜드마크의 x 좌표
        int y = static_cast<int>(current_shape.at<float>(j + numLandmarks)); // 현재 랜드마크의 y 좌표

        int local_histogram[256] = { 0 }; // 윈도우 내 지역 히스토그램 초기화

        // 3. 윈도우 위치에서 로컬 히스토그램 생성
        for (int wy = y - WINDOW_SIZE / 2; wy < y + WINDOW_SIZE / 2; wy++) {
            for (int wx = x - WINDOW_SIZE / 2; wx < x + WINDOW_SIZE / 2; wx++) {
                local_histogram[lbp_img_result.at<uchar>(wy, wx)]++; // 픽셀 값(~255)에 따른 히스토그램 증가
            }
        }

        // 4. 로컬 히스토그램을 전체 히스토그램에 복사
        for (int i = 0; i < 256; i++) {
            output_histogram[j * 256 + i] = local_histogram[i]; // 각 랜드마크에 대한 히스토그램 저장
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

            if (dir >= 180) dir -= 180;
            if (dir < 0) dir += 180;

            int bin = (int)(dir / ANGLE_BIN_NOM);

            magnitude_hog[y * width + x] = sqrt(fx * fx + fy * fy);
            orientation_hog[y * width + x] = bin;
        }
    }
}

