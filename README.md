# SRCNN-PyTorch

## Giới thiệu

Repository này trình bày quá trình tìm hiểu, kế thừa và thực nghiệm mô hình **SRCNN** trong bài toán **siêu độ phân giải ảnh đơn** (*Single Image Super-Resolution – SISR*). Đây là một phần thực nghiệm phục vụ báo cáo học phần **Ứng dụng học máy trong thiết kế**.

Bài toán siêu độ phân giải ảnh có mục tiêu khôi phục hoặc tái tạo ảnh có độ phân giải cao từ ảnh đầu vào có độ phân giải thấp. Trong bối cảnh thiết kế, kỹ thuật này có thể hỗ trợ nâng cấp ảnh chất lượng thấp, cải thiện tài nguyên hình ảnh cũ, ảnh sản phẩm, texture hoặc các tư liệu đồ họa trước khi đưa vào quá trình thiết kế.

## Convolutional Neural Network (CNN)

**Convolutional Neural Network (CNN)** là một loại mạng nơ-ron nhân tạo thường được sử dụng trong các bài toán xử lý ảnh và thị giác máy tính. Khác với mạng nơ-ron truyền thống, CNN sử dụng các lớp tích chập để học đặc trưng cục bộ của ảnh như cạnh, đường nét, hoa văn, kết cấu và các vùng có tần số cao.

Trong bài toán siêu độ phân giải ảnh, CNN có khả năng học mối quan hệ giữa ảnh độ phân giải thấp và ảnh độ phân giải cao từ dữ liệu huấn luyện. Nhờ đó, mô hình có thể tái tạo ảnh đầu ra sắc nét hơn so với các phương pháp nội suy truyền thống như Bicubic...

## Bài báo gốc về SRCNN

Mô hình **Super-Resolution Convolutional Neural Network (SRCNN)** được đề xuất bởi Chao Dong, Chen Change Loy, Kaiming He và Xiaoou Tang trong công trình:

> Dong, C., Loy, C. C., He, K., & Tang, X. (2016). *Image super-resolution using deep convolutional networks*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 38(2), 295–307.

Bản đầu tiên của công trình được công bố trên arXiv vào năm 2014, sau đó phiên bản hoàn chỉnh được xuất bản trên IEEE TPAMI vào năm 2016.

SRCNN là một trong những mô hình đầu tiên áp dụng học sâu vào bài toán siêu độ phân giải ảnh theo hướng học end-to-end. Kiến trúc cơ bản của SRCNN gồm ba tầng chính:

1. **Patch extraction and representation**: trích xuất và biểu diễn đặc trưng từ ảnh đầu vào.
2. **Non-linear mapping**: ánh xạ phi tuyến từ đặc trưng của ảnh độ phân giải thấp sang biểu diễn gần với ảnh độ phân giải cao.
3. **Reconstruction**: tái tạo ảnh đầu ra có chất lượng cao hơn.

## Nguồn mã kế thừa

Mã nguồn trong repository này được **kế thừa và chỉnh sửa** từ repository công khai của **Lornatang**:

> Lornatang. (n.d.). *SRCNN-PyTorch*. GitHub.
> https://github.com/Lornatang/SRCNN-PyTorch

Repository này không tuyên bố xây dựng toàn bộ mã nguồn từ đầu. Mã nguồn gốc được sử dụng làm cơ sở để phục vụ mục đích học tập, tìm hiểu kiến trúc SRCNN, cấu hình tham số, chuẩn bị dữ liệu, huấn luyện, kiểm thử và đánh giá kết quả thực nghiệm.

Các chỉnh sửa chủ yếu bao gồm:

* Điều chỉnh cấu hình thực nghiệm trong `config.py`.
* Chạy huấn luyện và kiểm thử trên tập T91 và Set5.
* Ghi nhận kết quả PSNR/SSIM.
* Lưu ảnh đầu ra sau khi xử lý bằng mô hình SRCNN.
* Sử dụng kết quả thực nghiệm để phục vụ phân tích trong báo cáo học phần.

## Dữ liệu sử dụng

Trong thực nghiệm cá nhân, repository sử dụng:

* **T91 Image Dataset**: dùng làm tập huấn luyện.
* **Set5 SuperResolution**: dùng làm tập kiểm thử, trong đó sử dụng thư mục `GTmod12`.

Nguồn dữ liệu được tải từ Kaggle:

* T91 Image Dataset: https://www.kaggle.com/datasets/ll01dm/t91-image-dataset
* Set5 SuperResolution: https://www.kaggle.com/datasets/bijaygurung/set5-superresolution

---

## Cấu trúc dự án

Cấu trúc chính của repository:

```text
SRCNN_24022983_TaBaoHan/
│
├── README.md
├── .gitignore
├── .gitattributes
│
└── SRCNN/
    ├── config.py
    ├── dataset.py
    ├── image_quality_assessment.py
    ├── imgproc.py
    ├── inference.py
    ├── model.py
    ├── train.py
    ├── test.py
    ├── requirements.txt
    ├── LICENSE
    └── scripts/
```

Ý nghĩa một số file chính:

| File / thư mục                | Chức năng                                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| `config.py`                   | Thiết lập đường dẫn dữ liệu, hệ số phóng đại, chế độ train/test, số epoch, learning rate và checkpoint |
| `model.py`                    | Định nghĩa kiến trúc mạng SRCNN                                                                        |
| `dataset.py`                  | Xử lý dữ liệu huấn luyện và kiểm thử                                                                   |
| `imgproc.py`                  | Các hàm xử lý ảnh: resize, chuyển đổi màu, chuyển đổi tensor                                           |
| `image_quality_assessment.py` | Tính các chỉ số đánh giá như PSNR và SSIM                                                              |
| `train.py`                    | Huấn luyện mô hình SRCNN                                                                               |
| `test.py`                     | Kiểm thử mô hình trên tập Set5                                                                         |
| `inference.py`                | Chạy mô hình trên một ảnh đầu vào riêng                                                                |
| `requirements.txt`            | Danh sách thư viện cần cài đặt                                                                         |
| `scripts/`                    | Thư mục chứa script hỗ trợ nếu có                                                                      |

## Cài đặt thư viện

Cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install -r requirements.txt
```

## Huấn luyện mô hình

Trước khi huấn luyện, cần kiểm tra và chỉnh các đường dẫn dữ liệu trong file `config.py`.

Sau đó chạy:

```bash
python train.py
```

## Kiểm thử mô hình

Để kiểm thử mô hình trên tập Set5, chỉnh chế độ trong `config.py` sang `"test"` và chạy:

```bash
python test.py
```

Kết quả ảnh đầu ra sẽ được lưu trong thư mục:

```text
results/test/
```

## Ghi chú

Repository này được thực hiện với mục đích học tập và tái hiện mô hình SRCNN ở mức cơ bản. Các kết quả thực nghiệm có thể khác với bài báo gốc do khác biệt về dữ liệu huấn luyện, checkpoint, cách tiền xử lý ảnh, cách tính PSNR/SSIM, môi trường chạy và phiên bản thư viện.

## Tài liệu tham khảo

[1] Dong, C., Loy, C. C., He, K., & Tang, X. (2016). *Image super-resolution using deep convolutional networks*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 38(2), 295–307.

[2] Lornatang. (n.d.). *SRCNN-PyTorch*. GitHub. https://github.com/Lornatang/SRCNN-PyTorch

[3] Gurung, B. (n.d.). *Set5 SuperResolution*. Kaggle. https://www.kaggle.com/datasets/bijaygurung/set5-superresolution

[4] ll01dm. (n.d.). *T91 Image Dataset*. Kaggle. https://www.kaggle.com/datasets/ll01dm/t91-image-dataset
