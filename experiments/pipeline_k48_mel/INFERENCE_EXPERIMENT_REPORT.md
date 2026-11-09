# Báo cáo thử nghiệm suy luận W standard Mel K=48

## Phạm vi

Các số liệu trong báo cáo này được tạo bằng Python trên máy tính. Bộ giải được
gọi là **MCU-style FISTA** là bản Python float32 mô phỏng các phép tính FISTA
trong firmware (tính trước Gram matrix, bước chiếu không âm và cận Lipschitz
theo tổng hàng). Không có kết quả nào dưới đây được đo trên ESP32 thật hoặc
trình giả lập MCU. Vì vậy báo cáo không kết luận về thời gian xử lý, RAM hay
độ chính xác thực tế trên board.

## Dữ liệu và pipeline

- Giữ 48 thành phần: 24 thành phần sự kiện, 12 nhiễu từ tô, 12 nhiễu môi trường.
- Giữ nguyên W cũ kích thước 1025×48; train W riêng cho từng cấu hình Mel.
- Âm thanh mono 44,1 kHz; Hann tuần hoàn dài 2048 mẫu; hop 1024; STFT không
  padding ở biên.
- Lấy phổ biên độ, chiếu lên Mel Slaney từ 20 Hz đến Nyquist, rồi nén
  `log1p(mel / reference)`. `reference` được ước lượng trên tập train và lưu
  cùng model.
- W Mel được train riêng cho ba nhóm thành phần 24/12/12; tối đa 2048 frame
  từ mỗi file train được lấy để giới hạn thời gian huấn luyện. Các cột W được
  chuẩn hóa L2 sau khi ghép.
- 14 file train cố định dùng để hiệu chỉnh ngưỡng đỉnh; 4 file train khác là
  validation giữ riêng; test gồm các WAV có nhãn trong `data/test` (4 file ở
  lần chạy tạo bảng dưới).

Độ chính xác đếm theo file được tính bằng
`max(0, 1 - abs(predicted - true) / true) * 100%`, rồi lấy trung bình các file
trong từng split. Prominence và khoảng cách đỉnh được tune trên calibration;
test không tham gia chọn số vòng lặp hay ngưỡng.

## Các dạng Mel đã thử

| W / frontend | Python SciPy NNLS validation | Python SciPy NNLS test | FISTA kết quả đã lưu validation / test |
|---|---:|---:|---:|
| 32 Mel, power | 92.10% | 87.52% | 2048 vòng: 90.79% / 88.71% |
| 64 Mel, power | 93.99% | 89.49% | 512 vòng: 95.12% / 88.89% |
| 64 Mel, magnitude + log1p | **98.11%** | **94.91%** | Xem sweep bên dưới |
| 128 Mel, magnitude + log1p | 93.81% | 77.80% | 2048 vòng: 93.81% / 78.63% |

Trong các dạng đã thử, cấu hình 64 Mel magnitude + log1p là ứng viên tốt nhất
để chạy sweep FISTA chi tiết. So sánh này train W riêng theo từng frontend;
nó không phải phép thử chỉ thay số hàng trên cùng một W.

## Sweep FISTA trên W 64 Mel magnitude + log1p

Python SciPy `scipy.optimize.nnls` là reference. Mỗi hàng FISTA dùng bộ giải
float32 MCU-style được chạy trong Python trên cùng features và W.

| Bộ giải | Validation | Test |
|---|---:|---:|
| SciPy NNLS | 98.11% | 94.91% |
| FISTA 8 vòng | 95.09% | 96.84% |
| FISTA 16 vòng | 94.16% | 96.36% |
| FISTA 32 vòng | 94.16% | 96.36% |
| FISTA 64 vòng | 91.54% | 87.43% |
| FISTA 128 vòng | 92.67% | 73.81% |
| FISTA 256 vòng | 93.81% | 73.19% |
| **FISTA 512 vòng** | **98.11%** | **97.23%** |
| FISTA 1024 vòng | 98.11% | 96.16% |
| FISTA 2048 vòng | 98.11% | 95.32% |
| FISTA 4096 vòng | 98.11% | 94.91% |

Độ chính xác không tăng đơn điệu theo số vòng. FISTA 512 là số vòng nhỏ nhất
trong sweep đạt độ chính xác calibration và validation ít nhất bằng SciPy NNLS;
đây là tiêu chí chọn cấu hình. Kết quả test được xem sau khi chọn, không dùng
để chọn. So với pipeline Python cũ (96.60% test theo báo cáo cũ trong
`iteration_comparison.json`), FISTA 512 đạt 97.23% trên lần đánh giá này.

## Phân biệt các mức kiểm chứng

1. **Đã chạy trên Python:** huấn luyện W Mel, SciPy NNLS reference, sweep
   FISTA float32, tune peak settings trên calibration và tính accuracy trên
   validation/test. Các CSV/JSON gốc ở `results/` và từng thư mục `runs/`.
2. **Chưa chạy trên MCU ảo:** MCU-style ở đây chỉ là mô phỏng các phép tính
   solver bằng Python, không phải giả lập ESP32, chu kỳ CPU hay bộ nhớ.
3. **Chưa chạy trên ESP32:** sketch Arduino Experiment Viewer đã được thêm để
   người dùng nạp sau này. Chưa có phép đo board và chưa xác nhận build bằng
   Arduino IDE trong lần đánh giá này. Kết quả firmware thật vẫn cần được thu
   bằng WAV giống nhau và đối chiếu với Python.

Sketch viewer hiện dùng coordinate descent cho các profile NNLS chạy trên MCU;
đây là xấp xỉ solver NNLS cũ, không phải SciPy exact NNLS. Python SciPy NNLS
được vẽ như reference riêng khi chạy host logger. Không nên coi các kết quả
Python hiện tại là bằng chứng firmware ESP32 đã đạt độ chính xác tương đương.
