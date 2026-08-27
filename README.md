# Cảm Biến Ảo Âm Thanh Ước Lượng Năng Suất Phễu Rung Cấp Phôi (NMF + CRNN)
**Acoustic Virtual Sensor for Monitoring & Counting Workpieces from Vibratory Bowl Feeder**

---

## 1. Giới thiệu Đề tài
Dự án phát triển hệ thống **Cảm biến ảo âm thanh (Acoustic Virtual Sensor)** phục vụ giám sát và ước lượng năng suất cấp phôi (*Feed Rate*) của Phễu rung công nghiệp (*Vibratory Bowl Feeder - VBF*) không tiếp xúc qua Microphone.

### Kiến trúc Kỹ thuật:
$$\text{Audio} \longrightarrow \text{STFT} \longrightarrow \text{Log Spectrum} \longrightarrow \text{Fixed NMF } (W_{\text{standard}}) \longrightarrow H_{\text{event\_sum}} \longrightarrow \text{Hybrid CRNN} \longrightarrow \text{Event Grouping} \longrightarrow \text{Part Count \& Feed Rate}$$

---

## 2. Cấu trúc Thư mục Dự án

```text
AcousticVirtualSensor/
├── config.py                 # File cấu hình trung tâm (DSP, NMF, CRNN, Paths)
├── main.py                   # Điểm khởi chạy chương trình (CLI / GUI)
├── requirements.txt          # Danh sách thư viện phụ thuộc
├── README.md                 # Tài liệu hướng dẫn sử dụng
│
├── data/
│   ├── train/                # Chứa file audio *.wav huấn luyện
│   ├── test/                 # Chứa file audio *.wav kiểm thử độc lập
│   ├── annotations/          # File ground-truth nhãn *.npz (train/test)
│   └── datasets/             # File dataset đóng gói crnn_train.npz
│
├── models/
│   ├── nmf/W_standard.npy    # Ma trận từ điển NMF cố định (1025 x 48)
│   └── crnn/crnn_model.pt    # File trọng số mạng Hybrid CRNN
│
├── src/
│   ├── audio/                # Ingestion & Framing geometry
│   ├── preprocessing/        # STFT & Log Compression
│   ├── nmf/                  # Dictionary Loader, NNLS Solver, Activation
│   ├── crnn/                 # Model PyTorch, Dataset, Trainer, Predictor
│   ├── annotation/           # Interactive GUI gán nhãn, Label Smoothing
│   ├── event/                # Peak baseline, Binary-to-Event Grouping, Counter
│   ├── runtime/              # AcousticVirtualSensor Core Engine, Streaming
│   └── evaluation/           # Frame, Event, Count Benchmark Metrics
│
├── scripts/
│   ├── train_nmf.py          # Huấn luyện tái tạo W_standard
│   ├── annotate_train.py     # Gán nhãn tương tác tập dữ liệu
│   ├── build_crnn_dataset.py # Cắt sliding windows & đóng gói dataset
│   ├── train_crnn.py         # Huấn luyện mạng Hybrid CRNN
│   ├── test_crnn.py          # Kiểm thử trực quan trên 1 file đơn lẻ
│   └── evaluate.py           # So sánh đối chuẩn tổng thể Baseline vs CRNN
│
└── outputs/
    ├── figures/              # Biểu đồ kết quả
    ├── logs/                 # File log runtime
    └── results/              # File CSV kết quả đối chuẩn