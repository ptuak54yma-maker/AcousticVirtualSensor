# Quy trình chạy và kiểm tra Acoustic Virtual Sensor trên STM32 ảo

## 1. Phạm vi và phân biệt hai luồng chạy

Có hai luồng riêng:

1. **STM32 ảo trên Renode:** nạp ELF bare-metal vào model board STM32F4 Discovery. Firmware chạy một frame âm thanh tổng hợp qua FFT, NNLS và event sum; sau đó chạy một peak test có kết quả biết trước. Đây là smoke test cho firmware và peripheral UART.
2. **Đối chiếu WAV trên host:** chạy C++ native và Python reference với toàn bộ `data/test/Bowl_0.wav`. WAV chưa được stream vào MCU ảo; kết quả WAV không phải output từ Renode.

Target firmware giả định STM32F407VG: Cortex-M4F, 1 MiB Flash, 128 KiB SRAM chính và 64 KiB CCM. Linker chỉ dùng Flash và SRAM chính. Renode dùng model STM32F4 Discovery tương thích, không phải mô hình memory chính xác riêng cho F407VG. Không có cấu hình ADC, microphone, DMA, I2S hoặc clock tree tùy chỉnh; clock mặc định 16 MHz HSI được dùng cho UART.

Các toolchain, CMSIS-DSP, Renode và model `W_standard.npy` đã được thiết lập trong môi trường triển khai. Các bước dưới đây chỉ build/chạy, không cài hoặc tải lại chúng. Mô hình NMF không được train lại hay thay đổi.

## 2. Điều kiện trước khi chạy

Mở PowerShell tại thư mục repository:

```powershell
Set-Location 'E:\HUST\Master\Luận văn\AcousticVirtualSensor'
```

Các script build đã tạo junction ASCII `%LOCALAPPDATA%\AVSRepoLink` để tránh lỗi xử lý đường dẫn tiếng Việt của một số binary Windows. Nếu repository được chuyển sang đường dẫn khác, xóa junction cũ chỉ khi chắc chắn nó là junction `AVSRepoLink` trỏ tới repository cũ; script sẽ báo lỗi nếu alias hiện tại trỏ sai.

## 3. Build firmware ARM

Chạy:

```powershell
.\embedded\firmware\build_stm32f407.ps1
```

Script cấu hình/build Cortex-M4 Thumb-2, FPv4-SP-D16, hard-float ABI; dùng CMSIS-DSP real FFT 2048 điểm; và chép kết quả vào:

- `embedded/firmware/out/avs_stm32f407.elf`
- `embedded/firmware/out/avs_stm32f407.map`

Kết quả lần triển khai gần nhất: ELF32 little-endian ARM EABI hard-float; Flash **234.444 B / 1 MiB (22,36%)**; RAM linker **84.832 B / 128 KiB (64,72%)**. RAM đã tính 8 KiB stack reservation; `.data` là 128 B, phần BSS tĩnh là 76.512 B, CCM không dùng. NNLS workspace caller-owned là 39.240 B và được đặt ở vùng tĩnh, không nằm trên stack.

## 4. Chạy smoke test trên STM32 ảo

### Chế độ headless và lưu log

```powershell
.\embedded\firmware\sim\run_renode.ps1
```

Script chạy ELF trên STM32F4 Discovery model trong Renode, chờ firmware hoàn tất, rồi xác nhận `AVS_TEST_PASS`. UART4 log được lưu tại `embedded/firmware/sim/uart.log`.

### Chế độ cửa sổ Renode

```powershell
.\embedded\firmware\sim\run_renode_gui.ps1
```

Lệnh này mở Renode GUI với board và firmware đã nạp. UART4 analyzer hiện các thông báo firmware; có thể mở log cùng lúc ở `embedded/firmware/sim/uart.log`. Cửa sổ mô phỏng giữ trạng thái sau lượt chạy để có thể xem analyzer. Đóng cửa sổ Renode khi xong.

### Dữ liệu mà MCU ảo nhận trong smoke test

Firmware tự tạo **một frame 2048 mẫu float**, không đọc file WAV:

```text
x[n] = 0.25 × sin(32 × 2πn / 2048)
     + 0.10 × cos(77 × 2πn / 2048),  n = 0..2047
```

Firmware trừ mean frame, áp dụng periodic Hann, chạy CMSIS-DSP RFFT 2048 điểm và tạo `log1p(abs(X))` với 1025 bin. Sau đó solver NNLS dùng dictionary `W_standard` kích thước **1025 × 48**, rồi tính event activation sum. Peak detector được kiểm tra riêng bằng envelope tổng hợp:

```text
[0, 0, 0.02, 0, 0, 0.04, 0, 0]
```

Peak kỳ vọng là index **5**, count **1**. Envelope này là input kiểm thử peak, không phải envelope sinh ra từ WAV.

Các số trong UART log được nhân 1.000.000 và làm tròn thành số nguyên để in không cần `printf`:

- `FRAMES_PROCESSED=1`
- `FFT_BIN32_MICRO=4859813` → log magnitude bin 32 khoảng **4,859813**
- `ACTIVATION_0_MICRO=0` → activation component 0 bằng 0 trong frame smoke test
- `EVENT_SUM_MICRO=25` → event activation sum khoảng **0,000025**
- `PEAK_TEST_OK count=1 index=5`
- `AVS_TEST_PASS`

Giá trị FFT bin là đầu ra DSP thực của firmware. `EVENT_SUM_MICRO` là kết quả NNLS cho frame tổng hợp trên, không phải điểm phát hiện từ `Bowl_0.wav`.

## 5. Chạy native build và unit test

Native C++ build dùng MSVC Developer environment. Từ PowerShell ở repository root, chạy:

```powershell
cmd.exe /d /s /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && "C:\Program Files\CMake\bin\cmake.exe" --build "E:\HUST\Master\Luận văn\AcousticVirtualSensor\embedded\build\ninja" --parallel 4 && "C:\Program Files\CMake\bin\ctest.exe" --test-dir "E:\HUST\Master\Luận văn\AcousticVirtualSensor\embedded\build\ninja" --output-on-failure'
```

Kết quả gần nhất: `avs_smoke_test` **1/1 PASS**. Các test gồm FFT/Hann, NNLS với dictionary synthetic, peak output, plateau/tie, buffer capacity/overflow và dữ liệu không hữu hạn.

## 6. Đối chiếu toàn WAV giữa C++ native và Python

Chạy từ repository root:

```powershell
python embedded\tools\verify_against_python.py data\test\Bowl_0.wav `
  --binary embedded\build\ninja\avs_infer.exe `
  --build-dir embedded\build\compare-final `
  --report-only
```

`--report-only` giúp in đầy đủ chênh lệch dù max activation vượt tolerance 0,05. Bỏ cờ này sẽ làm chương trình kết thúc với mã lỗi 21 khi tolerance activation không đạt. Python reference sử dụng pipeline và dictionary có sẵn; không train NMF.

Input `Bowl_0.wav` có sample rate **44.100 Hz**, **1.322.327 mẫu**. Pipeline dùng frame 2048 mẫu, hop 1024, periodic Hann, trừ mean mỗi frame, log magnitude và cùng `W_standard.npy`. Số frame là `floor((1.322.327 - 2.048) / 1.024) + 1 = 1290`.

Kết quả đã thu:

| Đại lượng | Sai số max | Sai số mean | RMSE / đối chiếu | Đánh giá |
|---|---:|---:|---:|---|
| Log spectrum | 5,1232203e-05 | 7,7400586e-07 | 2,2548286e-06 | PASS, tolerance 0,001 |
| Activation H | 0,13473675 | 0,00041961476 | 0,0045688952 | FAIL, max tolerance 0,05 |
| Event envelope | 0,00047073164 | 9,7301348e-06 | 3,2514333e-05 | Đã đo; không đặt pass threshold riêng |
| Peak indices | — | — | 33/33 khớp, danh sách giống hệt | PASS |

Max activation error nằm ở frame 1099, component 41, thuộc một vùng NNLS gần suy biến có active set khác nhau. Không thay model hoặc tham số để ép test pass. Kết quả host WAV này không được chạy trên MCU ảo.

Artifact của lần so sánh gần nhất:

- `embedded/build/compare-final/verify/cpp_output.csv` — event sum và 48 activation cho mỗi frame.
- `embedded/build/compare-final/verify/cpp_output.csv.spectrum.csv` — log spectrum 1025 bin cho mỗi frame.
- `embedded/build/compare-final/verify/cpp_output.csv.peaks.csv` — peak indices của C++.
- `embedded/firmware/sim/uart.log` — dữ liệu UART từ firmware trên Renode.

## 7. Tóm tắt giới hạn hiện tại

- Renode smoke test chạy một frame tổng hợp và một envelope peak tổng hợp; chưa nạp WAV, microphone stream hay ADC vào MCU ảo.
- UART4 được cấu hình để quan sát log; chưa cấu hình ngoại vi thu âm, DMA, I2S hay clock tree tùy chỉnh.
- STM32F407VG là target assumption. Model Renode là STM32F4 Discovery tương thích, nên cần kiểm tra thêm trên đúng board/chip vật lý trước khi coi firmware sẵn sàng triển khai thiết bị.
- C++/Python activation trên toàn WAV chưa đạt max tolerance 0,05; spectrum và peak indices đạt các tiêu chí nêu trên.
