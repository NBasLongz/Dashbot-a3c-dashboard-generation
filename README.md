# DashBot: Insight-Driven Dashboard Generation

Dự án tái hiện nghiên cứu **DashBot** dựa trên bài báo khoa học **"DashBot: Insight-Driven Dashboard Generation Based on Deep Reinforcement Learning"** trong môn học Trí tuệ nhân tạo.

---

##  Thành viên thực hiện (Nhóm 4 - UIT)

| MSSV | Họ và tên |
| :--- | :--- |
| **23520880** | Nguyễn Bá Long |
| **20521170** | Nguyễn Quốc Đạt |
| **23521355** | Nguyễn Nhật Sơn |

---

##  1. Cấu trúc thư mục dự án

```text
backend/dashbot/
  api/        - Các API FastAPI endpoint (health, profile, recommend)
  core/       - Xử lý dữ liệu (Profiler, Insight Detector, Chart Generator, Recommenders)
  rl_env/     - Môi trường RL (DashboardEnv, Reward Engine, Constraints)
  agent/      - Mô hình mạng học máy Bi-LSTM Actor-Critic và Policy Sampler
frontend/
  index.html  - Giao diện web người dùng tải CSV và xem biểu đồ gợi ý
  css/        - Stylesheet giao diện
  js/         - Xử lý gọi API và vẽ biểu đồ Vega-Lite
configs/      - Cấu hình huấn luyện mặc định (YAML)
reports/      - Logs huấn luyện (CSV) và các biểu đồ so sánh mô hình (Fig 6)
scripts/      - Các script chuẩn bị dữ liệu, huấn luyện (A3C, DQN) và đánh giá mô hình
tests/        - Bộ unit test kiểm thử môi trường và logic cốt lõi
```

---

##  2. Hướng dẫn cài đặt nhanh

### Bước 1: Cài đặt thư viện phụ thuộc
Cài đặt các thư viện cần thiết thông qua tệp `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Bước 2: Thiết lập biến môi trường `PYTHONPATH`
Để chạy các script trong thư mục `scripts/` hoặc khởi chạy API backend, bạn cần thêm thư mục `backend` vào biến môi trường:

* **Windows PowerShell**:
  ```powershell
  $env:PYTHONPATH='backend'
  ```
* **macOS / Linux**:
  ```bash
  export PYTHONPATH=backend
  ```
* **Windows CMD**:
  ```cmd
  set PYTHONPATH=backend
  ```

---

##  3. Khởi chạy hệ thống Demo

### Bước 1: Chạy FastAPI Backend API
Sau khi thiết lập `PYTHONPATH`, chạy lệnh sau để bật server backend:
```bash
python -m uvicorn dashbot.api.main:app --host 127.0.0.1 --port 8010
```

### Bước 2: Mở giao diện Frontend
* Chỉ cần mở tệp `frontend/index.html` bằng bất kỳ trình duyệt web nào.
* Bạn có thể tải lên các tệp dữ liệu CSV mẫu từ thư mục `data/processed/` (ví dụ `cars.csv`, `movies.csv`, `penguins.csv`) để kiểm nghiệm tính năng gợi ý dashboard tự động thời gian thực.

---

##  4. Chạy kiểm thử & Vẽ lại đồ thị đối chứng (Ablation Study)

### Chạy Unit Test
Để xác nhận tính đúng đắn của môi trường và các hàm tính toán phần thưởng:
```bash
python -m pytest -q
```

### Đánh giá hiệu suất mô hình thực tế
Đánh giá điểm chất lượng reward trung bình của mô hình trên các tập dữ liệu mẫu:
```bash
python scripts/evaluate.py data/processed/cars.csv
```

### Vẽ lại biểu đồ so sánh các mô hình (Fig. 6)
Chạy script sau để vẽ đồ thị so sánh giữa 4 mô hình (`DashBot`, `DashBot-ind.`, `DashBot-pen.`, và `DQN`):
```bash
python scripts/plot_paper_figures.py learning-curve --dashbot-log reports/ablation/training_curve_dashbot.csv --dashbot-ind-log reports/ablation/training_curve_dashbot_ind.csv --dashbot-pen-log reports/ablation/training_curve_dashbot_pen.csv --dqn-log reports/ablation/training_curve_dqn.csv --output reports/fig6_ablation_learning_curve.png
```
Đồ thị so sánh kết quả cuối cùng sẽ được lưu tại: `reports/fig6_ablation_learning_curve.png`.
