# Chạy Human Counting Project trong VS Code

Môi trường trên máy này đã sẵn sàng: `.venv` dùng Python 3.11, PyTorch CUDA và GPU RTX 3070. Các cấu hình F5 dùng thẳng Python trong `.venv`, không cần chạy lệnh kích hoạt môi trường.

## 1. Mở đúng thư mục

Trong VS Code, chọn **File → Open Folder…** và mở:

```text
D:\Kì IV\CPV301\Human_Counting_Project
```

VS Code đã có các extension Python, Python Debugger và Pylance. Danh sách đề xuất cũng được lưu trong `.vscode/extensions.json` để dùng trên máy khác.

Nếu thanh trạng thái đang chọn Python khác, nhấn **Ctrl+Shift+P → Python: Select Interpreter → Enter interpreter path…** và chọn:

```text
.venv\Scripts\python.exe
```

Python mặc định ngoài dự án trên máy này là 3.14. Không dùng lệnh `python app.py` hoặc nút **Run Python File** để khởi chạy Streamlit. Dùng các cấu hình bên dưới để tránh chọn nhầm Python hoặc sai cách khởi chạy.

## 2. Chạy với video có sẵn — nên thử đầu tiên

1. Nhấn **Ctrl+Shift+D** để mở **Run and Debug**.
2. Chọn **1. Video - Streamlit** ở danh sách cấu hình phía trên.
3. Nhấn **F5**. Bước `Check environment` kiểm tra Python, thư viện, model và GPU trước khi ứng dụng chạy.
4. Mở **http://127.0.0.1:8501** trong trình duyệt. Có thể giữ **Ctrl** và bấm URL trong terminal của VS Code.
5. Upload `data/videos/people_walking.mp4` hoặc video riêng, rồi nhấn **Run Human Counting**.

Giao diện tải video sử dụng YOLO11n và cách đếm của chế độ video hiện tại. Chế độ camera trực tiếp dùng bộ xử lý đã cải thiện riêng.

## 3. Chạy camera iPhone

1. Trong **Run and Debug**, chọn **2. Camera - YOLO11s (GPU)** rồi nhấn **F5**.
2. Terminal in dòng **Computer dashboard (keep private): http://127.0.0.1:8080/#…**.
3. Mở **toàn bộ link này**, gồm cả phần sau dấu `#`, để vào bảng điều khiển trên máy tính.
4. Nhấn **Ctrl+Shift+P → Tasks: Run Task → Start HTTPS tunnel (iPhone)**. Công cụ tunnel đã có sẵn trên máy này.
5. Trong terminal của tunnel, tìm địa chỉ dạng `https://….trycloudflare.com`.
6. Dán địa chỉ đó vào ô **Địa chỉ HTTPS** trên bảng điều khiển, rồi nhấn **Lưu cấu hình**.
7. Nhấn **Sao chép link camera**, gửi link đó cho iPhone, mở bằng Safari và nhấn **Bắt đầu truyền**.

Giữ cả server camera và tunnel đang chạy. iPhone có thể dùng 4G/5G, không cần cùng Wi-Fi. Link quản trị dùng trên máy tính; chỉ chia sẻ link camera. Mỗi lần khởi động lại server sẽ tạo mã truy cập mới; mỗi lần tạo tunnel mới có thể đổi địa chỉ HTTPS.

Nếu máy xử lý chậm, có thể chọn **3. Camera - YOLO11n (lighter)**. Cấu hình này nhẹ hơn nhưng có thể đếm kém chính xác hơn.

## 4. Kiểm thử khi chưa có camera

Nhấn **Ctrl+Shift+P → Tasks: Run Task**, rồi chọn:

| Task | Mục đích |
|---|---|
| `Check environment` | Kiểm tra Python, thư viện, model và GPU |
| `Run tests` | Chạy các bài kiểm thử bộ đếm, tracking và server |
| `Test sample video (YOLO + WebSocket)` | Gửi video mẫu qua WebSocket cục bộ và xử lý bằng model thật |

Task video mẫu không cần mở server camera trước; nó tự chạy một server kiểm thử riêng và đóng sau khi hoàn thành. Kết quả được lưu trong `outputs/vscode_smoke/`, gồm `report.json`, `events.csv` và `last_frame.jpg`.

## 5. Dừng chương trình

- Với ứng dụng chạy bằng F5: nhấn **Shift+F5**.
- Với tunnel chạy trong task: chọn đúng terminal của tunnel rồi nhấn **Ctrl+C**, hoặc chọn **Tasks: Terminate Task → Start HTTPS tunnel (iPhone)**.
- Chỉ chạy một cấu hình camera tại một thời điểm vì cả hai dùng cổng 8080.

## 6. Tạo lại môi trường khi cần

**Máy hiện tại không cần cài lại.** Khi clone sang máy Windows khác, cần có Python 3.11. Nhấn **Ctrl+Shift+P → Tasks: Run Task → Setup environment**, hoặc chạy từ terminal tại thư mục dự án:

```powershell
python scripts/setup_environment.py
```

Script tái sử dụng `.venv` có sẵn hoặc tìm Python 3.11 để tạo mới, cài `requirements.txt`, kiểm tra thư viện và tải model còn thiếu từ nguồn Ultralytics chính thức. Script không xóa môi trường cũ và không sửa Python mặc định hay execution policy của PowerShell.

Nếu script không tìm được Python 3.11, chỉ rõ đường dẫn:

```powershell
python scripts/setup_environment.py --python "C:\duong-dan\Python311\python.exe"
```

Nếu máy khác chưa có công cụ tunnel, chạy task **Install HTTPS tunnel tool**, hoặc:

```powershell
python scripts/setup_environment.py --with-tunnel
```

Máy mới có thể cần bản PyTorch CUDA phù hợp để dùng GPU. Script cài đặt không thay một bản CUDA đang thỏa `requirements.txt`; hãy giữ `.venv` hiện tại trên máy này.

## Xử lý lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `ModuleNotFoundError` hoặc đang chạy Python 3.14 | Chọn cấu hình F5 đã tạo và interpreter `.venv/Scripts/python.exe`; chạy `Check environment` |
| Không có cấu hình F5 | Mở **cả thư mục dự án**, không chỉ một file Python; kiểm tra extension Python Debugger |
| Cổng 8080/8501 đã được sử dụng | Dừng phiên chạy cũ trong terminal hoặc bằng Shift+F5 rồi chạy lại |
| Mở bảng điều khiển nhưng hỏi mã quản trị | Mở đầy đủ link mới trong terminal, gồm phần `#…` |
| Không mở được camera trên iPhone | Mở link camera HTTPS bằng Safari, cấp quyền camera, giữ trang ở phía trước |
| Task tunnel báo không tìm thấy file | Chạy `Install HTTPS tunnel tool` trước |

Các lệnh terminal trực tiếp, nếu không muốn dùng F5:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe live_server.py
```

Chạy từng lệnh ở terminal riêng nếu muốn mở đồng thời hai chế độ.
