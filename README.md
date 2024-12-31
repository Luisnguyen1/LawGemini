# VN Law Chat 🤖⚖️

VN Law Chat là một chatbot thông minh được thiết kế để hỗ trợ tra cứu và giải đáp các vấn đề pháp luật tại Việt Nam.

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Tính năng chính](#tính-năng-chính)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Cách sử dụng](#cách-sử-dụng)
- [Cài đặt và triển khai](#cài-đặt-và-triển-khai)
- [Đóng góp](#đóng-góp)
- [Giấy phép](#giấy-phép)

## 🎯 Tổng quan

VN Law Chat là một ứng dụng AI được xây dựng để giúp người dùng dễ dàng tiếp cận và hiểu rõ các vấn đề pháp luật. Hệ thống sử dụng các mô hình ngôn ngữ tiên tiến để xử lý câu hỏi bằng tiếng Việt và cung cấp câu trả lời chính xác, đáng tin cậy dựa trên cơ sở dữ liệu pháp luật Việt Nam.

## ✨ Tính năng chính

- **Tra cứu văn bản pháp luật**: Tìm kiếm nhanh chóng các văn bản pháp luật liên quan
- **Hỏi đáp tương tác**: Trả lời các câu hỏi pháp luật bằng ngôn ngữ tự nhiên
- **Trích dẫn nguồn**: Cung cấp trích dẫn chính xác từ văn bản pháp luật gốc
- **Giải thích dễ hiểu**: Diễn giải các điều luật phức tạp bằng ngôn ngữ đơn giản
- **Cập nhật liên tục**: Dữ liệu pháp luật được cập nhật thường xuyên

## 🏗 Kiến trúc hệ thống

```
VN Law Chat/
├── frontend/         # Giao diện người dùng (React)
├── backend/          # API và xử lý logic (FastAPI)
├── models/          # Các mô hình AI và xử lý ngôn ngữ
├── data/            # Cơ sở dữ liệu văn bản pháp luật
└── docker/          # Cấu hình Docker
```

## 🚀 Cách sử dụng

1. Truy cập ứng dụng tại: [Link đến ứng dụng]
2. Nhập câu hỏi pháp luật của bạn vào ô chat
3. Hệ thống sẽ phân tích và trả về câu trả lời kèm theo trích dẫn nguồn
4. Bạn có thể đặt câu hỏi thêm để làm rõ vấn đề

## ⚙️ Cài đặt và triển khai

### Yêu cầu hệ thống
- Docker
- Python 3.8+
- Node.js 14+

### Các bước cài đặt

1. Clone repository:
```bash
git clone https://github.com/your-username/vn-law-chat.git
cd vn-law-chat
```

2. Xây dựng Docker image:
```bash
docker build -t vn-law-chat .
```

3. Chạy container:
```bash
docker run -p 8000:8000 vn-law-chat
```

## 🤝 Đóng góp

Chúng tôi rất hoan nghênh mọi đóng góp để cải thiện VN Law Chat. Bạn có thể:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit các thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request


## 👨‍💻 Tác giả

**Nguyễn Văn Mạnh**
- 🌐 Website: [https://vanmanh-dev.id.vn/](https://vanmanh-dev.id.vn/)
- 💼 Kỹ sư phần mềm tại TP. Hồ Chí Minh
- 🎯 Chuyên môn: Web Development, Software Engineering

---
Developed with ❤️ by Nguyễn Văn Mạnh
```
