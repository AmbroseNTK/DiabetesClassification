# Mini Project 

## 1. Đặt vấn đề

<p align="justify">
Tiểu đường là một trong những bệnh rất phổ biến hiện nay, nó gây ra những biến chứng ảnh hưởng đến sức khỏe con người, phát hiện sớm có thể giúp bệnh nhân có chiến lực tốt để ngăn ngừa kịp thời. Nhóm thực hiện đề tài ứng dụng các phương pháp học máy SVM, kNN và Random Forest để thực hiện xây dựng các mô hình học máy nhằm chẩn đoán bệnh tiểu đường dựa trên các thông số liên quan. Từ đó, khám phá ra những mô hình học máy tốt nhất trong các giải thuật trên để xây dựng một hệ thống dự đoán bệnh tiểu đường.
</p>

### 1.1. Chức năng yêu cầu của hệ thống

Hệ thống sẽ thực hiện thu thập các thông tin cần thiết về bệnh nhân như sau:  

- Tần suất mang thai (nếu có)
- Nồng độ Glucose trong máu
- Huyết áp
- Mức độ dày của da
- Nồng độ Insulin trong cơ thể
- Chỉ số khối của cơ thể (BMI)
- Các thông tin chức năng di truyền
- Tuổi

Từ đó, hệ thống sẽ thực hiện chẩn đoán bệnh tiểu đường và cho kết quả.

### 1.2. Non-functional Requirements

<p align="justify">
Sử dụng các thuật toán học máy SVM, kNN và Random Forest để huấn luyện. Đánh giá và so sánh để chọn ra giải thuật tối ưu nhất để [giải quyết](#datvande) bài toán này.
</p>

## 2. Data Collection

### 2.1. Mô tả dữ liệu huấn luyện mô hình học máy

Dữ liệu thu thập từ các khảo sát thực tế gồm 688 cá thể với 8 biến dữ liệu như sau:

- **Pregnancy**: Tần suất mang thai
- **Glucose**: Nồng độ glucose trong máu (mg/dL)
- **BP**: Huyết áp (mm Hg)
- **Skin**: Độ dày của da (mm)
- **Insulin**: Nồng độ *Insulin* sản xuất ra trong 2 giờ (mu U/ml)	
- **BMI**: Chỉ số khối của cơ thể (kg/m2)
- **Pedigree**: Chức năng di truyền đối với bệnh tiểu đường
- **Age**: Độ tuổi (log (years))
