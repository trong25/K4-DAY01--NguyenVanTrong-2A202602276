# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU / GPU (T4)

**Python / PyTorch / Ultralytics:** Python 3.10+ / PyTorch 2.x / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không (sử dụng đúng mã nguồn, cấu hình và trọng số chuẩn của bản lab)

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  `{"class_id": 468, "class_name": "cab", "rank": 1, "score": 0.510915, "taxonomy_name": "ImageNet-1K"}` (thuộc sample `traffic`, COCO Image ID `210273`, kích thước 640x428).
- Record này mô tả toàn ảnh như thế nào?
  Record này đưa ra một nhãn phân loại duy nhất ở mức độ toàn cảnh (image-level prediction) cho cả bức ảnh. Nó kết luận ảnh thuộc lớp xe taxi (`cab`) dựa trên xác suất cao nhất trong số các lớp dự đoán, nhưng hoàn toàn không định vị tọa độ, kích thước hay phân biệt các thực thể riêng rẽ khác đang cùng xuất hiện trong ảnh (như xe buýt, người đi bộ, ô tô con).
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  Class list do những người xây dựng tập dữ liệu chuẩn ImageNet định nghĩa theo taxonomy ImageNet-1K (gồm 1.000 danh mục lớp được thiết kế sẵn) và được ghim cố định trong checkpoint `yolo11n-cls.pt`. Mô hình chỉ tính toán phân phối xác suất trên 1.000 lớp này chứ không thể tự phát sinh thêm nhãn nằm ngoài taxonomy.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - `class_id`: Định danh dạng số chuẩn hóa giúp máy tính xử lý truy vấn, mapping dữ liệu chính xác và tối ưu bộ nhớ, tránh xung đột về ngôn ngữ hoặc lỗi chính tả.
  - `class_name`: Giúp con người (annotator, reviewer, kỹ sư) đọc hiểu trực tiếp ý nghĩa ngữ nghĩa để đối soát và đánh giá trực quan.
  - `taxonomy_name` (`ImageNet-1K`): Cung cấp không gian ngữ cảnh (namespace). Cùng một mã ID hoặc tên lớp có thể mang phạm vi ngữ nghĩa hoàn toàn khác nhau giữa các taxonomy (ví dụ ImageNet-1K phân biệt `cab`, `minibus`, `streetcar` chi tiết, trong khi COCO-80 chỉ gộp chung là `car` hoặc `bus`). Việc lưu taxonomy đảm bảo tính toàn vẹn và khả năng truy xuất nguồn gốc (provenance) của dữ liệu.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  Guideline cần quy định rõ ràng:
  1. Tiêu chí lựa chọn nhãn đơn: Chọn đối tượng chiếm diện tích lớn nhất (dominant subject), đối tượng nằm ở tiền cảnh trung tâm, hay đối tượng là tiêu điểm lấy nét của ống kính.
  2. Cách xử lý khi không có chủ thể vượt trội: Quy định gán nhãn cảnh tổng quan bao quát (scene-level label) hoặc yêu cầu chuyển sang bài toán phân loại đa nhãn (multi-label classification).
  3. Thứ bậc ưu tiên phân loại (class hierarchy): Quy định mức độ ưu tiên cụ thể khi có nhiều loại phương tiện đan xen trong cùng một cảnh giao thông.
- Vì sao model score không phải ground truth?
  Model score (ở đây là 0.510915) chỉ là giá trị xác suất thống kê (confidence score từ hàm softmax) phản ánh mức độ tự tin của mạng nơ-ron dựa trên các trọng số đã học, không đại diện cho sự thật khách quan. Mô hình hoàn toàn có thể tự tin cao vào một phán đoán sai (overconfidence) hoặc cho điểm thấp đối với nhãn đúng. Ground truth bắt buộc phải là nhãn do con người thẩm định và xác nhận theo guideline chuẩn mực.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  `{"class_name": "person", "score": 0.912625, "bbox_xyxy": [385.33, 69.24, 498.92, 348.92], "bbox_width": 113.58, "bbox_height": 279.68}` (thuộc sample `kitchen`, COCO Image ID `397133`, ngưỡng lọc `score_threshold: 0.35`).
- Diễn giải vị trí box bằng lời:
  Hộp giới hạn (bounding box) bao quanh người đứng nấu bếp ở khu vực phía bên phải của khung hình. Tọa độ góc trên-bên trái là (x_min = 385.33 px, y_min = 69.24 px) và góc dưới-bên phải là (x_max = 498.92 px, y_max = 348.92 px). Hộp có bề ngang 113.58 px và chiều cao 279.68 px, bao quát từ phần đầu xuống đến quá nửa người của nhân vật.
- So sánh số prediction ở hai threshold:
  - Ở ngưỡng `threshold = 0.35`: Mô hình giữ lại 11 prediction (2 person, 5 bowl, 2 oven, 2 cup).
  - Ở ngưỡng `threshold = 0.60`: Mô hình chỉ giữ lại 6 prediction có độ tự tin cao (gồm 2 person, 2 bowl, 2 oven; loại bỏ 5 đối tượng có score dưới 0.60 gồm 3 bowl và 2 cup).
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  - Khi đặt ngưỡng thấp (như 0.35): Độ bao phủ (recall) tăng, nhận diện được nhiều vật thể nhỏ hoặc mờ, nhưng kéo theo nhiều dương tính giả (false positives, ví dụ các vùng nền bị nhận diện nhầm). Khối lượng kiểm tra của reviewer/QC tăng lên đáng kể vì phải rà soát và xóa bỏ nhiều box rác.
  - Khi đặt ngưỡng cao (như 0.60): Độ chính xác (precision) của prediction cao hơn, reviewer xử lý ít box hơn nhưng độ bao phủ giảm mạnh, làm tăng nguy cơ bỏ sót vật thể thực tế (false negatives). Khi đó annotator phải tốn công gán bù thủ công các vật thể bị thiếu để đạt chuẩn ground truth.
- Đề xuất một quy tắc box chặt:
  Bounding box phải bao trọn các điểm cực viền (extreme edge pixels) của vật thể theo cả 4 hướng, khoảng trống dư thừa tiếp giáp nền không được vượt quá 2-3 pixel và tuyệt đối không cắt lẹm vào bất kỳ phần nhìn thấy nào của đối tượng.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  - Guideline cần quy định: Tỷ lệ hiển thị tối thiểu để gán nhãn (ví dụ: vật thể bị che khuất > 80% hoặc bị cắt mép chỉ còn < 15% diện tích thì bỏ qua hay vẫn gán nhãn); đồng thời nêu rõ vẽ box theo phần nhìn thấy (visible box) hay vẽ ước lượng cả phần bị khuất (amodal bounding box).
  - Escalation: Khi gặp đối tượng bị chia cắt thành nhiều phần không liên tục do vật cản ở giữa (ví dụ một người bị cột chắn ngang thân), annotator cần gửi yêu cầu escalation xin quyết định: vẽ 1 box lớn gộp chung hay tách thành 2 box riêng biệt có cờ thuộc tính `occluded`.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  `{"instance_id": "kitchen-001", "class_name": "person", "score": 0.899318, "polygon_point_count": 348, "polygon_xy": [[434.0, 70.0], [427.0, 70.0], [423.0, 74.0], [417.0, 74.0], [414.0, 77.0], [409.0, 77.0], ...]}`.
- Polygon bổ sung chi tiết gì so với box?
  Polygon cung cấp mặt nạ ranh giới chuẩn xác tới từng pixel (pixel-level contour) mô tả đúng hình dạng thực tế của đối tượng. Khác với bounding box hình chữ nhật luôn chứa cả các khoảng trống nền xung quanh, polygon bám khít theo đường lượn của cơ thể, tay áo và các chi tiết uốn cong, giúp phân biệt rõ từng pixel thuộc về đối tượng và pixel thuộc về hậu cảnh hoặc vật dụng khác.
- `instance_id` dùng để làm gì và không phải loại ID nào?
  `instance_id` (như `kitchen-001`, `kitchen-002`) là mã định danh cá thể cục bộ để phân biệt các đối tượng độc lập với nhau trong cùng một mẫu ảnh (ví dụ phân biệt người thứ nhất với người thứ hai, hoặc tách biệt từng chiếc bát).
  `instance_id` **không phải** là `class_id` (mã định danh danh mục lớp chung, ví dụ class 0 là person) và **không phải** là `tracking_id` (mã theo dõi xuyên suốt danh tính qua các chuỗi frame video liên tiếp).
- Đề xuất một quy tắc biên mask:
  Quy tắc bám biên liên tục (Boundary adherence rule): Đường polygon viền quanh vật thể phải nằm đúng trên rìa tương phản tách biệt giữa đối tượng và nền; sai số lệch biên không được vượt quá 1-2 pixel. Mask không được cắt lẹm vào chi tiết vật thể và không được bao trùm phần bề mặt của vật thể khác hoặc nền xung quanh.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  - Ranh giới tiếp xúc đè lớp: Khi hai đối tượng chồng lấn hoặc chạm nhau (ví dụ cái bát đặt trên mặt bàn, bàn tay cầm thìa), guideline cần quy định rõ pixel ranh giới thuộc về đối tượng ở lớp trên (foreground instance).
  - Vùng mờ chuyển động và bóng đổ: Guideline cần xác định quy tắc loại bỏ bóng đổ bề mặt (cast shadows) ra khỏi mask và cách xử lý biên nhòe do chuyển động (motion blur).
  - Escalation: Khi một phần đối tượng bị che khuất một phần ranh giới hoặc ánh sáng quá tối không thể phân biệt ranh giới bằng mắt thường, annotator cần escalate để xin hướng dẫn xem có nội suy giả định hình học hay chỉ bao khép kín phần nhìn thấy rõ ràng.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn cấp ảnh: Một cặp `(class_id, class_name)` hoặc vector phân phối đa nhãn (multi-label) cho toàn bức ảnh. | Ảnh `traffic` có nhiều loại xe nhưng mô hình chỉ gán 1 nhãn `cab` (score 0.51); ảnh `kitchen` bị dự đoán sai thành `gong` (score 0.42). | Xác định chủ thể chính hoặc bối cảnh chủ đạo của ảnh dựa theo đúng tiêu chí ưu tiên trong guideline; không gán theo cảm tính. | Kiểm tra nhãn có bao quát đúng nội dung chính theo guideline hay không; đối soát quy tắc gán nhãn khi ảnh có nhiều chủ thể. |
| Phát hiện vật thể | Danh sách các bounding box cho từng thực thể: `[class_id, x_min, y_min, x_max, y_max]` theo pixel kèm nhãn lớp. | Bỏ sót các vật thể nhỏ/xa; ở ảnh `kitchen` mô hình chia tủ kệ thành 2 lò nướng `oven`, box người ở góc trái bị cắt lẹm (`score=0.61`). | Vẽ box bám sát 4 mép ngoài cùng của từng vật thể; thêm box cho các vật thể bị bỏ sót; gắn cờ thuộc tính che khuất/cắt mép. | Kiểm tra độ ôm khít của box (IoU), phát hiện box vẽ ẩu thừa nền hoặc lẹm vào vật thể; rà soát không để sót object mục tiêu. |
| Instance segmentation | Danh sách các đa giác mặt nạ cho từng cá thể: `instance_id`, `class_id`, tọa độ đa giác `polygon_xy` theo pixel. | Đường viền người trong `kitchen` bị gợn răng cưa; mask mặt bàn `kitchen-005` (598 điểm) lan rộng đè lên sàn nhà và chân tủ lò. | Vẽ polygon bám sát từng khúc uốn của viền đối tượng; tách riêng các phần chồng lấn giữa các thực thể chạm nhau. | Phóng to kiểm tra độ mượt và chuẩn xác của đường biên (boundary IoU); kiểm tra mask không lấn sang nền hoặc vật thể lân cận. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  Tuyệt đối không tải lên, lưu trữ hoặc chia sẻ dữ liệu nhạy cảm, thông tin nhận dạng cá nhân (PII như khuôn mặt rõ nét chưa làm mờ, biển số xe đọc được, thông tin căn cước/khách hàng) hoặc dữ liệu nội bộ/độc quyền lên Google Colab, GitHub repository công khai hay bất kỳ nền tảng trực tuyến nào; chỉ sử dụng các tập dữ liệu mẫu công khai được cấp phép và kiểm tra tính toàn vẹn bằng checksum (SHA-256).
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  Tôi sẽ dừng thao tác xử lý ngay lập tức, không sao chép hay lan truyền dữ liệu bất thường đó, và báo cáo ngay cho **Giảng viên hướng dẫn / Lab Coach / cán bộ phụ trách an toàn dữ liệu của khóa học** để nhận chỉ dẫn xử lý và cô lập dữ liệu.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
