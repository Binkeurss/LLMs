---
title: 'Retrieval-Augmented Generation for AI-Generated Content: A Survey'

---

# 4 type generators are frequently used in RAG:
- **Transformer model**: Transformer hoạt động theo cơ chế nhìn toàn bộ chuỗi đầu vào, tập trung vào phần quan trọng, xử lý và sinh ra chuỗi đầu ra một cách tuần tự, rất mạnh trong các nhiệm vụ như dịch máy, tóm tắt văn bản, trả lời câu hỏi, v.v.
- **Long Short-Term Memory (LSTM)**: LSTM sử dụng phương pháp phân loại từ vựng (vocabulary classification) giống như mô hình Transformer, nhưng với cơ chế tự hồi quy (autoregressive), tức là mô hình sẽ sinh ra các đầu ra tuần tự, dựa vào các thông tin từ bước trước đó để dự đoán tiếp các từ tiếp theo trong chuỗi.
- **Diffusion Model**: là một kỹ thuật mạnh mẽ trong việc sinh ra các dữ liệu mới, đặc biệt hữu ích trong việc tạo hình ảnh, video, và nhiều dạng dữ liệu phức tạp khác.
    - Mô hình sẽ dần dần thêm nhiễu vào dữ liệu ban đầu cho đến khi nó trở thành một dữ liệu hoàn toàn ngẫu nhiên (noise).
    - Sau đó, mô hình sẽ đảo ngược quá trình này, tức là từ dữ liệu nhiễu (random noise), mô hình sẽ tạo ra dữ liệu mới dựa trên nhiễu đã có.
- **GAN (Generative Adversarial Networks)**: học đối kháng (adversarial learning)
    - Generator sẽ cố gắng cải thiện khả năng tạo ra các mẫu dữ liệu ngày càng thực tế hơn.
    - Discriminator sẽ liên tục cải thiện khả năng phân biệt giữa dữ liệu thật và giả.

# Retriever
- Mã hóa mỗi đối tượng (object) thành một biểu diễn cụ thể.
- Xây dựng chỉ mục (index) để tổ chức nguồn dữ liệu, từ đó giúp tìm kiếm hiệu quả hơn.
## **Sparse Retriever** (truy xuất thưa):
- Thường được sử dụng trong việc truy xuất tài.
- Giúp tìm kiếm tài liệu bằng cách sử dụng các chỉ số từ vựng và chỉ mục nghịch đảo, từ đó xác định và xếp hạng các tài liệu có liên quan dựa trên các phép đo thống kê.

## **Dense retrieval** 
Sử dụng các vector nhúng dày đặc và phương pháp ANN để truy xuất dữ liệu hiệu quả, đặc biệt là với các loại dữ liệu phức tạp như văn bản, mã nguồn, âm thanh, hình ảnh và video.

# METHODOLOGIES

## RAG Foundations

### Query-based RAG
- Ý tưởng tăng cường prompt, kết hợp truy vấn của người dùng với thông tin thu được từ truy xuất. Thông tin này sau đó được đưa trực tiếp vào giai đoạn đầu của input cho generator.

![image](https://hackmd.io/_uploads/S1QOye-Exl.png)

### Latent Representation-based RAG 
- Trong mô hình latent representation-based RAG, các đối tượng thu được từ truy xuất được tích hợp vào các mô hình sinh dưới dạng biểu diễn tiềm ẩn (latent representations).

- Cách này giúp tăng cường khả năng hiểu của mô hình và cải thiện chất lượng nội dung sinh ra.

![image](https://hackmd.io/_uploads/B1BKJlZ4xg.png)

### Logit-based RAG (RAG dựa trên logit):

- Trong mô hình logit-based RAG, các mô hình sinh tích hợp thông tin truy xuất thông qua logits trong quá trình giải mã.

- Thông tin truy xuất và các xác suất từ mô hình sinh được kết hợp để tạo ra đầu ra trong từng bước.

![image](https://hackmd.io/_uploads/ByWqJlWVgx.png)

### Speculative RAG
- Speculative RAG tìm kiếm cơ hội sử dụng truy xuất thay vì sinh thuần túy, nhằm tiết kiệm tài nguyên và tăng tốc độ phản hồi.

![image](https://hackmd.io/_uploads/BJg2JlW4gg.png)

## RAG Enhancements
- Mục tiêu nâng cao: input (đầu vào), retriever (trình truy xuất), generator (mô hình sinh), result (kết quả) và toàn bộ quy trình RAG (pipeline).

### 1) Input Enhancement:
Các phương pháp nâng cao đầu vào bao gồm biến đổi truy vấn (query transformation) và tăng cường dữ liệu (data augmentation).
#### Query Transformation:
Phương pháp này cải thiện kết quả truy xuất bằng cách thay đổi truy vấn ban đầu:

#### Data Augmentation:
Tăng cường dữ liệu trước khi truy xuất, bao gồm các kỹ thuật như loại bỏ thông tin không liên quan, giảm thiểu sự mơ hồ, cập nhật tài liệu cũ, tạo dữ liệu mới, v.v.

### 2) Retriever Enhancement:
Chất lượng của trình truy xuất xác định thông tin mà mô hình sinh sẽ sử dụng.

Các phương pháp nâng cao truy xuất bao gồm:

#### Recursive Retrieval (Truy xuất đệ quy):
Truy xuất đệ quy thực hiện nhiều lượt tìm kiếm để lấy nội dung phong phú và chất lượng hơn.

#### Chunk Optimization (Tối ưu hóa đoạn):
Điều chỉnh kích thước các đoạn văn bản để có kết quả truy xuất tốt hơn.

#### Retriever Finetuning (Tinh chỉnh trình truy xuất):
Trình truy xuất có thể được tinh chỉnh với các mô hình nhúng mạnh mẽ (embedding models) để cải thiện chất lượng truy xuất.

#### Retriever Finetuning (Tinh chỉnh trình truy xuất):
Trình truy xuất có thể được tinh chỉnh với các mô hình nhúng mạnh mẽ (embedding models) để cải thiện chất lượng truy xuất.

#### Hybrid Retrieval (Truy xuất lai):
Sử dụng kết hợp giữa truy xuất thưa và truy xuất dày đặc để cải thiện chất lượng truy xuất.

#### Re-ranking (Sắp xếp lại kết quả truy xuất):
Kỹ thuật Re-ranking (sắp xếp lại) liên quan đến việc đặt lại thứ tự của các kết quả truy xuất để đạt được sự đa dạng cao hơn và kết quả tốt hơn.****

#### Retrieval Transformation (Biến đổi truy xuất):
Retrieval Transformation là quá trình chỉnh sửa lại nội dung truy xuất để tối ưu hóa khả năng của generator, từ đó cải thiện kết quả đầu ra.

#### Các phương pháp tối ưu hóa khác trong quá trình truy xuất:
Ngoài các phương pháp trên, còn có một số phương pháp tối ưu hóa khác giúp cải thiện quá trình truy xuất:

- Meta-data filtering: Đây là phương pháp sử dụng dữ liệu siêu (metadata) như thời gian, mục đích, v.v. để lọc các tài liệu truy xuất nhằm cải thiện kết quả truy xuất.

- GENREAD và GRG giới thiệu một cách tiếp cận mới, thay thế hoặc cải thiện quá trình truy xuất bằng cách khuyến khích LLM tạo ra tài liệu để trả lời câu hỏi cụ thể.

= Multi-Head-RAG sử dụng nhiều mô hình embedding để chiếu một đoạn văn bản vào nhiều không gian vector khác nhau và sử dụng multi-head attention layer để nắm bắt các khía cạnh thông tin khác nhau, từ đó tăng độ chính xác của quá trình truy xuất.

![image](https://hackmd.io/_uploads/HJu3ol-4ex.png)

### 3) Generator Enhancement
Trong hệ thống RAG, chất lượng của generator (mô hình sinh) thường quyết định chất lượng của kết quả cuối cùng. Do đó, khả năng của generator sẽ xác định giới hạn tối đa về hiệu quả của toàn bộ hệ thống RAG.