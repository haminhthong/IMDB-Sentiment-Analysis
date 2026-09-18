# Dataset layout

`data/raw/` chứa Large Movie Review Dataset official sau khi chạy
`scripts/download_imdb.py`. `data/processed/` dành cho dữ liệu đã chuẩn hóa.
Hai thư mục này được gitignore vì dữ liệu đầy đủ không thuộc source repository.

Fixture smoke nằm ở `tests/fixtures/imdb_smoke/` và được tạo bởi
`python -m scripts.create_smoke_dataset`. Fixture này dùng để kiểm tra thủ công,
không được đặt tên `train.csv`/`test.csv` ở root, không đại diện cho IMDB và
không được CI dùng để báo cáo chất lượng.
