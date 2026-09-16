"""Giao diện Streamlit tối giản cho phân loại sentiment review phim."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from sentiment.inference import PredictionResult, Predictor, load_predictor

DEFAULT_CHECKPOINT = os.getenv(
    "CHECKPOINT_PATH",
    "artifacts/model.pt",
)
MAX_BATCH_SIZE = 100

st.set_page_config(
    page_title="CineSentiment",
    page_icon="🎬",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_predictor(checkpoint_path: str) -> Predictor:
    """Nạp và cache model theo đúng predictor contract của project."""
    return load_predictor(checkpoint_path, device="cpu")


def render_prediction(result: PredictionResult) -> None:
    """Hiển thị một kết quả dự đoán và thông tin audit ngắn gọn."""
    confidence = max(result.probability, 1.0 - result.probability)
    columns = st.columns(3)
    columns[0].metric("Nhãn", result.label)
    columns[1].metric("Xác suất Positive", f"{result.probability:.1%}")
    columns[2].metric("Độ tin cậy", f"{confidence:.1%}")

    if result.warnings:
        st.warning("Cảnh báo: " + ", ".join(result.warnings))

    truncated = "Có" if result.truncated else "Không"
    st.caption(f"Token: {result.token_count} | OOV: {result.oov_rate:.1%} | Cắt chuỗi: {truncated}")


def clean_lines(value: str) -> list[str]:
    """Tách batch input theo dòng và bỏ dòng trống."""
    return [line.strip() for line in value.splitlines() if line.strip()]


def render_batch(predictor: Predictor) -> None:
    """Hiển thị form dự đoán nhiều review."""
    batch_input = st.text_area(
        "Mỗi dòng là một review tiếng Anh",
        height=180,
        placeholder="Great acting and a moving story.\nThe plot was boring.",
    )
    if not st.button("Phân tích batch", type="primary"):
        return

    texts = clean_lines(batch_input)
    if not texts:
        st.warning("Vui lòng nhập ít nhất một review.")
        return
    if len(texts) > MAX_BATCH_SIZE:
        st.warning(f"Batch tối đa {MAX_BATCH_SIZE} review.")
        return

    with st.spinner("Đang phân tích..."):
        results = predictor.predict_batch(texts)

    rows = [
        {
            "Review": text[:120] + ("..." if len(text) > 120 else ""),
            "Nhãn": result.label,
            "Xác suất Positive": f"{result.probability:.1%}",
            "Độ tin cậy": f"{max(result.probability, 1.0 - result.probability):.1%}",
            "Token": result.token_count,
            "OOV": f"{result.oov_rate:.1%}",
            "Cắt chuỗi": "Có" if result.truncated else "Không",
            "Cảnh báo": ", ".join(result.warnings),
        }
        for text, result in zip(texts, results, strict=True)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


st.title("🎬 CineSentiment")
st.write("Phân loại cảm xúc review phim tiếng Anh bằng model đã huấn luyện.")

with st.sidebar:
    st.header("Model")
    checkpoint_path = st.text_input(
        "Đường dẫn checkpoint",
        value=DEFAULT_CHECKPOINT,
        help="Có thể trỏ tới model.pt hoặc model.joblib.",
    )

predictor: Predictor | None = None
checkpoint = Path(checkpoint_path)
if checkpoint.is_file():
    try:
        predictor = get_predictor(str(checkpoint))
    except (EOFError, KeyError, OSError, RuntimeError, ValueError) as error:
        st.sidebar.error(f"Không thể nạp model: {error}")
else:
    st.sidebar.warning("Chưa tìm thấy checkpoint.")

if predictor is None:
    st.info(
        "Hãy huấn luyện model trước, hoặc nhập đường dẫn tới model.pt/model.joblib ở thanh bên."
    )
else:
    st.sidebar.success(f"Đã nạp: {predictor.config.model_type}")
    single_tab, batch_tab = st.tabs(["Review đơn", "Review theo batch"])

    with single_tab:
        review = st.text_area(
            "Nội dung review",
            height=180,
            placeholder="This movie was touching, well acted, and memorable.",
        )
        if st.button("Phân tích review", type="primary"):
            if not review.strip():
                st.warning("Vui lòng nhập nội dung review.")
            else:
                with st.spinner("Đang phân tích..."):
                    render_prediction(predictor.predict(review))

    with batch_tab:
        render_batch(predictor)
