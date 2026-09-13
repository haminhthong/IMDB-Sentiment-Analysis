"""Giao diện Web Demo Phân Loại Cảm Xúc & Hiệu Chuẩn Đánh Giá Phim — CineSentiment AI Studio v2.0."""

import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from sentiment.inference import Predictor, load_predictor
from sentiment.text import tokenize

# Cấu hình đường dẫn Artifacts & Checkpoints
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "artifacts/model.pt")
BASELINE_MODEL_PATH = Path("artifacts/baseline/model.joblib")
BASELINE_EXPLAINABILITY = Path("artifacts/baseline/explainability.json")
VAL_METRICS_PATH = Path("artifacts/validation_metrics.json")
TEST_METRICS_PATH = Path("artifacts/test_metrics.json")
ERROR_ANALYSIS_PATH = Path("artifacts/error_analysis.json")
COMPARISON_MD_PATH = Path("artifacts/model_comparison.md")

# Cấu hình Trang Streamlit
st.set_page_config(
    page_title="CineSentiment AI Studio — IMDB Sentiment & Reliability",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hệ thống Thiết Kế Giao Diện: Dark Cinematic Studio & Glassmorphism
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-dark: #070a12;
    --card-bg: rgba(15, 23, 42, 0.65);
    --card-border: rgba(255, 255, 255, 0.08);
    --accent-cyan: #06b6d4;
    --accent-indigo: #6366f1;
    --accent-purple: #a855f7;
    --accent-emerald: #10b981;
    --accent-rose: #f43f5e;
    --accent-amber: #f59e0b;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-primary);
}

h1, h2, h3, .brand-title {
    font-family: 'Outfit', sans-serif !important;
}

code, pre, .mono-font {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Header Banner */
.cine-hero {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 27, 75, 0.6) 50%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 20px;
    padding: 30px 36px;
    margin-bottom: 26px;
    backdrop-filter: blur(16px);
    box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.cine-hero::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 320px;
    height: 100%;
    background: radial-gradient(circle at 80% 20%, rgba(6, 182, 212, 0.15), transparent 70%);
    pointer-events: none;
}

.cine-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    display: inline-block;
}

.cine-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    max-width: 820px;
    line-height: 1.6;
    margin-bottom: 14px;
}

.tag-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #cbd5e1;
    margin-right: 8px;
}

/* Luxury Glass Card */
.glass-panel {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 22px 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-panel:hover {
    border-color: rgba(99, 102, 241, 0.3);
}

/* Metric Cards */
.kpi-card {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.75) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 18px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    transition: all 0.25s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    border-color: rgba(255, 255, 255, 0.16);
}

.kpi-label {
    font-size: 0.8rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    font-family: 'Outfit', sans-serif;
    color: #f8fafc;
    margin: 6px 0 2px 0;
}

.kpi-sub {
    font-size: 0.8rem;
    color: #64748b;
}

/* Duel Sentiment Badges */
.duel-card {
    border-radius: 18px;
    padding: 24px;
    backdrop-filter: blur(14px);
    position: relative;
    border: 1px solid transparent;
    transition: all 0.3s ease;
}

.duel-bilstm-pos {
    background: linear-gradient(145deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.25) 100%);
    border-color: rgba(16, 185, 129, 0.35);
    box-shadow: 0 10px 35px -10px rgba(16, 185, 129, 0.25);
}

.duel-bilstm-neg {
    background: linear-gradient(145deg, rgba(244, 63, 94, 0.12) 0%, rgba(136, 19, 55, 0.25) 100%);
    border-color: rgba(244, 63, 94, 0.35);
    box-shadow: 0 10px 35px -10px rgba(244, 63, 94, 0.25);
}

.duel-baseline-pos {
    background: linear-gradient(145deg, rgba(14, 165, 233, 0.1) 0%, rgba(3, 105, 161, 0.2) 100%);
    border-color: rgba(14, 165, 233, 0.3);
}

.duel-baseline-neg {
    background: linear-gradient(145deg, rgba(245, 158, 11, 0.1) 0%, rgba(180, 83, 9, 0.2) 100%);
    border-color: rgba(245, 158, 11, 0.3);
}

.verdict-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}

.verdict-badge {
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    font-family: 'Outfit', sans-serif;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Consensus Pill */
.consensus-match {
    background: rgba(16, 185, 129, 0.18);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.consensus-diverge {
    background: rgba(245, 158, 11, 0.18);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #fbbf24;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Token Inspector Chips */
.token-chip-box {
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    max-height: 220px;
    overflow-y: auto;
    line-height: 1.8;
}

.token-chip {
    display: inline-block;
    padding: 3px 9px;
    margin: 3px 2px;
    border-radius: 6px;
    font-size: 0.84rem;
    font-family: 'JetBrains Mono', monospace;
    transition: all 0.15s ease;
}

.token-chip-normal {
    background: rgba(255, 255, 255, 0.05);
    color: #cbd5e1;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.token-chip-normal:hover {
    background: rgba(255, 255, 255, 0.12);
    color: #ffffff;
}

.token-chip-unk {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(185, 28, 28, 0.35));
    color: #fca5a5;
    border: 1px solid #ef4444;
    font-weight: 600;
    box-shadow: 0 0 10px rgba(239, 68, 68, 0.2);
}

/* Callout Box */
.info-callout {
    background: rgba(30, 41, 59, 0.45);
    border-left: 4px solid var(--accent-indigo);
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 14px 0;
    font-size: 0.92rem;
    color: #cbd5e1;
}

/* Filmstrip Card for Errors */
.filmstrip-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-left: 4px solid #ef4444;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
    transition: border-color 0.2s;
}

.filmstrip-card:hover {
    border-color: rgba(239, 68, 68, 0.5);
}

/* Streamlit Tabs Customization */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.6);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
    color: #94a3b8;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(6, 182, 212, 0.2));
    color: #f8fafc !important;
    border: 1px solid rgba(99, 102, 241, 0.4);
}

/* Custom Status Dot */
.status-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.status-dot-green {
    background-color: #10b981;
    box-shadow: 0 0 10px #10b981;
}
.status-dot-yellow {
    background-color: #f59e0b;
    box-shadow: 0 0 8px #f59e0b;
}
.status-dot-red {
    background-color: #ef4444;
    box-shadow: 0 0 8px #ef4444;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# QUẢN LÝ BỘ NHỚ ĐỆM (CACHING) & NẠP ARTIFACTS
# ==============================================================================
@st.cache_resource
def get_model(path: str) -> Predictor | None:
    """Nạp predictor mô hình vào bộ nhớ đệm (singleton CPU)."""
    try:
        p = Path(path)
        if p.is_file():
            return load_predictor(p, device="cpu")
    except Exception:
        return None
    return None


@st.cache_data
def load_json_artifact(path: Path) -> dict | None:
    """Đọc tệp JSON từ thư mục artifacts một cách an toàn."""
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


bilstm_predictor = get_model(CHECKPOINT_PATH)
baseline_predictor = get_model(str(BASELINE_MODEL_PATH))
explain_data = load_json_artifact(BASELINE_EXPLAINABILITY)
error_data = load_json_artifact(ERROR_ANALYSIS_PATH)
val_metrics = load_json_artifact(VAL_METRICS_PATH)
test_metrics = load_json_artifact(TEST_METRICS_PATH)


# ==============================================================================
# SIDEBAR: ĐIỀU KHIỂN & THEO DÕI ENGINE
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
            <span style="font-size: 1.8rem;">🎬</span>
            <div>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; letter-spacing: -0.01em;">CineSentiment</h3>
                <span style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;">Production Studio</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Hệ thống Phân tích Cảm xúc & Hiệu chuẩn Độ tin cậy Xác suất")
    st.markdown("---")

    # Engine Health Section
    st.markdown("##### ⚡ Trạng Thái Engine")

    if bilstm_predictor is not None:
        temp_val = getattr(bilstm_predictor, "temperature", 1.0)
        vocab_size = (
            len(bilstm_predictor.vocabulary) if hasattr(bilstm_predictor, "vocabulary") else 0
        )
        param_count = (
            bilstm_predictor.model.count_parameters()
            if hasattr(bilstm_predictor, "model")
            else 658_049
        )
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #34d399; margin-bottom: 4px;">
                    <span class="status-dot status-dot-green"></span> BiLSTM (Calibrated)
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.6;">
                    • Tham số: <strong style="color: #e2e8f0;">{param_count:,}</strong><br>
                    • Từ điển: <strong style="color: #e2e8f0;">{vocab_size:,} từ</strong><br>
                    • Nhiệt độ T: <strong style="color: #38bdf8;">{temp_val:.4f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 12px; padding: 10px 14px; margin-bottom: 10px;">
                <span class="status-dot status-dot-red"></span> <strong style="color: #fca5a5;">BiLSTM:</strong> Thiếu checkpoint `{CHECKPOINT_PATH}`
            </div>
            """,
            unsafe_allow_html=True,
        )

    if baseline_predictor is not None:
        st.markdown(
            """
            <div style="background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.25); border-radius: 12px; padding: 12px 14px;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; margin-bottom: 4px;">
                    <span class="status-dot status-dot-green"></span> TF-IDF + Logistic Regression
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8;">
                    • Max Features: <strong style="color: #e2e8f0;">50,000 (1-2 ngrams)</strong><br>
                    • Tốc độ CPU: <strong style="color: #34d399;">~0.33 ms / sample</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 12px; padding: 10px 14px;">
                <span class="status-dot status-dot-yellow"></span> <strong style="color: #fde68a;">Baseline:</strong> Chưa tìm thấy model.joblib
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Decision Threshold Controller
    st.markdown("##### 🎚️ Ngưỡng Phán Quyết (Decision Threshold)")
    custom_threshold = st.slider(
        "Ngưỡng gán nhãn Positive:",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Điều chỉnh ngưỡng xác suất để cân bằng Precision và Recall theo yêu cầu nghiệp vụ.",
    )
    st.caption(f"Ngưỡng hiện tại: `{custom_threshold:.2f}` (Mặc định chuẩn: 0.50)")

    st.markdown("---")
    st.markdown("##### 🛡️ Tiêu Chuẩn Kỹ Thuật Cốt Lõi")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.6;">
            • <strong>Chống Rò Rỉ:</strong> Từ điển fit 100% từ Train split (80%).<br>
            • <strong>pack_padded_sequence:</strong> Triệt tiêu lãng phí tính toán padding.<br>
            • <strong>Temperature Scaling:</strong> Học T trên Calibration split độc lập.<br>
            • <strong>Audit OOV & Truncation:</strong> Chẩn đoán rủi ro suy luận tức thì.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("CineSentiment Studio • PyTorch & Scikit-Learn")


# ==============================================================================
# HEADER HERO BANNER
# ==============================================================================
st.markdown(
    """
    <div class="cine-hero">
        <div class="cine-title">CineSentiment AI Studio</div>
        <p class="cine-subtitle">
            Hệ thống phân loại cảm xúc đánh giá phim IMDB tiêu chuẩn công nghiệp:
            kết hợp mô hình học sâu <strong>BiLSTM với hiệu chuẩn xác suất (Temperature Scaling)</strong>
            đối chiếu trực tiếp với <strong>Baseline TF-IDF giải thích được</strong>.
        </p>
        <div>
            <span class="tag-pill">🎬 IMDB Large Dataset (50k)</span>
            <span class="tag-pill">🧠 BiLSTM 2-Layer + PackPadded</span>
            <span class="tag-pill">🎯 Calibrated (ECE: 2.6%)</span>
            <span class="tag-pill">⚡ Low CPU Latency</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# CẤU TRÚC 5 TABS CHỨC NĂNG CHÍNH
# ==============================================================================
tab_live, tab_compare, tab_errors, tab_batch, tab_arch = st.tabs(
    [
        "🧪 Live Sentiment Lab",
        "⚖️ So Sánh & Hiệu Chuẩn",
        "🔍 Lát Cắt Lỗi Ngôn Ngữ",
        "📦 Kiểm Thử Hàng Loạt",
        "📐 Kiến Trúc & REST API",
    ]
)


# ==============================================================================
# TAB 1: LIVE SENTIMENT LAB
# ==============================================================================
with tab_live:
    st.markdown("### 🧪 Phòng Thí Nghiệm Cảm Xúc Thời Gian Thực")
    st.write(
        "Nhập bài đánh giá phim tiếng Anh hoặc bấm chọn các ca kiểm thử điển hình bên dưới để "
        "quan sát đối chiếu song song giữa **BiLSTM Calibrated** và **Baseline TF-IDF**."
    )

    sample_library = {
        "masterpiece": (
            "This movie is an absolute masterpiece! Captivating performances, brilliant "
            "cinematography, and a thrilling soundtrack that stays with you forever."
        ),
        "mixed_ending": (
            "The performances were great and visuals were okay, but the ending was completely "
            "awful and ruined the entire movie experience for me."
        ),
        "negation_trick": (
            "I cannot say that I didn't enjoy the film, because the acting wasn't bad at all."
        ),
        "harsh_critic": (
            "Terrible script, zero character development, "
            "and incredibly boring dialogue throughout."
        ),
        "oov_heavy": (
            "The protagonist exhibited excessive schadenfreude and solipsistic megalomania in the "
            "cinematographic avant-garde portrayal."
        ),
    }

    if "input_review" not in st.session_state:
        st.session_state["input_review"] = sample_library["masterpiece"]

    # Preset Selection Grid
    p1, p2, p3, p4, p5, p_clr = st.columns(6)
    if p1.button("✨ Kiệt Tác Điện Ảnh", use_container_width=True):
        st.session_state["input_review"] = sample_library["masterpiece"]
    if p2.button("⚠️ Hỗn Hợp (Khen-Chê)", use_container_width=True):
        st.session_state["input_review"] = sample_library["mixed_ending"]
    if p3.button("🔄 Bẫy Phủ Định Kép", use_container_width=True):
        st.session_state["input_review"] = sample_library["negation_trick"]
    if p4.button("💥 Phê Bình Gay Gắt", use_container_width=True):
        st.session_state["input_review"] = sample_library["harsh_critic"]
    if p5.button("🧬 Từ Lạ (Nhiều OOV)", use_container_width=True):
        st.session_state["input_review"] = sample_library["oov_heavy"]
    if p_clr.button("🗑️ Xóa Nội Dung", use_container_width=True):
        st.session_state["input_review"] = ""

    review_text = st.text_area(
        "Nội dung văn bản đánh giá (Movie Review):",
        value=st.session_state["input_review"],
        height=130,
        placeholder="Type or paste your English movie review here...",
        key="review_area_input",
    )

    word_count = len(review_text.split()) if review_text.strip() else 0
    char_count = len(review_text)
    st.caption(f"Độ dài văn bản: `{char_count}` ký tự • `{word_count}` từ (ước lượng sơ bộ)")

    if st.button("🚀 Thực Hiện Phân Tích Song Song", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("⚠️ Vui lòng nhập nội dung đánh giá trước khi thực hiện phân tích.")
        elif bilstm_predictor is None:
            st.error(f"⚠️ Không tìm thấy checkpoint BiLSTM tại `{CHECKPOINT_PATH}`.")
        else:
            try:
                bilstm_res = bilstm_predictor.predict(review_text)
                baseline_res = (
                    baseline_predictor.predict(review_text)
                    if baseline_predictor is not None
                    else None
                )

                # Quyết định nhãn theo custom_threshold
                bilstm_is_pos = bilstm_res.probability >= custom_threshold
                bilstm_label_display = "Positive" if bilstm_is_pos else "Negative"

                baseline_is_pos = (
                    (baseline_res.probability >= custom_threshold)
                    if baseline_res is not None
                    else None
                )
                baseline_label_display = (
                    ("Positive" if baseline_is_pos else "Negative")
                    if baseline_res is not None
                    else "N/A"
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # Consensus Alert
                if baseline_res is not None:
                    if bilstm_is_pos == baseline_is_pos:
                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                                <div class="consensus-match">
                                    <span>🤝</span>
                                    <span><strong>ĐỒNG THUẬN HOÀN TOÀN:</strong> Cả hai mô hình đều kết luận là <strong>{bilstm_label_display.upper()}</strong></span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                                <div class="consensus-diverge">
                                    <span>⚡</span>
                                    <span><strong>BẤT ĐỒNG PHÁN QUYẾT:</strong> BiLSTM chọn <strong>{bilstm_label_display}</strong>, Baseline chọn <strong>{baseline_label_display}</strong></span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # DUEL CARDS
                col_m1, col_m2 = st.columns(2)

                # BiLSTM Card
                with col_m1:
                    card_class = (
                        "duel-bilstm-pos"
                        if bilstm_label_display == "Positive"
                        else "duel-bilstm-neg"
                    )
                    badge_color = "#34d399" if bilstm_label_display == "Positive" else "#f87171"
                    badge_emoji = "😃" if bilstm_label_display == "Positive" else "🙁"
                    prob_pct = f"{bilstm_res.probability:.2%}"

                    st.markdown(
                        f"""
                        <div class="duel-card {card_class}">
                            <div class="verdict-header">
                                <span style="font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#94a3b8;">
                                    🧠 MÔ HÌNH HỌC SÂU (BiLSTM)
                                </span>
                                <span style="font-size:0.75rem; background:rgba(255,255,255,0.1); padding:3px 8px; border-radius:6px; color:#cbd5e1;">
                                    Calibrated (T = {getattr(bilstm_predictor, "temperature", 1.0):.4f})
                                </span>
                            </div>
                            <div class="verdict-badge" style="color: {badge_color};">
                                <span>{badge_emoji}</span>
                                <span>{bilstm_label_display.upper()}</span>
                            </div>
                            <div style="font-size: 1.25rem; font-weight: 700; margin: 8px 0 14px 0; color: #f8fafc;">
                                Xác suất hiệu chuẩn: <span style="color: {badge_color};">{prob_pct}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(float(bilstm_res.probability))
                    st.caption(
                        f"Ngưỡng phân định: `{custom_threshold:.2f}` • Trạng thái: "
                        f"{'Vượt ngưỡng (Positive)' if bilstm_is_pos else 'Dưới ngưỡng (Negative)'}"
                    )

                # Baseline Card
                with col_m2:
                    if baseline_res is not None:
                        base_card_class = (
                            "duel-baseline-pos"
                            if baseline_label_display == "Positive"
                            else "duel-baseline-neg"
                        )
                        base_color = (
                            "#38bdf8" if baseline_label_display == "Positive" else "#fbbf24"
                        )
                        base_emoji = "👍" if baseline_label_display == "Positive" else "👎"

                        st.markdown(
                            f"""
                            <div class="duel-card {base_card_class}">
                                <div class="verdict-header">
                                    <span style="font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#94a3b8;">
                                        📊 BASELINE (TF-IDF + LOGISTIC REGRESSION)
                                    </span>
                                    <span style="font-size:0.75rem; background:rgba(255,255,255,0.1); padding:3px 8px; border-radius:6px; color:#cbd5e1;">
                                        50k N-Grams
                                    </span>
                                </div>
                                <div class="verdict-badge" style="color: {base_color};">
                                    <span>{base_emoji}</span>
                                    <span>{baseline_label_display.upper()}</span>
                                </div>
                                <div style="font-size: 1.25rem; font-weight: 700; margin: 8px 0 14px 0; color: #f8fafc;">
                                    Xác suất mô hình: <span style="color: {base_color};">{baseline_res.probability:.2%}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.progress(float(baseline_res.probability))
                        st.caption("Dựa trên tổng trọng số tuyến tính các cụm từ unigram & bigram")
                    else:
                        st.info("Chưa tìm thấy checkpoint mô hình Baseline.")

                # KPI DIAGNOSTIC METRICS
                st.markdown("<br>", unsafe_allow_html=True)
                k1, k2, k3, k4 = st.columns(4)

                with k1:
                    st.markdown(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-label">Tổng Tokens</div>
                            <div class="kpi-value">{bilstm_res.token_count}</div>
                            <div class="kpi-sub">Sau tách từ regex</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with k2:
                    oov_color = "#ef4444" if bilstm_res.oov_rate > 0.20 else "#10b981"
                    st.markdown(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-label">Tỷ Lệ Từ Ngoài Từ Điển</div>
                            <div class="kpi-value" style="color:{oov_color};">{bilstm_res.oov_rate:.1%}</div>
                            <div class="kpi-sub">{"⚠️ Cảnh báo OOV cao" if bilstm_res.oov_rate > 0.20 else "✅ Độ phủ từ điển tốt"}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with k3:
                    trunc_str = "CÓ (256)" if bilstm_res.truncated else "KHÔNG"
                    trunc_color = "#f59e0b" if bilstm_res.truncated else "#10b981"
                    st.markdown(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-label">Cắt Ngắn (Head-Tail)</div>
                            <div class="kpi-value" style="color:{trunc_color};">{trunc_str}</div>
                            <div class="kpi-sub">Giới hạn max_length=256</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with k4:
                    w_count = len(bilstm_res.warnings)
                    w_color = "#10b981" if w_count == 0 else "#f59e0b"
                    st.markdown(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-label">Cảnh Báo Độ Tin Cậy</div>
                            <div class="kpi-value" style="color:{w_color};">{w_count}</div>
                            <div class="kpi-sub">{"Không phát hiện bất thường" if w_count == 0 else "Cần lưu ý phân tích"}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # DIAGNOSTIC WARNINGS PANEL
                if bilstm_res.warnings or bilstm_res.truncated:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("##### ⚠️ Nhật Ký Chẩn Đoán Dữ Liệu Đầu Vào (Input Diagnostics):")
                    if bilstm_res.truncated:
                        st.warning(
                            "• **Chuỗi văn bản bị cắt ngắn (Head-Tail Truncation):** Văn bản vượt quá 256 tokens. "
                            "Hệ thống đã giữ lại 128 tokens mở đầu và 128 tokens kết luận để bảo tồn thông điệp then chốt."
                        )
                    if bilstm_res.oov_rate > 0.20:
                        st.warning(
                            f"• **Mật độ từ ngoài từ điển cao ({bilstm_res.oov_rate:.1%}):** "
                            "Chứa nhiều thuật ngữ hiếm chưa gặp trong tập Train. Độ tin cậy xác suất có thể bị giảm."
                        )
                    if "OUT_OF_DOMAIN_LANGUAGE_HEURISTIC" in bilstm_res.warnings:
                        st.info(
                            "• **Phát hiện ngôn ngữ ngoài miền (Out of Domain):** Văn bản có dấu hiệu không hoàn toàn "
                            "là tiếng Anh chuẩn."
                        )

                # TOKEN AUDIT INSPECTOR
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander(
                    "🔍 Kiểm Toán Tách Từ & Phân Giải Từ Lạ (Token Inspection & UNK Highlighter)",
                    expanded=True,
                ):
                    st.write(
                        "Kiểm tra luồng xử lý văn bản sau khi chuẩn hóa ký tự, loại bỏ thẻ HTML và phân tích cú pháp. "
                        "Các từ có dấu viền màu đỏ kèm ký tự `*` là các token ngoài từ điển (`<UNK>`):"
                    )
                    tokens = tokenize(review_text)
                    vocab = getattr(bilstm_predictor, "vocabulary", None)
                    token_html = []
                    unk_count = 0
                    for t in tokens:
                        if vocab and t not in vocab.token_to_index:
                            unk_count += 1
                            chip = f'<span class="token-chip token-chip-unk" title="Ngoài từ điển Train">{t}*</span>'
                            token_html.append(chip)
                        else:
                            token_html.append(
                                f'<span class="token-chip token-chip-normal">{t}</span>'
                            )

                    st.markdown(
                        f"""
                        <div class="token-chip-box">
                            {"".join(token_html)}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Tổng số tokens: `{len(tokens)}` • Số token UNK: `{unk_count}` "
                        f"• Tỷ lệ: `{unk_count / len(tokens):.1%}`"
                        if tokens
                        else ""
                    )

            except Exception as exc:
                st.error(f"Đã xảy ra lỗi trong quá trình suy luận: {exc}")


# ==============================================================================
# TAB 2: SO SÁNH & HIỆU CHUẨN XÁC SUẤT
# ==============================================================================
with tab_compare:
    st.markdown("### ⚖️ Đối Chiếu Mô Hình & Phân Tích Hiệu Chuẩn Xác Suất")
    st.write(
        "Đối chiếu khoa học và thực nghiệm giữa Baseline truyền thống và Deep Learning BiLSTM, "
        "được kiểm toán trên tập Validation và tập Test chính thức độc lập."
    )

    # 4 KPI Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">BiLSTM Val Accuracy</div>
                <div class="kpi-value" style="color: #38bdf8;">88.38%</div>
                <div class="kpi-sub">Test Accuracy: <strong>87.89%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Baseline Val Accuracy</div>
                <div class="kpi-value" style="color: #818cf8;">89.58%</div>
                <div class="kpi-sub">TF-IDF + Logistic Regression</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">BiLSTM Brier Score</div>
                <div class="kpi-value" style="color: #34d399;">0.0883</div>
                <div class="kpi-sub">Sai số bình phương xác suất (Thấp = Tốt)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Calibrated ECE</div>
                <div class="kpi-value" style="color: #10b981;">0.0264</div>
                <div class="kpi-sub">Lệch xác suất chỉ ~2.6% qua 10 bins</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Head-to-Head Comparison Table
    st.markdown("#### 📋 Bảng So Sánh Toàn Diện (Head-to-Head Benchmark)")
    comparison_data = [
        {
            "Mô Hình": "TF-IDF + Logistic Regression (Baseline)",
            "Accuracy": "89.58%",
            "Macro-F1": "89.58%",
            "Brier Score": "0.0768",
            "ECE": "0.0276",
            "Tham Số": "100,001",
            "CPU Latency": "0.33 ms",
            "Đặc Tính Cốt Lõi": "Tốc độ cực nhanh, giải thích rõ qua trọng số từ vựng",
        },
        {
            "Mô Hình": "BiLSTM + Temperature Scaling",
            "Accuracy": "88.38%",
            "Macro-F1": "88.38%",
            "Brier Score": "0.0883",
            "ECE": "0.0264",
            "Tham Số": "658,049",
            "CPU Latency": "3.55 ms",
            "Đặc Tính Cốt Lõi": "Nắm bắt ngữ cảnh tuần tự, xác suất hiệu chuẩn đáng tin cậy",
        },
    ]
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    # Why Calibration Matters Callout
    st.markdown(
        """
        <div class="info-callout">
            <strong>💡 Tại sao cần Hiệu Chuẩn Xác Suất (Probability Calibration)?</strong><br>
            Trong môi trường sản xuất thực tế, phân loại chỉ đưa ra nhãn là chưa đủ; doanh nghiệp cần biết 
            <em>"Khi mô hình nói tự tin 90%, liệu xác suất đúng thực tế có đạt 90% không?"</em>.<br>
            Các mạng nơ-ron sâu thường bị <strong>Overconfident</strong> (quá tự tin nhưng sai). Bằng cách học hệ số 
            <strong>Temperature Scaling ($T > 0$)</strong> qua tối ưu L-BFGS trên tập Calibration độc lập, 
            CineSentiment AI đưa chỉ số sai lệch <strong>ECE xuống còn 2.6%</strong> mà không làm suy giảm Accuracy!
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### 📈 Bộ Đồ Thị Đánh Giá & Đường Cong Tin Cậy (Reliability Diagrams)")

    img_col1, img_col2 = st.columns(2)
    history_img = Path("artifacts/training_history.png")
    cm_img = Path("artifacts/confusion_matrix.png")
    rel_img = Path("artifacts/reliability_diagram.png")
    test_rel_img = Path("artifacts/test_reliability_diagram.png")

    with img_col1:
        if history_img.is_file():
            st.image(
                str(history_img),
                caption="Lịch sử Huấn luyện BiLSTM (Loss & Accuracy qua các Epochs)",
                use_container_width=True,
            )
        if rel_img.is_file():
            st.image(
                str(rel_img),
                caption="Reliability Diagram trên tập Validation (Đường cong Calibration)",
                use_container_width=True,
            )

    with img_col2:
        if cm_img.is_file():
            st.image(
                str(cm_img),
                caption="Ma trận nhầm lẫn (Confusion Matrix trên tập Validation)",
                use_container_width=True,
            )
        if test_rel_img.is_file():
            st.image(
                str(test_rel_img),
                caption="Reliability Diagram trên tập Test chính thức độc lập",
                use_container_width=True,
            )


# ==============================================================================
# TAB 3: PHÂN TÍCH LÁT CẮT LỖI NGÔN NGỮ HỌC
# ==============================================================================
with tab_errors:
    st.markdown("### 🔍 Phân Tích Lát Cắt Lỗi Ngôn Ngữ Học (Linguistic Slices)")
    st.write(
        "Mô hình học máy trong môi trường thực tế cần được kiểm toán các lát cắt ngôn ngữ học "
        "dễ gây sai lệch (Linguistic Failure Modes) để chuẩn bị cho rủi ro thực tế."
    )

    if error_data:
        err_c1, err_c2, err_c3 = st.columns(3)
        with err_c1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Tổng Mẫu Phân Tích</div>
                    <div class="kpi-value">{error_data.get("total_samples", 0):,}</div>
                    <div class="kpi-sub">Tập kiểm toán độc lập</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with err_c2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Tổng Số Lỗi Phán Đoán</div>
                    <div class="kpi-value" style="color:#ef4444;">
                        {error_data.get("total_errors", 0):,}
                    </div>
                    <div class="kpi-sub">Số mẫu dự đoán sai nhãn</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with err_c3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Tỷ Lệ Lỗi Toàn Cục</div>
                    <div class="kpi-value">{error_data.get("overall_error_rate", 0.0):.2%}</div>
                    <div class="kpi-sub">Baseline Error Rate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        slice_col1, slice_col2 = st.columns(2)

        with slice_col1:
            st.markdown("#### 🗣️ Lát Cắt Ngôn Ngữ Học (Linguistic Slices)")
            ling_rows = []
            for _k, v in error_data.get("linguistic_slices", {}).items():
                ling_rows.append(
                    {
                        "Lát Cắt": v["name"],
                        "Số Mẫu": f"{v['total_count']:,}",
                        "Số Lỗi": v["error_count"],
                        "Tỷ Lệ Lỗi": f"{v['error_rate']:.2%}",
                    }
                )
            st.dataframe(pd.DataFrame(ling_rows), use_container_width=True, hide_index=True)

            st.markdown("#### 📏 Lát Cắt Theo Độ Dài Câu (Length Slices)")
            len_rows = []
            for k, v in error_data.get("length_slices", {}).items():
                len_rows.append(
                    {
                        "Độ Dài Câu": k,
                        "Tổng Mẫu": f"{v['total']:,}",
                        "Số Lỗi": v["errors"],
                        "Tỷ Lệ Lỗi": f"{v['error_rate']:.2%}",
                    }
                )
            st.dataframe(pd.DataFrame(len_rows), use_container_width=True, hide_index=True)

        with slice_col2:
            st.markdown("#### 🧩 Lát Cắt Từ Ngoài Từ Điển (OOV Slices)")
            oov_rows = []
            for k, v in error_data.get("oov_slices", {}).items():
                oov_rows.append(
                    {
                        "Mức Độ OOV": k,
                        "Tổng Mẫu": f"{v['total']:,}",
                        "Số Lỗi": v["errors"],
                        "Tỷ Lệ Lỗi": f"{v['error_rate']:.2%}",
                    }
                )
            st.dataframe(pd.DataFrame(oov_rows), use_container_width=True, hide_index=True)

            st.markdown("#### 🎯 Top Mẫu Tự Tin Sai (High-Confidence Errors)")
            high_conf = error_data.get("high_confidence_examples", [])
            for idx, ex in enumerate(high_conf[:3], 1):
                lbl_str = "Positive" if ex["label"] == 1 else "Negative"
                pred_str = "Positive" if ex["predicted"] == 1 else "Negative"
                lbl_color = "#34d399" if ex["label"] == 1 else "#f87171"
                pred_color = "#34d399" if ex["predicted"] == 1 else "#f87171"

                st.markdown(
                    f"""
                    <div class="filmstrip-card">
                        <div style="font-size:0.85rem; font-weight:700; margin-bottom: 4px;">
                            <span style="color:#ef4444;">#{idx}</span> • 
                            Nhãn Thực: <strong style="color:{lbl_color};">{lbl_str}</strong> | 
                            Dự Đoán: <strong style="color:{pred_color};">{pred_str}</strong> | 
                            Độ Tin Cậy: <strong style="color:#f59e0b;">{ex["confidence"]:.1%}</strong>
                        </div>
                        <div style="font-size:0.88rem; color:#cbd5e1; font-style:italic;">
                            "{ex["text_snippet"]}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("Chưa tìm thấy báo cáo phân tích lỗi `artifacts/error_analysis.json`.")

    # Lexical Explainability from Baseline
    if explain_data:
        st.markdown("---")
        st.markdown("#### 💡 Tín Hiệu Từ Vựng Quan Trọng (Lexical Signals từ TF-IDF)")
        st.write(
            "Các từ và cụm từ (n-grams) đóng góp trọng số lớn nhất trong việc phân định cảm xúc của Baseline:"
        )
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("##### 🟢 Top 10 N-grams Tích Cực:")
            pos_df = pd.DataFrame(explain_data.get("top_positive_ngrams", [])[:10])
            if not pos_df.empty:
                st.dataframe(pos_df, use_container_width=True, hide_index=True)
        with exp_col2:
            st.markdown("##### 🔴 Top 10 N-grams Tiêu Cực:")
            neg_df = pd.DataFrame(explain_data.get("top_negative_ngrams", [])[:10])
            if not neg_df.empty:
                st.dataframe(neg_df, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 4: BATCH REVIEW TESTER
# ==============================================================================
with tab_batch:
    st.markdown("### 📦 Kiểm Thử Đánh Giá Theo Lô (Batch Inference)")
    st.write(
        "Nhập danh sách nhiều câu đánh giá cùng lúc hoặc tải lên tệp CSV/TXT chứa cột `text` "
        "để mô hình BiLSTM phân tích hàng loạt với tốc độ tối ưu."
    )

    batch_mode = st.radio(
        "Phương thức nhập dữ liệu:",
        ["Nhập văn bản từng dòng", "Tải lên tệp CSV/TXT"],
        horizontal=True,
    )
    batch_texts: list[str] = []

    if batch_mode == "Nhập văn bản từng dòng":
        default_batch = (
            "An incredible masterpiece with magnificent soundtrack and acting.\n"
            "Utterly boring and a complete waste of two hours.\n"
            "Decent effects but the script makes very little sense.\n"
            "I genuinely enjoyed the witty humor and unexpected twist ending.\n"
            "Worst film of the decade, totally unwatchable."
        )
        batch_raw = st.text_area(
            "Danh sách các câu đánh giá (mỗi dòng là một bài review riêng biệt):",
            value=default_batch,
            height=140,
        )
        batch_texts = [line.strip() for line in batch_raw.splitlines() if line.strip()]
    else:
        uploaded_file = st.file_uploader(
            "Tải lên tệp dữ liệu CSV hoặc TXT (nếu CSV cần có cột 'text'):",
            type=["csv", "txt"],
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_up = pd.read_csv(uploaded_file)
                    if "text" in df_up.columns:
                        batch_texts = df_up["text"].dropna().astype(str).tolist()
                        st.success(f"✅ Đã nạp thành công {len(batch_texts):,} câu từ tệp CSV.")
                    else:
                        st.error("⚠️ Tệp CSV cần có cột tên là 'text'.")
                else:
                    content = uploaded_file.read().decode("utf-8", errors="replace")
                    batch_texts = [line.strip() for line in content.splitlines() if line.strip()]
                    st.success(f"✅ Đã nạp thành công {len(batch_texts):,} dòng từ tệp TXT.")
            except Exception as e:
                st.error(f"Lỗi khi đọc file: {e}")

    if st.button("⚡ Chạy Dự Đoán Toàn Bộ Lô", type="primary", use_container_width=True):
        if not batch_texts:
            st.warning("⚠️ Vui lòng cung cấp danh sách dữ liệu trước khi chạy.")
        elif bilstm_predictor is None:
            st.error("⚠️ Chưa nạp được checkpoint mô hình BiLSTM.")
        else:
            with st.spinner(f"Đang phân tích {len(batch_texts):,} đánh giá..."):
                results = bilstm_predictor.predict_batch(batch_texts)

                pos_count = sum(1 for r in results if r.probability >= custom_threshold)
                neg_count = len(results) - pos_count
                pos_pct = pos_count / len(results) if results else 0.0

                st.markdown("<br>", unsafe_allow_html=True)
                b_c1, b_c2, b_c3 = st.columns(3)
                with b_c1:
                    st.metric("Tổng Số Mẫu", f"{len(results):,}")
                with b_c2:
                    st.metric(
                        "Tỷ Lệ Tích Cực (Positive)",
                        f"{pos_pct:.1%}",
                        delta=f"{pos_count} câu",
                    )
                with b_c3:
                    st.metric(
                        "Tỷ Lệ Tiêu Cực (Negative)",
                        f"{1 - pos_pct:.1%}",
                        delta=f"-{neg_count} câu",
                        delta_color="inverse",
                    )

                result_records = []
                for t, r in zip(batch_texts, results, strict=False):
                    is_pos = r.probability >= custom_threshold
                    pred_label = "Positive" if is_pos else "Negative"
                    result_records.append(
                        {
                            "Review Text": t[:140] + ("..." if len(t) > 140 else ""),
                            "Prediction": pred_label,
                            "Calibrated Prob": f"{r.probability:.2%}",
                            "Tokens": r.token_count,
                            "OOV Rate": f"{r.oov_rate:.1%}",
                            "Truncated": "Yes" if r.truncated else "No",
                        }
                    )
                res_df = pd.DataFrame(result_records)
                st.dataframe(res_df, use_container_width=True, hide_index=True)

                csv_buffer = io.StringIO()
                res_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Tải Kết Quả Dưới Dạng CSV",
                    data=csv_buffer.getvalue(),
                    file_name="cinesentiment_batch_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ==============================================================================
# TAB 5: KIẾN TRÚC HỆ THỐNG & REST API
# ==============================================================================
with tab_arch:
    st.markdown("### 📐 Kiến Trúc Hệ Thống & Tích Hợp REST API")
    st.write(
        "CineSentiment AI được thiết kế theo tiêu chuẩn kỹ thuật Machine Learning hiện đại, "
        "tách biệt rõ ràng giữa các giai đoạn: Tiền xử lý, Huấn luyện, Hiệu chuẩn và Phục vụ suy luận."
    )

    # Pipeline Infographic Box
    st.markdown(
        """
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 16px; padding: 24px; margin-bottom: 24px;">
            <h4 style="margin-top: 0; color: #818cf8;">🛠️ Quy Trình Xử Lý Dữ Liệu & Mô Hình Hóa</h4>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #cbd5e1; line-height: 1.8;">
                [1. Raw Movie Review] ➔ Loại bỏ thẻ HTML, chuẩn hóa chữ thường<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
                [2. Tokenizer Regex] ➔ Bảo tồn từ phủ định then chốt ("don't", "didn't", "isn't", "can't")<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
                [3. Head-Tail Truncation] ➔ Giữ 128 tokens đầu + 128 tokens cuối nếu độ dài > 256 tokens<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
                [4. Embedding Layer] ➔ Biểu diễn vector 128 chiều, padding_idx không cập nhật gradient<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
                [5. pack_padded_sequence] ➔ Bỏ qua bước lặp vô ích trên padding, tăng tốc tính toán RNN<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
                [6. 2-Layer BiLSTM] ➔ Hidden 128 chiều, trích xuất đặc trưng tuần tự 2 chiều (Forward + Backward)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
                [7. Feature Concat + Linear] ➔ Vector 256d ➔ Dropout(0.4) ➔ Linear(256, 1) ➔ Logit z<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
                [8. Temperature Scaling] ➔ Hiệu chuẩn z / T (với T học từ Calibration Split qua L-BFGS)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
                [9. Calibrated Probability] ➔ p = Sigmoid(z / T) ➔ Phán định nhãn theo Decision Threshold
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🔌 Hướng Dẫn Tích Hợp FastAPI REST API")
    st.write("Khởi chạy REST API độc lập trong môi trường production:")
    st.code("python api.py  # Hoặc: uvicorn api:app --host 0.0.0.0 --port 8000", language="bash")

    api_code_python = """import requests

url = "http://localhost:8000/predict"
payload = {
    "text": "This movie was absolutely captivating with stunning visuals and great acting!"
}
response = requests.post(url, json=payload)
data = response.json()

print(f"Nhãn cảm xúc: {data['label']}")
print(f"Xác suất hiệu chuẩn: {data['probability']:.2%}")
print(f"Số token: {data['token_count']} | OOV rate: {data['oov_rate']:.1%}")
"""

    api_code_curl = """curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "The cinematography was great, but the plot was completely boring."}'
"""

    code_tab1, code_tab2 = st.tabs(["Python Code (requests)", "cURL Command (Bash)"])
    with code_tab1:
        st.code(api_code_python, language="python")
    with code_tab2:
        st.code(api_code_curl, language="bash")

    st.markdown(
        """
        <div class="info-callout">
            <strong>🚀 Swagger UI Documentation:</strong><br>
            Khi API đang chạy, bạn có thể truy cập tài liệu Swagger UI tương tác trực tiếp tại:
            <code>http://localhost:8000/docs</code> hoặc Redoc tại <code>http://localhost:8000/redoc</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )
