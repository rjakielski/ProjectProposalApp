import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from classify_tmpc import THRESHOLD, classify_dataframe, classify_project

SEARCH_SRC = Path(__file__).resolve().parents[1] / "Oversight Semantic Search" / "src"
if SEARCH_SRC.exists() and str(SEARCH_SRC) not in sys.path:
    sys.path.insert(0, str(SEARCH_SRC))

try:
    from oversight_semantic_search.index import SemanticSearchIndex
except ImportError:
    SemanticSearchIndex = None

st.set_page_config(
    page_title="TMPC Classifier",
    page_icon="📋",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 22%),
          linear-gradient(135deg, #f7f2ea, #ece5d8);
      }
      .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
      }
      h1, h2, h3 {
        font-family: Georgia, "Times New Roman", serif;
        color: #1e2b2b;
      }
      .hero {
        background: rgba(255, 253, 248, 0.92);
        border: 1px solid #d8cdc0;
        border-radius: 24px;
        padding: 1.5rem 1.6rem;
        box-shadow: 0 18px 50px rgba(30, 43, 43, 0.10);
        margin-bottom: 1.2rem;
      }
      .result-card {
        background: rgba(255, 253, 248, 0.96);
        border: 1px solid #d8cdc0;
        border-radius: 20px;
        padding: 1.2rem;
        margin-top: 0.8rem;
      }
      .match-card {
        background: rgba(250, 248, 242, 0.96);
        border: 1px solid #d8cdc0;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-top: 0.8rem;
      }
      .result-pill {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(15, 118, 110, 0.10);
        color: #115e59;
        font-weight: 700;
        margin-bottom: 0.9rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_semantic_index():
    if SemanticSearchIndex is None:
        return None
    index = SemanticSearchIndex()
    index.ensure_ready()
    return index

st.markdown(
    f"""
    <div class="hero">
      <h1>TMPC Project Classifier</h1>
      <p style="margin-bottom:0;color:#61706f;font-size:1.02rem;">
        Score a single audit project in the browser or upload a CSV for batch classification.
        Predictions below {THRESHOLD:.0%} confidence are labeled <strong>Needs review</strong>.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

single_tab, batch_tab = st.tabs(["Single project", "CSV upload"])

with single_tab:
    with st.form("single_project_form"):
        project_name = st.text_input("Project name")
        project_objective = st.text_area("Project objective", height=180)
        submitted = st.form_submit_button("Classify project", use_container_width=True)

    if submitted:
        if not project_name.strip() or not project_objective.strip():
            st.error("Enter both a project name and a project objective.")
        else:
            try:
                result = classify_project(project_name, project_objective)
                st.markdown(
                    f"""
                    <div class="result-card">
                      <div class="result-pill">{result["predicted_TMPC"]}</div>
                      <h3>Suggested TMPC classification</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Prediction confidence", f'{result["prediction_confidence"] * 100:.1f}%')
                col2.metric("Top class", result["top1_class"])
                col3.metric("Runner-up class", result["top2_class"])
                col4.metric("Runner-up confidence", f'{result["top2_confidence"] * 100:.1f}%')

                st.markdown("### Similar OIG reports")
                index = get_semantic_index()
                if index is None:
                    st.info("Semantic search package not found. Expected sibling folder: Oversight Semantic Search.")
                else:
                    matches = index.search_project(project_name, project_objective, top_k=10)
                    if not matches:
                        st.info("No similar reports found for this project.")
                    for rank, match in enumerate(matches, start=1):
                        summary = (match.get("summary") or "No summary available.")[:380]
                        st.markdown(
                            f"""
                            <div class="match-card">
                              <div class="result-pill">#{rank} | Similarity {match["score"]:.3f}</div>
                              <h3 style="margin-bottom:0.35rem;">{match["title"]}</h3>
                              <p style="margin:0.25rem 0;color:#61706f;">
                                <strong>Published:</strong> {match.get("publication_date") or "Unknown"}<br>
                                <strong>Agency:</strong> {match.get("agency") or "Unknown"}<br>
                                <strong>Type:</strong> {match.get("report_type") or "Unknown"}
                              </p>
                              <p style="margin:0.55rem 0;color:#435255;">{summary}</p>
                              <p style="margin:0;">
                                <a href="{match["detail_url"]}" target="_blank">Open report</a>
                              </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            except Exception as exc:
                st.error(f"Classification failed: {exc}")

with batch_tab:
    st.write("Upload a CSV with a `name_and_objective` column for batch scoring.")
    uploaded_file = st.file_uploader("CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file, encoding="latin-1")
            st.write("Preview")
            st.dataframe(input_df.head(10), use_container_width=True)

            if st.button("Classify CSV", use_container_width=True):
                result_df = classify_dataframe(input_df)
                st.success("Classification complete.")
                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download classified CSV",
                    data=BytesIO(csv_bytes),
                    file_name="tmpc_classification_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as exc:
            st.error(f"Could not process the CSV: {exc}")
