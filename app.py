import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googleapiclient.discovery import build
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# --------------------------------------------------
# PAGE + DICTIONARY
# --------------------------------------------------
st.set_page_config(page_title="SlangSense | Lost in Slang", page_icon="💬", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #101113; }
      [data-testid="stHeader"] { background: rgba(16,17,19,0.92); }
      h1, h2, h3 { color: #f7f7f7; }
      p, label, [data-testid="stCaptionContainer"] { color: #d7d7d7; }
      div.stButton > button[kind="primary"] {
        background: #e62117; color: white; border: 1px solid #e62117;
        border-radius: 10px; padding: 0.62rem 1.1rem; font-weight: 650;
      }
      div.stButton > button[kind="primary"]:hover {
        background: #c91d15; border-color: #c91d15; color: white;
      }
      [data-testid="stMetric"] {
        background: #191a1e; border: 1px solid #303136; border-radius: 12px; padding: 14px;
      }
    </style>
    """, unsafe_allow_html=True,
)
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:14px;margin:0.2rem 0 0.6rem 0;">
      <div style="width:54px;height:54px;border-radius:16px;background:#e62117;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <svg width="34" height="34" viewBox="0 0 48 48" aria-hidden="true">
          <path d="M8 11 Q8 5 15 5 H34 Q41 5 41 12 V27 Q41 34 34 34 H23 L13 42 V34 Q8 32 8 26 Z" fill="white"/>
          <circle cx="18" cy="20" r="2.5" fill="#e62117"/><circle cx="25" cy="20" r="2.5" fill="#e62117"/><circle cx="32" cy="20" r="2.5" fill="#e62117"/>
        </svg>
      </div>
      <div><div style="font-size:2.2rem;font-weight:800;line-height:1.1;color:#f7f7f7;">SlangSense</div>
      <div style="font-size:0.95rem;color:#a6a6a6;margin-top:4px;">Lost in Slang</div></div>
    </div>
    """, unsafe_allow_html=True,
)
st.subheader("Is social listening misreading YouTube audiences?")
st.write(
    "Compare VADER and RoBERTa on the same comments, investigate slang, "
    "and explore how slang may affect sentiment predictions."
)

DICTIONARY_PATH = Path(__file__).parent / "slang_dictionary.csv"


@st.cache_data
def load_slang_dictionary():
    dictionary = pd.read_csv(DICTIONARY_PATH)
    required = {"term", "meaning", "sentiment_hint", "category"}
    missing = required - set(dictionary.columns)
    if missing:
        raise ValueError("Dictionary missing columns: " + ", ".join(sorted(missing)))
    dictionary["term"] = dictionary["term"].fillna("").astype(str).str.lower().str.strip()
    dictionary = dictionary[dictionary["term"] != ""]
    # Ignore single-letter alphabetic entries (for example, L and W), but keep emoji.
    dictionary = dictionary[~dictionary["term"].str.fullmatch(r"[a-z]", na=False)]
    dictionary = dictionary.drop_duplicates("term")
    return dictionary


try:
    slang_dictionary = load_slang_dictionary()
except Exception as exc:
    st.error(f"Could not load slang_dictionary.csv: {exc}")
    st.stop()


# --------------------------------------------------
# MODELS
# --------------------------------------------------
@st.cache_resource
def get_vader():
    return SentimentIntensityAnalyzer()


@st.cache_resource
def get_roberta():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )


def normalize_label(label):
    value = str(label).strip().lower()
    mapping = {
        "label_0": "negative", "label_1": "neutral", "label_2": "positive",
        "negative": "negative", "neutral": "neutral", "positive": "positive",
    }
    return mapping.get(value, value)


def prepare_comments(raw):
    df = raw.copy()
    candidates = ["comment_text", "text", "comment", "content", "Comment", "Text", "review", "Review"]
    source_col = next((col for col in candidates if col in df.columns), None)
    if source_col is None:
        st.error("Could not identify the text column. Expected comment_text, text, comment, content, or review.")
        st.write("Available columns:", list(df.columns))
        st.stop()
    if source_col != "comment_text":
        df = df.rename(columns={source_col: "comment_text"})
    df["comment_text"] = df["comment_text"].fillna("").astype(str)
    df = df[df["comment_text"].str.strip() != ""].copy().reset_index(drop=True)
    return df


def run_both_models(df):
    """Run both models on exactly the same rows and preserve original predictions."""
    df = df.copy()
    analyzer = get_vader()
    df["vader_compound"] = df["comment_text"].apply(
        lambda text: analyzer.polarity_scores(str(text))["compound"]
    )
    df["vader_sentiment"] = df["vader_compound"].apply(
        lambda score: "positive" if score >= 0.05 else ("negative" if score <= -0.05 else "neutral")
    )

    model = get_roberta()
    texts = df["comment_text"].fillna("").astype(str).tolist()
    results = []
    batch_size = 16
    progress = st.progress(0, text="Running RoBERTa on the same comments...")
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        results.extend(model(batch, truncation=True, max_length=512))
        progress.progress(min(1.0, len(results) / max(1, len(texts))), text=f"RoBERTa: {len(results):,}/{len(texts):,} comments")
    progress.empty()
    df["roberta_sentiment"] = [normalize_label(item["label"]) for item in results]
    df["roberta_confidence"] = [item["score"] for item in results]
    return df


# --------------------------------------------------
# DICTIONARY MATCHING + EXPLICIT HEURISTIC ADJUSTMENT
# --------------------------------------------------
def add_slang_analysis(df):
    df = df.copy()
    text = df["comment_text"].fillna("").astype(str)
    dictionary = slang_dictionary.copy()
    terms = sorted(dictionary["term"].tolist(), key=len, reverse=True)
    meaning_map = dictionary.set_index("term")["meaning"].to_dict()
    hint_map = dictionary.set_index("term")["sentiment_hint"].to_dict()
    category_map = dictionary.set_index("term")["category"].to_dict()

    def find_slang(comment):
        found = []
        lower = comment.lower()
        for term in terms:
            # Match emoji literally; use word boundaries for ordinary terms.
            if re.search(r"\w", term):
                pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
            else:
                pattern = re.escape(term)
            if re.search(pattern, lower):
                found.append(term)
        return found

    df["detected_slang"] = text.apply(find_slang)
    df["possible_slang"] = df["detected_slang"].apply(bool)
    df["slang_meanings"] = df["detected_slang"].apply(
        lambda terms_found: "; ".join(f"{term}: {meaning_map.get(term, '')}" for term in terms_found)
    )
    df["slang_sentiment_hints"] = df["detected_slang"].apply(
        lambda terms_found: "; ".join(f"{term}: {hint_map.get(term, 'context-dependent')}" for term in terms_found)
    )
    df["slang_categories"] = df["detected_slang"].apply(
        lambda terms_found: "; ".join(f"{term}: {category_map.get(term, '')}" for term in terms_found)
    )
    df["has_emoji_or_symbol"] = text.apply(lambda value: bool(re.search(r"[^\w\s.,!?']", value)))
    df["repeated_punctuation"] = text.str.contains(r"[!?.,]{2,}", regex=True, na=False)

    def adjust(original, terms_found):
        # Only use explicit single-label hints. Ambiguous terms are not forced into a sentiment.
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        for term in terms_found:
            hint = str(hint_map.get(term, "")).strip().lower()
            if hint in counts:
                counts[hint] += 1
        highest = max(counts.values())
        if highest == 0:
            return original
        winners = [label for label, count in counts.items() if count == highest]
        return winners[0] if len(winners) == 1 else original

    if "vader_sentiment" in df.columns:
        df["vader_sentiment_with_slang"] = df.apply(lambda row: adjust(row["vader_sentiment"], row["detected_slang"]), axis=1)
        df["vader_changed_with_slang"] = df["vader_sentiment"] != df["vader_sentiment_with_slang"]
    if "roberta_sentiment" in df.columns:
        df["roberta_sentiment_with_slang"] = df.apply(lambda row: adjust(row["roberta_sentiment"], row["detected_slang"]), axis=1)
        df["roberta_changed_with_slang"] = df["roberta_sentiment"] != df["roberta_sentiment_with_slang"]
    if {"vader_sentiment", "roberta_sentiment"}.issubset(df.columns):
        df["model_disagreement"] = df["vader_sentiment"] != df["roberta_sentiment"]
    if {"vader_sentiment_with_slang", "roberta_sentiment_with_slang"}.issubset(df.columns):
        df["model_disagreement_with_slang"] = df["vader_sentiment_with_slang"] != df["roberta_sentiment_with_slang"]
    return df


# --------------------------------------------------
# EVALUATION AGAINST HUMAN-REVIEWED LABELS
# --------------------------------------------------
def normalize_reference_label(value):
    value = str(value).strip().lower()
    aliases = {"pos": "positive", "neg": "negative", "neu": "neutral"}
    value = aliases.get(value, value)
    return value if value in {"positive", "neutral", "negative"} else None


def evaluate_predictions(df):
    if "sentiment_label" in df.columns:
        label_col = "sentiment_label"
    elif "final_label" in df.columns:
        label_col = "final_label"
    else:
        return

    scored = df.copy()
    scored["reference_label_normalized"] = scored[label_col].apply(normalize_reference_label)
    valid = scored["reference_label_normalized"].notna()
    excluded = int((~valid).sum())
    scored = scored[valid].copy()
    if scored.empty:
        st.warning("No positive, neutral, or negative reference labels were found. Check the label values in your CSV.")
        return

    st.subheader("8. Evaluation against reference labels")
    st.caption(f"Scored {len(scored):,} reviews with positive/neutral/negative labels. Excluded {excluded:,} rows with uncertain, missing, or unsupported labels.")
    prediction_columns = [
        ("VADER — before", "vader_sentiment"),
        ("RoBERTa — before", "roberta_sentiment"),
        ("VADER — after slang", "vader_sentiment_with_slang"),
        ("RoBERTa — after slang", "roberta_sentiment_with_slang"),
    ]
    rows = []
    for name, col in prediction_columns:
        if col not in scored.columns:
            continue
        rows.append({
            "Version": name,
            "Accuracy": accuracy_score(scored["reference_label_normalized"], scored[col]),
            "Macro-F1": f1_score(scored["reference_label_normalized"], scored[col], labels=["negative", "neutral", "positive"], average="macro", zero_division=0),
        })
    if not rows:
        st.info("Run both models first to calculate evaluation metrics.")
        return
    metrics = pd.DataFrame(rows)
    st.dataframe(
        metrics.style.format({"Accuracy": "{:.1%}", "Macro-F1": "{:.3f}"}),
        hide_index=True, use_container_width=True,
    )
    metric_long = metrics.melt(id_vars="Version", var_name="Metric", value_name="Score")
    chart = alt.Chart(metric_long).mark_bar().encode(
        x=alt.X("Version:N", title=None, axis=alt.Axis(labelAngle=-25)),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
        color="Metric:N", xOffset="Metric:N",
        tooltip=["Version", alt.Tooltip("Score:Q", format=".3f")],
    ).properties(height=350)
    st.altair_chart(chart, use_container_width=True)

    st.markdown("**Confusion matrices**")
    model_cols = st.columns(2)
    for box, (name, col) in zip(model_cols, [("VADER — before", "vader_sentiment"), ("RoBERTa — before", "roberta_sentiment")]):
        if col not in scored.columns:
            continue
        with box:
            st.write(name)
            matrix = confusion_matrix(scored["reference_label_normalized"], scored[col], labels=["negative", "neutral", "positive"])
            st.dataframe(pd.DataFrame(matrix, index=["Actual negative", "Actual neutral", "Actual positive"], columns=["Pred negative", "Pred neutral", "Pred positive"]), use_container_width=True)

    if "notes" in scored.columns:
        with st.expander("Review notes for labeled examples"):
            st.dataframe(scored[[c for c in ["comment_text", label_col, "notes"] if c in scored.columns]], use_container_width=True, hide_index=True)


def show_dashboard(df):
    df = df.copy()
    total = len(df)
    st.subheader("1. Audience sentiment overview")
    st.metric("Comments analyzed", f"{total:,}")

    # Three graph views: original model comparison, adjusted comparison, all four versions.
    st.subheader("2. Compare VADER and RoBERTa")
    graph_choice = st.selectbox(
        "Choose comparison graph",
        ["Graph A — Original predictions", "Graph B — Slang-adjusted predictions", "Graph C — Before vs after (all four versions)"],
        key="comparison_graph",
    )
    versions = {
        "Graph A — Original predictions": [("VADER", "vader_sentiment"), ("RoBERTa", "roberta_sentiment")],
        "Graph B — Slang-adjusted predictions": [("VADER + slang", "vader_sentiment_with_slang"), ("RoBERTa + slang", "roberta_sentiment_with_slang")],
        "Graph C — Before vs after (all four versions)": [("VADER before", "vader_sentiment"), ("VADER after", "vader_sentiment_with_slang"), ("RoBERTa before", "roberta_sentiment"), ("RoBERTa after", "roberta_sentiment_with_slang")],
    }[graph_choice]
    graph_rows = []
    for version_name, col in versions:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        for sentiment in ["positive", "neutral", "negative"]:
            graph_rows.append({"Version": version_name, "Sentiment": sentiment, "Comments": int(counts.get(sentiment, 0))})
    if graph_rows:
        graph_df = pd.DataFrame(graph_rows)
        chart = alt.Chart(graph_df).mark_bar().encode(
            x=alt.X("Sentiment:N", sort=["positive", "neutral", "negative"]),
            y=alt.Y("Comments:Q", title="Number of comments"),
            color=alt.Color("Version:N", title="Model/version"),
            xOffset="Version:N",
            tooltip=["Version", "Sentiment", "Comments"],
        ).properties(height=380)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Run both models to display comparison graphs.")

    st.subheader("3. What changed after slang adjustment?")
    metric_boxes = st.columns(2)
    for box, name, before_col, after_col in [
        (metric_boxes[0], "VADER", "vader_sentiment", "vader_sentiment_with_slang"),
        (metric_boxes[1], "RoBERTa", "roberta_sentiment", "roberta_sentiment_with_slang"),
    ]:
        if before_col in df.columns and after_col in df.columns:
            changed = int((df[before_col] != df[after_col]).sum())
            with box:
                st.markdown(f"**{name}**")
                st.metric("Predictions changed", f"{changed:,}")
                st.metric("Share changed", f"{changed / total:.1%}" if total else "0.0%")
                distribution = pd.DataFrame({"Before": df[before_col].value_counts(), "After": df[after_col].value_counts()}).reindex(["positive", "neutral", "negative"]).fillna(0).astype(int)
                st.dataframe(distribution, use_container_width=True)

    if {"model_disagreement", "model_disagreement_with_slang"}.issubset(df.columns):
        before = int(df["model_disagreement"].sum())
        after = int(df["model_disagreement_with_slang"].sum())
        left, right = st.columns(2)
        left.metric("Model disagreements before", f"{before:,} ({before / total:.1%})" if total else "0")
        right.metric("Model disagreements after", f"{after:,} ({after / total:.1%})" if total else "0")
        st.caption("Agreement is not accuracy; human-reviewed labels are needed to assess correctness.")

    st.subheader("4. Investigate slang impact")
    slang_comments = df[df["possible_slang"]].copy() if "possible_slang" in df.columns else pd.DataFrame()
    st.metric("Comments with dictionary terms", f"{len(slang_comments):,}")
    if not slang_comments.empty:
        cols = ["comment_text", "detected_slang", "slang_meanings", "slang_sentiment_hints"]
        cols += [c for c in ["vader_sentiment", "vader_sentiment_with_slang", "roberta_sentiment", "roberta_sentiment_with_slang"] if c in df.columns]
        st.dataframe(slang_comments[cols], use_container_width=True, hide_index=True)

        # Count how often each slang term occurs and whether each model changed on those rows.
        impact_rows = []
        for _, row in slang_comments.iterrows():
            for term in row["detected_slang"]:
                item = {"Slang term": term}
                if "vader_changed_with_slang" in df.columns:
                    item["VADER changed"] = bool(row["vader_changed_with_slang"])
                if "roberta_changed_with_slang" in df.columns:
                    item["RoBERTa changed"] = bool(row["roberta_changed_with_slang"])
                impact_rows.append(item)
        if impact_rows:
            impact = pd.DataFrame(impact_rows).groupby("Slang term").agg(
                Occurrences=("Slang term", "size"),
                **({"VADER changes": ("VADER changed", "sum")} if "vader_changed_with_slang" in df.columns else {}),
                **({"RoBERTa changes": ("RoBERTa changed", "sum")} if "roberta_changed_with_slang" in df.columns else {}),
            ).reset_index().sort_values("Occurrences", ascending=False)
            st.markdown("**Which slang terms are associated with changed predictions?**")
            st.dataframe(impact, use_container_width=True, hide_index=True)
    else:
        st.info("No dictionary terms were detected in this dataset.")

    st.subheader("5. Informal-language cues")
    flags = [c for c in ["possible_slang", "has_emoji_or_symbol", "repeated_punctuation"] if c in df.columns]
    if flags:
        st.dataframe(df[flags].fillna(False).astype(bool).sum().rename("Comments flagged").to_frame(), use_container_width=True)

    evaluate_predictions(df)

    st.subheader("9. Inspect and download comments")
    filters = ["All comments"]
    if "possible_slang" in df.columns:
        filters.append("Comments with slang")
    if "model_disagreement" in df.columns:
        filters.append("Model disagreements before adjustment")
    if "vader_changed_with_slang" in df.columns:
        filters.append("Comments where VADER changed")
    if "roberta_changed_with_slang" in df.columns:
        filters.append("Comments where RoBERTa changed")
    chosen = st.selectbox("Filter comments", filters, key="comment_filter")
    view = df.copy()
    if chosen == "Comments with slang":
        view = view[view["possible_slang"]]
    elif chosen == "Model disagreements before adjustment":
        view = view[view["model_disagreement"]]
    elif chosen == "Comments where VADER changed":
        view = view[view["vader_changed_with_slang"]]
    elif chosen == "Comments where RoBERTa changed":
        view = view[view["roberta_changed_with_slang"]]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download full analyzed dataset as CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="lost_in_slang_analysis.csv", mime="text/csv")


# --------------------------------------------------
# INPUT MODES: CSV EVALUATION OR YOUTUBE
# --------------------------------------------------
mode = st.radio("Choose an analysis mode", ["Analyze existing CSV", "Analyze a YouTube video"], horizontal=True)

if mode == "Analyze existing CSV":
    # The default dataset can be replaced with a larger file using the same name.
    DEFAULT_DATASET_PATH = Path(__file__).parent / "reviews_dataset.csv"
    uploaded = st.file_uploader(
        "Optional: upload a different comments/reviews CSV",
        type=["csv"],
        help="Leave this empty to use the default reviews_dataset.csv.",
    )

    raw = None
    dataset_source = None
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        dataset_source = uploaded.name
    elif DEFAULT_DATASET_PATH.exists():
        raw = pd.read_csv(DEFAULT_DATASET_PATH)
        dataset_source = "reviews_dataset.csv"

    if raw is None:
        st.warning(
            "Add reviews_dataset.csv to the same GitHub folder as app.py, "
            "or upload a CSV above. The file should include comment_text and, "
            "for evaluation, sentiment_label."
        )
    else:
        df = prepare_comments(raw)
        # Evaluation metrics appear automatically only when human-reviewed labels are present.

        # Clear old results if the input dataset changes, preventing stale predictions.
        dataset_signature = f"{dataset_source}:{len(df)}:{'|'.join(df['comment_text'].head(3).tolist())}"
        if st.session_state.get("csv_dataset_signature") != dataset_signature:
            st.session_state.pop("csv_full_result", None)
            st.session_state["csv_dataset_signature"] = dataset_signature

        if st.button(
            "Run Sentiment Analysis",
            type="primary",
            key="run_csv_both",
        ):
            with st.spinner(
                "Running both models on the same dataset. "
                "RoBERTa may take a few minutes for 1,000+ reviews..."
            ):
                result = run_both_models(df)
                result = add_slang_analysis(result)
                st.session_state["csv_full_result"] = result

        result = st.session_state.get("csv_full_result")
        if result is not None and len(result) == len(df):
            show_dashboard(add_slang_analysis(result))
        else:
            st.info(
                "Select Run Sentiment Analysis to compare both models and explore slang impact."
            )

elif mode == "Analyze a YouTube video":
    video_url = st.text_input("YouTube video URL")
    max_comments = st.slider("Maximum top-level comments", min_value=20, max_value=1000, value=300, step=50)
    if st.button("Fetch & Analyze Comments", type="primary", key="run_youtube_both"):
        match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", video_url)
        if not match:
            st.error("Please enter a valid YouTube video URL.")
        else:
            try:
                api_key = st.secrets.get("YOUTUBE_API_KEY", "")
                if not api_key:
                    st.error("Add YOUTUBE_API_KEY in Streamlit app secrets first.")
                    st.stop()
                youtube = build("youtube", "v3", developerKey=api_key)
                video_id = match.group(1)
                comments, token = [], None
                while len(comments) < max_comments:
                    response = youtube.commentThreads().list(
                        part="snippet", videoId=video_id,
                        maxResults=min(100, max_comments - len(comments)),
                        pageToken=token, textFormat="plainText", order="relevance",
                    ).execute()
                    for item in response.get("items", []):
                        snippet = item["snippet"]["topLevelComment"]["snippet"]
                        comments.append({
                            "comment_text": snippet.get("textDisplay", ""),
                            "author": snippet.get("authorDisplayName", ""),
                            "like_count": snippet.get("likeCount", 0),
                            "published_at": snippet.get("publishedAt", ""),
                        })
                    token = response.get("nextPageToken")
                    if not token or not response.get("items"):
                        break
                if not comments:
                    st.warning("No accessible comments were returned. Comments may be disabled or unavailable.")
                else:
                    df = prepare_comments(pd.DataFrame(comments))
                    with st.spinner("Running VADER and RoBERTa on the same YouTube comments..."):
                        result = run_both_models(df)
                        result = add_slang_analysis(result)
                    st.session_state["youtube_result"] = result
                    st.success(f"Analyzed {len(result):,} comments (requested up to {max_comments:,}).")
                    if len(result) < max_comments:
                        st.caption("YouTube returned fewer accessible top-level comments than requested. The selected number is a maximum; availability depends on the video and YouTube API response.")
            except Exception as exc:
                st.error("Unable to retrieve or analyze comments. Check the API key, quota, and video comment settings.")
                st.caption(str(exc))
    result = st.session_state.get("youtube_result")
    if result is not None:
        show_dashboard(result)
