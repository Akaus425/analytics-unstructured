
import os
import re
import pandas as pd
import streamlit as st
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googleapiclient.discovery import build

st.set_page_config(
    page_title="Lost in Slang",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Lost in Slang")
st.subheader("Is social listening misreading YouTube audiences?")
st.write(
    "Analyze comment sentiment, investigate model disagreements, "
    "and identify comments that may need contextual review."
)

@st.cache_resource
def get_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def get_roberta():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

def normalize_label(label):
    value = str(label).strip().lower()
    mapping = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    }
    return mapping.get(value, value)

def add_vader(df):
    analyzer = get_vader()
    df = df.copy()
    df["vader_compound"] = df["comment_text"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"]
    )
    df["vader_sentiment"] = df["vader_compound"].apply(
        lambda x: "positive" if x >= 0.05
        else ("negative" if x <= -0.05 else "neutral")
    )
    return df

def add_roberta(df):
    model = get_roberta()
    df = df.copy()
    texts = df["comment_text"].fillna("").astype(str).tolist()
    results = []
    batch_size = 16

    progress = st.progress(0)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch_results = model(batch, truncation=True, max_length=512)
        results.extend(batch_results)
        progress.progress(min(1.0, len(results) / max(1, len(texts))))
    progress.empty()

    df["roberta_sentiment"] = [
        normalize_label(result["label"]) for result in results
    ]
    df["roberta_confidence"] = [
        result["score"] for result in results
    ]
    return df

def add_flags(df):
    df = df.copy()
    text = df["comment_text"].fillna("").astype(str)

    slang_terms = r"\b(goat|rizz|slay|fire|mid|sus|lit|bro|cap|based|no cap)\b"
    df["possible_slang"] = text.str.contains(
        slang_terms, case=False, regex=True
    )
    df["has_emoji_or_symbol"] = text.apply(
        lambda x: bool(re.search(r"[^\w\s.,!?']", x))
    )
    df["repeated_punctuation"] = text.str.contains(
        r"([!?.,])\1{1,}", regex=True
    )

    if {"vader_sentiment", "roberta_sentiment"}.issubset(df.columns):
        df["model_disagreement"] = (
            df["vader_sentiment"] != df["roberta_sentiment"]
        )
    return df

def prepare_comments(raw):
    df = raw.copy()

    candidates = [
        "comment_text", "text", "comment", "content",
        "Comment", "Text"
    ]
    source_col = next(
        (col for col in candidates if col in df.columns), None
    )

    if source_col is None:
        st.error(
            "Couldn't identify the comment-text column. "
            "Expected comment_text, text, comment, or content."
        )
        st.write("Available columns:", list(df.columns))
        st.stop()

    if source_col != "comment_text":
        df = df.rename(columns={source_col: "comment_text"})

    df["comment_text"] = df["comment_text"].fillna("").astype(str)
    df = df[df["comment_text"].str.strip() != ""].copy()
    df = df.reset_index(drop=True)
    return df

def show_dashboard(df):
    total = len(df)
    st.subheader("Audience sentiment overview")
    st.metric("Comments analyzed", f"{total:,}")

    available = [
        col for col in ["vader_sentiment", "roberta_sentiment"]
        if col in df.columns
    ]

    if not available:
        st.info("No sentiment predictions are available yet.")
        return

    tabs = st.tabs([col.replace("_", " ").title() for col in available])

    for tab, col in zip(tabs, available):
        with tab:
            counts = (
                df[col].value_counts()
                .rename_axis("Sentiment")
                .reset_index(name="Comments")
            )
            chart = alt.Chart(counts).mark_bar().encode(
                x=alt.X("Sentiment:N", sort=None),
                y=alt.Y("Comments:Q"),
                tooltip=["Sentiment", "Comments"]
            )
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(counts, hide_index=True, use_container_width=True)

    if "model_disagreement" in df.columns:
        disagreements = int(df["model_disagreement"].sum())
        st.metric(
            "VADER–RoBERTa disagreements",
            f"{disagreements:,}",
            f"{disagreements / total:.1%} of comments" if total else None
        )
        st.caption(
            "Disagreement is a review signal, not proof that either model is wrong."
        )

    flag_cols = [
        col for col in [
            "possible_slang", "has_emoji_or_symbol",
            "repeated_punctuation"
        ] if col in df.columns
    ]
    if flag_cols:
        st.subheader("Potential informal-language cues")
        st.dataframe(
            df[flag_cols].sum().rename("Comments flagged").to_frame(),
            use_container_width=True
        )

    st.subheader("Inspect comments")
    if "model_disagreement" in df.columns:
        only_disagreements = st.checkbox(
            "Show only VADER–RoBERTa disagreements"
        )
        view = df[df["model_disagreement"]] if only_disagreements else df
    else:
        view = df

    st.dataframe(view, use_container_width=True, hide_index=True)

    st.download_button(
        "Download analyzed comments as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="lost_in_slang_analysis.csv",
        mime="text/csv"
    )

mode = st.radio(
    "Choose an analysis mode",
    ["Analyze existing CSV", "Analyze a YouTube video"],
    horizontal=True
)

if mode == "Analyze existing CSV":
    uploaded = st.file_uploader("Upload your comments CSV", type=["csv"])

    if uploaded:
        raw = pd.read_csv(uploaded)
        df = prepare_comments(raw)
        st.success(f"Loaded {len(df):,} comments.")

        has_vader = "vader_sentiment" in df.columns
        has_roberta = "roberta_sentiment" in df.columns

        if not has_vader:
            if st.button("Run VADER"):
                df = add_vader(df)
                st.session_state["csv_analysis"] = df

        if not has_roberta:
            if st.button("Run RoBERTa"):
                with st.spinner("Loading model and analyzing comments..."):
                    df = add_roberta(df)
                st.session_state["csv_analysis"] = df

        if "csv_analysis" in st.session_state:
            df = st.session_state["csv_analysis"]
        elif has_vader or has_roberta:
            df = df.copy()
        else:
            st.info("Upload a CSV with existing predictions or run a model above.")

        if "vader_sentiment" in df.columns and "roberta_sentiment" in df.columns:
            df = add_flags(df)
            show_dashboard(df)
        elif "vader_sentiment" in df.columns or "roberta_sentiment" in df.columns:
            show_dashboard(df)

elif mode == "Analyze a YouTube video":
    video_url = st.text_input("YouTube video URL")
    max_comments = st.slider(
        "Maximum top-level comments",
        min_value=20,
        max_value=500,
        value=100,
        step=20
    )

    st.caption(
        "Requires a YouTube Data API v3 key configured in Streamlit secrets."
    )

    if st.button("Fetch and analyze comments"):
        match = re.search(
            r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})",
            video_url
        )

        if not match:
            st.error("Please enter a valid YouTube video URL.")
        else:
            try:
                api_key = st.secrets.get("YOUTUBE_API_KEY", "")
                if not api_key:
                    st.error(
                        "Add YOUTUBE_API_KEY in Streamlit app secrets first."
                    )
                    st.stop()

                youtube = build(
                    "youtube", "v3", developerKey=api_key
                )
                video_id = match.group(1)
                comments = []
                token = None

                while len(comments) < max_comments:
                    response = youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=min(100, max_comments - len(comments)),
                        pageToken=token,
                        textFormat="plainText",
                        order="relevance"
                    ).execute()

                    for item in response.get("items", []):
                        snippet = item["snippet"]["topLevelComment"]["snippet"]
                        comments.append({
                            "comment_text": snippet.get("textDisplay", ""),
                            "author": snippet.get("authorDisplayName", ""),
                            "like_count": snippet.get("likeCount", 0),
                            "published_at": snippet.get("publishedAt", "")
                        })

                    token = response.get("nextPageToken")
                    if not token or not response.get("items"):
                        break

                if not comments:
                    st.warning(
                        "No accessible comments were returned. "
                        "Comments may be disabled or unavailable."
                    )
                else:
                    df = prepare_comments(pd.DataFrame(comments))
                    with st.spinner("Running VADER..."):
                        df = add_vader(df)

                    run_roberta = st.checkbox(
                        "Also run RoBERTa (slower on CPU)",
                        value=False
                    )
                    if run_roberta:
                        with st.spinner("Running RoBERTa..."):
                            df = add_roberta(df)

                    df = add_flags(df)
                    st.success(f"Retrieved {len(df):,} comments.")
                    show_dashboard(df)

            except Exception as exc:
                st.error(
                    "Unable to retrieve or analyze comments. "
                    "Check the API key, quota, and video comment settings."
                )
                st.caption(str(exc))
