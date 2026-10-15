
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googleapiclient.discovery import build


# --------------------------------------------------
# PAGE SETUP AND SLANG DICTIONARY
# --------------------------------------------------

st.set_page_config(
    page_title="Lost in Slang",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Lost in Slang")
st.subheader("Is social listening misreading YouTube audiences?")
st.write(
    "Analyze audience sentiment, detect informal language, "
    "compare VADER and RoBERTa, and investigate how slang-aware "
    "interpretation changes sentiment predictions."
)

DICTIONARY_PATH = Path(__file__).parent / "slang_dictionary.csv"


@st.cache_data
def load_slang_dictionary():
    dictionary = pd.read_csv(DICTIONARY_PATH)

    required_columns = {"term", "meaning", "sentiment_hint", "category"}
    missing = required_columns - set(dictionary.columns)

    if missing:
        raise ValueError(
            "The slang dictionary is missing these columns: "
            + ", ".join(sorted(missing))
        )

    dictionary["term"] = (
        dictionary["term"].fillna("").astype(str).str.strip().str.lower()
    )
    dictionary = dictionary[dictionary["term"] != ""]
    dictionary = dictionary.drop_duplicates(subset=["term"])

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
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    }

    return mapping.get(value, value)


def add_vader(df):
    df = df.copy()
    analyzer = get_vader()

    df["vader_compound"] = df["comment_text"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"]
    )

    df["vader_sentiment"] = df["vader_compound"].apply(
        lambda x: (
            "positive"
            if x >= 0.05
            else ("negative" if x <= -0.05 else "neutral")
        )
    )

    return df


def add_roberta(df):
    df = df.copy()
    model = get_roberta()

    texts = df["comment_text"].fillna("").astype(str).tolist()
    results = []
    batch_size = 16

    progress = st.progress(0)

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        batch_results = model(
            batch,
            truncation=True,
            max_length=512,
        )

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


# --------------------------------------------------
# SLANG DETECTION AND SLANG-AWARE ADJUSTMENT
# --------------------------------------------------

def add_flags(df):
    df = df.copy()
    text = df["comment_text"].fillna("").astype(str)

    dictionary = slang_dictionary.copy()

    # Match longer expressions first.
    terms = dictionary["term"].tolist()
    terms = sorted(terms, key=len, reverse=True)

    def find_slang(comment):
        comment_lower = comment.lower()
        matched = []

        for term in terms:
            pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"

            if re.search(pattern, comment_lower):
                matched.append(term)

        return matched

    df["detected_slang"] = text.apply(find_slang)
    df["possible_slang"] = df["detected_slang"].apply(bool)

    meaning_map = dictionary.set_index("term")["meaning"].to_dict()
    hint_map = dictionary.set_index("term")["sentiment_hint"].to_dict()
    category_map = dictionary.set_index("term")["category"].to_dict()

    df["slang_meanings"] = df["detected_slang"].apply(
        lambda matches: "; ".join(
            f"{term}: {meaning_map.get(term, '')}"
            for term in matches
        )
    )

    df["slang_sentiment_hints"] = df["detected_slang"].apply(
        lambda matches: "; ".join(
            f"{term}: {hint_map.get(term, 'context-dependent')}"
            for term in matches
        )
    )

    df["slang_categories"] = df["detected_slang"].apply(
        lambda matches: "; ".join(
            f"{term}: {category_map.get(term, '')}"
            for term in matches
        )
    )

    df["has_emoji_or_symbol"] = text.apply(
        lambda x: bool(re.search(r"[^\w\s.,!?']", x))
    )

    df["repeated_punctuation"] = text.str.contains(
        r"[!?.,]{2,}",
        regex=True,
        na=False,
    )

    # Derive a slang-aware label from clear dictionary hints.
    # Ambiguous or tied hints leave the original model prediction unchanged.
    def adjust_sentiment(original_label, matched_terms):
        if not matched_terms:
            return original_label

        hint_counts = {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
        }

        for term in matched_terms:
            hint = str(hint_map.get(term, "")).strip().lower()

            if hint in hint_counts:
                hint_counts[hint] += 1

        highest_count = max(hint_counts.values())

        if highest_count == 0:
            return original_label

        winners = [
            label
            for label, count in hint_counts.items()
            if count == highest_count
        ]

        # Do not force an adjustment when dictionary hints conflict.
        if len(winners) != 1:
            return original_label

        return winners[0]

    if "vader_sentiment" in df.columns:
        df["vader_sentiment_with_slang"] = df.apply(
            lambda row: adjust_sentiment(
                row["vader_sentiment"],
                row["detected_slang"],
            ),
            axis=1,
        )

        df["vader_changed_with_slang"] = (
            df["vader_sentiment"]
            != df["vader_sentiment_with_slang"]
        )

    if "roberta_sentiment" in df.columns:
        df["roberta_sentiment_with_slang"] = df.apply(
            lambda row: adjust_sentiment(
                row["roberta_sentiment"],
                row["detected_slang"],
            ),
            axis=1,
        )

        df["roberta_changed_with_slang"] = (
            df["roberta_sentiment"]
            != df["roberta_sentiment_with_slang"]
        )

    if {
        "vader_sentiment",
        "roberta_sentiment",
    }.issubset(df.columns):
        df["model_disagreement"] = (
            df["vader_sentiment"] != df["roberta_sentiment"]
        )

    if {
        "vader_sentiment_with_slang",
        "roberta_sentiment_with_slang",
    }.issubset(df.columns):
        df["model_disagreement_with_slang"] = (
            df["vader_sentiment_with_slang"]
            != df["roberta_sentiment_with_slang"]
        )

    return df


# --------------------------------------------------
# DASHBOARD: SENTIMENT AND COMPARISON GRAPHS
# --------------------------------------------------

def show_dashboard(df):
    total = len(df)

    st.subheader("1. Audience sentiment overview")
    st.metric("Comments analyzed", f"{total:,}")

    original_models = []
    if "vader_sentiment" in df.columns:
        original_models.append(("VADER", "vader_sentiment"))
    if "roberta_sentiment" in df.columns:
        original_models.append(("RoBERTa", "roberta_sentiment"))

    if not original_models:
        st.info("No sentiment predictions are available yet.")
        return

    # Original model sentiment distributions
    st.subheader("Original model predictions")

    tabs = st.tabs([name for name, _ in original_models])

    for tab, (name, col) in zip(tabs, original_models):
        with tab:
            counts = (
                df[col]
                .value_counts()
                .reindex(["positive", "neutral", "negative"], fill_value=0)
                .rename_axis("Sentiment")
                .reset_index(name="Comments")
            )

            chart = alt.Chart(counts).mark_bar().encode(
                x=alt.X("Sentiment:N", sort=["positive", "neutral", "negative"]),
                y=alt.Y("Comments:Q"),
                tooltip=["Sentiment", "Comments"],
            )

            st.altair_chart(chart, use_container_width=True)
            st.dataframe(counts, hide_index=True, use_container_width=True)

    # Optional comparison graph
    st.subheader("2. Compare VADER and RoBERTa")

    available_comparisons = []

    if {
        "vader_sentiment",
        "roberta_sentiment",
    }.issubset(df.columns):
        available_comparisons.append("Original predictions")

    if {
        "vader_sentiment_with_slang",
        "roberta_sentiment_with_slang",
    }.issubset(df.columns):
        available_comparisons.append("Slang-adjusted predictions")

    if {
        "vader_sentiment",
        "roberta_sentiment",
        "vader_sentiment_with_slang",
        "roberta_sentiment_with_slang",
    }.issubset(df.columns):
        available_comparisons.append("Before vs after slang adjustment")

    if available_comparisons:
        graph_choice = st.selectbox(
            "Choose the comparison graph",
            available_comparisons,
        )

        graph_data = []

        if graph_choice == "Original predictions":
            columns = [
                ("VADER", "vader_sentiment"),
                ("RoBERTa", "roberta_sentiment"),
            ]

        elif graph_choice == "Slang-adjusted predictions":
            columns = [
                ("VADER", "vader_sentiment_with_slang"),
                ("RoBERTa", "roberta_sentiment_with_slang"),
            ]

        else:
            columns = [
                ("VADER — Before", "vader_sentiment"),
                ("VADER — After", "vader_sentiment_with_slang"),
                ("RoBERTa — Before", "roberta_sentiment"),
                ("RoBERTa — After", "roberta_sentiment_with_slang"),
            ]

        for model_name, col in columns:
            counts = df[col].value_counts()

            for sentiment in ["positive", "neutral", "negative"]:
                graph_data.append({
                    "Model": model_name,
                    "Sentiment": sentiment,
                    "Comments": int(counts.get(sentiment, 0)),
                })

        graph_df = pd.DataFrame(graph_data)

        chart = alt.Chart(graph_df).mark_bar().encode(
            x=alt.X(
                "Sentiment:N",
                sort=["positive", "neutral", "negative"],
                title="Sentiment",
            ),
            y=alt.Y("Comments:Q", title="Number of comments"),
            xOffset="Model:N",
            color=alt.Color("Model:N", title="Model / version"),
            tooltip=["Model", "Sentiment", "Comments"],
        ).properties(height=380)

        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "The graph compares sentiment counts. A changed distribution "
            "does not by itself demonstrate better accuracy."
        )

    # Before/after impact metrics
    st.subheader("3. What changed after slang adjustment?")

    metric_cols = st.columns(2)

    for col_box, model_name, original_col, adjusted_col in [
        (
            metric_cols[0],
            "VADER",
            "vader_sentiment",
            "vader_sentiment_with_slang",
        ),
        (
            metric_cols[1],
            "RoBERTa",
            "roberta_sentiment",
            "roberta_sentiment_with_slang",
        ),
    ]:
        if original_col in df.columns and adjusted_col in df.columns:
            changed = int((df[original_col] != df[adjusted_col]).sum())
            rate = changed / total if total else 0

            with col_box:
                st.markdown(f"**{model_name}**")
                st.metric("Predictions changed", f"{changed:,}")
                st.metric("Share of all comments changed", f"{rate:.1%}")

                comparison = pd.DataFrame({
                    "Before": df[original_col].value_counts(),
                    "After": df[adjusted_col].value_counts(),
                }).reindex(["positive", "neutral", "negative"]).fillna(0).astype(int)

                st.dataframe(comparison, use_container_width=True)

    # Agreement before and after
    if {
        "model_disagreement",
        "model_disagreement_with_slang",
    }.issubset(df.columns):
        st.subheader("4. Model agreement before and after")

        before = int(df["model_disagreement"].sum())
        after = int(df["model_disagreement_with_slang"].sum())

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Original model disagreements",
                f"{before:,} ({before / total:.1%})" if total else "0",
            )

        with col2:
            st.metric(
                "Disagreements after adjustment",
                f"{after:,} ({after / total:.1%})" if total else "0",
            )

        st.caption(
            "Greater agreement is not necessarily greater accuracy. "
            "Use human-reviewed labels to evaluate whether predictions "
            "are actually more correct."
        )

    # Informal-language indicators
    st.subheader("5. Informal-language cues")

    flag_cols = [
        col for col in [
            "possible_slang",
            "has_emoji_or_symbol",
            "repeated_punctuation",
        ]
        if col in df.columns
    ]

    if flag_cols:
        flag_summary = (
            df[flag_cols]
            .fillna(False)
            .astype(bool)
            .sum()
            .rename("Comments flagged")
            .to_frame()
        )

        st.dataframe(flag_summary, use_container_width=True)

    # Dictionary detail
    if "detected_slang" in df.columns:
        st.subheader("6. Slang dictionary insights")

        slang_comments = df[df["possible_slang"]].copy()

        st.metric(
            "Comments containing dictionary terms",
            f"{len(slang_comments):,}",
        )

        if not slang_comments.empty:
            display_cols = [
                "comment_text",
                "detected_slang",
                "slang_meanings",
                "slang_sentiment_hints",
            ]

            if "vader_sentiment" in df.columns:
                display_cols += [
                    "vader_sentiment",
                    "vader_sentiment_with_slang",
                ]

            if "roberta_sentiment" in df.columns:
                display_cols += [
                    "roberta_sentiment",
                    "roberta_sentiment_with_slang",
                ]

            display_cols = [
                col for col in display_cols if col in slang_comments.columns
            ]

            st.dataframe(
                slang_comments[display_cols],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No expressions from the current dictionary were detected.")

    # Filter comments for inspection
    st.subheader("7. Inspect comments")

    filter_options = ["All comments"]

    if "model_disagreement" in df.columns:
        filter_options.append("Original model disagreements")

    if "possible_slang" in df.columns:
        filter_options.append("Comments with detected slang")

    if "vader_changed_with_slang" in df.columns:
        filter_options.append("Comments where VADER changed")

    if "roberta_changed_with_slang" in df.columns:
        filter_options.append("Comments where RoBERTa changed")

    selected_filter = st.selectbox(
        "Filter the comments table",
        filter_options,
    )

    view = df.copy()

    if selected_filter == "Original model disagreements":
        view = view[view["model_disagreement"]]
    elif selected_filter == "Comments with detected slang":
        view = view[view["possible_slang"]]
    elif selected_filter == "Comments where VADER changed":
        view = view[view["vader_changed_with_slang"]]
    elif selected_filter == "Comments where RoBERTa changed":
        view = view[view["roberta_changed_with_slang"]]

    st.dataframe(view, use_container_width=True, hide_index=True)

    st.download_button(
        "Download analyzed comments as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="lost_in_slang_analysis.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# INPUT MODE: CSV OR YOUTUBE
# --------------------------------------------------

mode = st.radio(
    "Choose an analysis mode",
    ["Analyze existing CSV", "Analyze a YouTube video"],
    horizontal=True,
)

if mode == "Analyze existing CSV":
    uploaded = st.file_uploader("Upload your comments CSV", type=["csv"])

    if uploaded:
        raw = pd.read_csv(uploaded)
        df = raw.copy()

        candidates = [
            "comment_text", "text", "comment", "content", "Comment", "Text"
        ]
        source_col = next(
            (col for col in candidates if col in df.columns),
            None,
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

        st.success(f"Loaded {len(df):,} comments.")

        # Keep previously calculated predictions when they exist.
        if "vader_sentiment" not in df.columns:
            if st.button("Run VADER", key="csv_vader"):
                with st.spinner("Running VADER..."):
                    df = add_vader(df)
                st.session_state["csv_analysis"] = df

        if "roberta_sentiment" not in df.columns:
            if st.button("Run RoBERTa", key="csv_roberta"):
                with st.spinner("Loading RoBERTa and analyzing comments..."):
                    df = add_roberta(df)
                st.session_state["csv_analysis"] = df

        if "csv_analysis" in st.session_state:
            saved = st.session_state["csv_analysis"]
            if len(saved) == len(df):
                df = saved

        has_vader = "vader_sentiment" in df.columns
        has_roberta = "roberta_sentiment" in df.columns

        if has_vader or has_roberta:
            df = add_flags(df)
            show_dashboard(df)
        else:
            st.info(
                "Upload a CSV with existing sentiment predictions, "
                "or run VADER or RoBERTa above."
            )

elif mode == "Analyze a YouTube video":
    video_url = st.text_input("YouTube video URL")

    max_comments = st.slider(
        "Maximum top-level comments",
        min_value=20,
        max_value=500,
        value=100,
        step=20,
    )

    run_roberta_option = st.checkbox(
        "Also run RoBERTa (slower on CPU)",
        value=False,
    )

    st.caption(
        "Requires a YouTube Data API v3 key configured in Streamlit secrets."
    )

    if st.button("Fetch and analyze comments"):
        match = re.search(
            r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})",
            video_url,
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
                    "youtube",
                    "v3",
                    developerKey=api_key,
                )

                video_id = match.group(1)
                comments = []
                token = None

                while len(comments) < max_comments:
                    response = (
                        youtube.commentThreads()
                        .list(
                            part="snippet",
                            videoId=video_id,
                            maxResults=min(
                                100,
                                max_comments - len(comments),
                            ),
                            pageToken=token,
                            textFormat="plainText",
                            order="relevance",
                        )
                        .execute()
                    )

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
                    st.warning(
                        "No accessible comments were returned. "
                        "Comments may be disabled or unavailable."
                    )
                else:
                    df = pd.DataFrame(comments)
                    df["comment_text"] = (
                        df["comment_text"].fillna("").astype(str)
                    )
                    df = df[df["comment_text"].str.strip() != ""].copy()
                    df = df.reset_index(drop=True)

                    with st.spinner("Running VADER..."):
                        df = add_vader(df)

                    if run_roberta_option:
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
