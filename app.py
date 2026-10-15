        help="If you do not upload a file, the app uses reviews_dataset.csv from the repository.",
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
        st.success(f"Loaded {len(df):,} text rows from {dataset_source}.")
        if "sentiment_label" in df.columns or "final_label" in df.columns:
            st.caption("Reference labels found. Uncertain, missing, and unsupported labels will be excluded from three-class accuracy and macro-F1.")
        else:
            st.caption("No reference-label column found. Predictions and comparisons will run, but accuracy and macro-F1 cannot be calculated.")

        # Clear old results if the input dataset changes, preventing stale predictions.
        dataset_signature = f"{dataset_source}:{len(df)}:{'|'.join(df['comment_text'].head(3).tolist())}"
        if st.session_state.get("csv_dataset_signature") != dataset_signature:
            st.session_state.pop("csv_full_result", None)
            st.session_state["csv_dataset_signature"] = dataset_signature

        if st.button(
            "Run Full Dataset Evaluation (VADER + RoBERTa)",
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
                "Click the button above to run both models together and "
                "generate all comparison graphs, slang-impact analysis, "
                "and evaluation metrics."
            )

elif mode == "Analyze a YouTube video":
    video_url = st.text_input("YouTube video URL")
    max_comments = st.slider("Maximum top-level comments", min_value=20, max_value=500, value=100, step=20)
    st.caption("Requires YOUTUBE_API_KEY in Streamlit app secrets. Both models run together when you click the button.")
    if st.button("Fetch comments + run both models", type="primary", key="run_youtube_both"):
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
                    st.success(f"Retrieved and analyzed {len(result):,} comments.")
            except Exception as exc:
                st.error("Unable to retrieve or analyze comments. Check the API key, quota, and video comment settings.")
                st.caption(str(exc))
    result = st.session_state.get("youtube_result")
    if result is not None:
        show_dashboard(result)
