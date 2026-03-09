def localize_timestamps(fake_scores, timestamps, threshold=0.7):
    """
    Input:
        fake_scores (dict): {frame_path: fake_probability}
        timestamps (dict): {frame_path: time_in_seconds}
        threshold (float): fake score threshold
    Output:
        segments (list): list of {start: float, end: float}
    """
    segments = []
    return segments
