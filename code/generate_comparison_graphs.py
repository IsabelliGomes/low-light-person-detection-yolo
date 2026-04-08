import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to process the CSV and fill missing frames
def process_videos(csv_file):
    # Load the CSV
    df = pd.read_csv(csv_file, header=None)
    df.columns = ['video_name', 'class', 'frame', 'confidence']

    # Convert confidence to percentage
    df['confidence'] = df['confidence'] * 100

    # Determine the global frame range (common frame interval across all videos)
    min_frame = df['frame'].min()
    max_frame = df['frame'].max()

    # Dictionary to store all videos grouped by base video
    all_frames = {}
    for video in df['video_name'].unique():
        video_base = video.split('_', 1)[0]  # Remove the variation
        video_df = df[df['video_name'] == video]

        # Create a DataFrame with the full frame range and fill missing frames with 0
        full_frames = pd.DataFrame({'frame': np.arange(min_frame, max_frame + 1)})
        video_df = pd.merge(full_frames, video_df, how='left', on='frame')
        video_df['video_name'] = video
        video_df['class'] = video_df['class'].fillna('None')
        video_df['confidence'] = video_df['confidence'].fillna(0)

        # Group by base video
        if video_base not in all_frames:
            all_frames[video_base] = []
        all_frames[video_base].append(video_df)

    return all_frames

# Function to generate comparison graphs for each video group
def plot_comparison(all_frames):
    for video_base, video_dfs in all_frames.items():
        plt.figure(figsize=(10, 6))

        for video_df in video_dfs:
            plt.plot(video_df['frame'], video_df['confidence'], label=video_df['video_name'].iloc[0])

        plt.xlabel('Frames')
        plt.ylabel('Confidence Score (%)')
        plt.title(f'Confidence Score Comparison')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 100)
        plt.xlim(0, 1600)
        plt.savefig(f'{video_base}.png')
        plt.close()

# CSV file path
csv_file = # YOUR CSV FILE PATH HERE

# Process videos and generate the graphs
all_frames = process_videos(csv_file)
plot_comparison(all_frames)