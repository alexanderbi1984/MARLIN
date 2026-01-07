import pandas as pd
import os

meta_path = "/data/Nbi/Syracuse/meta_with_outcomes.xlsx"
exclude_ids = [
    "IMG_0006", "IMG_0008", "IMG_0009", "IMG_0015", 
    "IMG_0036", "IMG_0040", "IMG_0057", "IMG_0098", "IMG_0108"
]

# Load meta
try:
    df = pd.read_excel(meta_path)
    # Normalize columns
    df.columns = [str(c).strip() for c in df.columns]
    
    # Find filename column
    fname_col = None
    for cand in ("file_name", "filename", "video_file", "video_name"):
        if cand in df.columns:
            fname_col = cand
            break
    
    if not fname_col:
        print("Error: Could not find filename column")
        exit(1)
        
    # Helper to get base name
    def get_base(x):
        base = os.path.basename(str(x))
        if "." in base:
            return base.split(".")[0]
        return base.strip()

    df["video_base"] = df[fname_col].apply(get_base)
    
    # Group by subject
    if "subject_id" not in df.columns:
        print("Error: subject_id column missing")
        exit(1)
        
    # Filter out invalid subjects first (NaN)
    df = df.dropna(subset=["subject_id"])
    
    all_subjects = df["subject_id"].unique()
    print(f"Total unique subjects in metadata: {len(all_subjects)}")
    
    # Check per subject video counts
    subject_videos = {}
    for idx, row in df.iterrows():
        sid = row["subject_id"]
        vid = row["video_base"]
        if sid not in subject_videos:
            subject_videos[sid] = []
        subject_videos[sid].append(vid)
        
    print("\nChecking exclusions...")
    removed_subjects = []
    
    for sid, vids in subject_videos.items():
        remaining = [v for v in vids if v not in exclude_ids]
        if len(remaining) == 0:
            print(f"Subject {sid} has NO videos left after exclusion! (Original videos: {vids})")
            removed_subjects.append(sid)
        elif len(remaining) < len(vids):
             print(f"Subject {sid} has partial videos removed. Remaining: {len(remaining)}/{len(vids)}")

    print(f"\nTotal subjects remaining: {len(all_subjects) - len(removed_subjects)}")

except Exception as e:
    print(f"Error: {e}")




