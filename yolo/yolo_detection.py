import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import math
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------------
# Page config + CSS
# -------------------------
st.set_page_config(page_title="YOLOv8 Traffic Violation Dashboard", layout="wide")
st.markdown("""
<style>
  .stApp { background-color: #0b0b0b; color: #ffffff; }
  .block-container { padding-top: 0.75rem; }
  .css-1d391kg, .css-1v3fvcr { color: #fff; }
  .stButton>button { background-color:#1f1f1f; color:#fff; }
  .stSidebar { background-color:#0b0b0b; color:#fff; }
  header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("🚦 YOLOv8 Traffic Violation Dashboard")
st.write("Upload video/image → configure → press Run detection. Annotated video will play in-app; CSVs & charts available after run.")

# -------------------------
# Initialize session_state variables
# -------------------------
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "processed_vio_df" not in st.session_state:
    st.session_state.processed_vio_df = pd.DataFrame()
if "processed_det_df" not in st.session_state:
    st.session_state.processed_det_df = pd.DataFrame()
if "processed_out_vid_bytes" not in st.session_state:
    st.session_state.processed_out_vid_bytes = None

# -------------------------
# # Sidebar form: parameters
# -------------------------
with st.sidebar.form(key="params_form"):
    st.header("Input / Model")
    uploaded = st.file_uploader(
        "Upload video or image", 
        type=["mp4", "mov", "avi", "jpg", "png", "jpeg"]
    )
    if uploaded is not None:
        st.session_state.uploaded_file = uploaded

    input_file = st.session_state.uploaded_file
    model_path = st.text_input("YOLOv8 model path (local)", value="yolov8n.pt")
    helmet_model_path = st.text_input("Helmet model path (optional)", value="")

    st.header("Speed calibration")
    fps_input = st.number_input("Fallback Video FPS", min_value=1, value=25)
    meters_per_pixel = st.number_input("Meters per pixel", min_value=0.0001, value=0.05, format="%.5f")

    st.header("Detection & Performance")
    speed_threshold_kmph = st.number_input("Speed threshold (km/h)", min_value=1, value=60)
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.35)
    imgsz = st.selectbox("Inference image size (px)", [640, 480, 320], index=0)
    iou_tracker_threshold = st.slider("IoU threshold for tracker", 0.0, 1.0, 0.3)
    frame_skip = st.number_input("Frame skip (process every Nth frame)", min_value=1, value=2)

    run_button = st.form_submit_button("Run detection")


# Clear uploaded file (safe)
if st.sidebar.button("Clear uploaded file"):
    st.session_state.uploaded_file = None
    st.session_state.processed_vio_df = pd.DataFrame()
    st.session_state.processed_det_df = pd.DataFrame()
    st.session_state.processed_out_vid_bytes = None
    st.stop()  # stops execution to let user upload a new file

# -------------------------
# Tracker and utils
# -------------------------
class SimpleTracker:
    def __init__(self, iou_threshold=0.3):
        self.next_id = 0
        self.tracks = {}
        self.iou_threshold = iou_threshold

    @staticmethod
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, (boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
        boxBArea = max(0, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
        denom = boxAArea + boxBArea - interArea
        return 0.0 if denom==0 else interArea/denom

    @staticmethod
    def centroid(box):
        x1,y1,x2,y2 = box
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def update(self, boxes):
        assigned = {}
        new_tracks = {}
        for tid, t in list(self.tracks.items()):
            best_iou = 0; best_idx = -1
            for i, b in enumerate(boxes):
                if i in assigned: continue
                iou_val = self.iou(t['box'], b)
                if iou_val > best_iou: best_iou = iou_val; best_idx = i
            if best_iou >= self.iou_threshold and best_idx != -1:
                assigned[best_idx] = tid
                c = self.centroid(boxes[best_idx])
                history = t['history'] + [c]
                new_tracks[tid] = {'box': boxes[best_idx], 'centroid': c, 'history': history}
        for i, b in enumerate(boxes):
            if i not in assigned:
                tid = self.next_id; self.next_id += 1
                c = self.centroid(b)
                new_tracks[tid] = {'box': b, 'centroid': c, 'history': [c]}
        self.tracks = new_tracks
        return self.tracks

def box_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    return (xB - xA) > 0 and (yB - yA) > 0

def pixels_to_kmph(dx, dy, frames_elapsed, fps, meters_per_pixel):
    dist_pixels = math.hypot(dx, dy)
    meters = dist_pixels * meters_per_pixel
    seconds = frames_elapsed / fps if frames_elapsed>0 else 1.0
    return (meters / seconds) * 3.6

# -------------------------
# Cached model loaders
# -------------------------
@st.cache_resource
def load_yolo(path):
    return YOLO(path)

@st.cache_resource
def load_helmet(path):
    return YOLO(path)

# -------------------------
# Main processing
# -------------------------
def process_media(temp_path, model_obj, helmet_obj, fps_user, meters_per_pixel,
                  speed_thresh_kmph, min_conf, iou_tracker, imgsz, frame_skip, st_progress=None):
    cap = cv2.VideoCapture(temp_path)
    input_is_image = False
    img = None
    if not cap.isOpened():
        img = cv2.imread(temp_path)
        if img is None: raise ValueError("Cannot open media")
        input_is_image = True

    fps = int(cap.get(cv2.CAP_PROP_FPS)) if not input_is_image else int(fps_user)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if not input_is_image else img.shape[1]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if not input_is_image else img.shape[0]
    if width==0 or height==0: raise ValueError("Invalid video dimensions")

    out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = out_temp.name; out_temp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    tracker = SimpleTracker(iou_threshold=iou_tracker)

    frame_id = 0
    violations = []
    detections_summary = []
    last_annotated = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not input_is_image else 1
    total_frames = max(total_frames, 1)

    while True:
        if input_is_image: frame = img.copy()
        else:
            ret, frame = cap.read()
            if not ret: break

        do_infer = frame_id % frame_skip == 0
        if do_infer:
            results = model_obj(frame, imgsz=imgsz)[0]
            names = results.names
            boxes=[]; classes=[]; confs=[]
            if getattr(results, "boxes", None) is not None and len(results.boxes)>0:
                xyxy = results.boxes.xyxy.cpu().numpy()
                conf_arr = results.boxes.conf.cpu().numpy()
                cls_arr = results.boxes.cls.cpu().numpy().astype(int)
                for b,c,cf in zip(xyxy, cls_arr, conf_arr):
                    if cf < min_conf: continue
                    boxes.append([int(b[0]),int(b[1]),int(b[2]),int(b[3])])
                    classes.append(c); confs.append(float(cf))

            vehicle_boxes=[]; vehicle_names=[]; vehicle_confs=[]
            for b, cls_id, cf in zip(boxes, classes, confs):
                cls_name = names.get(cls_id,str(cls_id)).lower()
                if cls_name in ("car","truck","bus","motorcycle","bicycle","motorbike","bike","van","auto"):
                    vehicle_boxes.append(b); vehicle_names.append(cls_name); vehicle_confs.append(cf)

            tracks = tracker.update(vehicle_boxes)
            track_map = {tid:t for tid,t in tracks.items()}

            helmet_boxes=[]
            if helmet_obj:
                hres = helmet_obj(frame, imgsz=imgsz)[0]
                if getattr(hres,"boxes",None) is not None and len(hres.boxes)>0:
                    hxyxy = hres.boxes.xyxy.cpu().numpy()
                    hconf = hres.boxes.conf.cpu().numpy()
                    for hb,hcf in zip(hxyxy,hconf):
                        if hcf>=min_conf: helmet_boxes.append([int(hb[0]),int(hb[1]),int(hb[2]),int(hb[3])])

            person_boxes=[]
            for b,cls_id,cf in zip(boxes,classes,confs):
                cls_name = names.get(cls_id,str(cls_id)).lower()
                if cls_name=="person": person_boxes.append(b)

            for tid, track in track_map.items():
                box = track['box']; cx,cy = track['centroid']; history = track['history']
                speed_kmph = 0
                if len(history)>=2:
                    dx = cx - history[-2][0]; dy = cy - history[-2][1]
                    speed_kmph = pixels_to_kmph(dx, dy, 1, fps, meters_per_pixel)

                matched_label="vehicle"
                for b,nm,cf in zip(vehicle_boxes,vehicle_names,vehicle_confs):
                    if SimpleTracker.iou(box,b)>0.5: matched_label=nm; break

                no_helmet=None
                if matched_label in ("motorcycle","bike","bicycle","motorbike") and helmet_obj:
                    helmet_present = any(box_overlap(box,hb) for hb in helmet_boxes)
                    no_helmet = not helmet_present

                triple_riding=False
                if matched_label in ("motorcycle","bike","bicycle","motorbike"):
                    persons_overlapping=sum(1 for pb in person_boxes if box_overlap(box,pb))
                    if persons_overlapping>=3: triple_riding=True

                overspeed=speed_kmph>=speed_thresh_kmph
                record_time=datetime.now().isoformat()
                violation_types=[]
                if overspeed: violation_types.append("Overspeed")
                if no_helmet is True: violation_types.append("No Helmet")
                if triple_riding: violation_types.append("Triple Riding")

                detections_summary.append({
                    "frame_id": frame_id,
                    "track_id": tid,
                    "vehicle_type": matched_label,
                    "speed_kmph": round(speed_kmph,2),
                    "violations": ";".join(violation_types) if violation_types else "",
                    "bbox": box
                })

                if violation_types:
                    violations.append({
                        "frame_id": frame_id,
                        "track_id": tid,
                        "vehicle_type": matched_label,
                        "speed_kmph": round(speed_kmph,2),
                        "violation_type": ";".join(violation_types),
                        "timestamp": record_time,
                        "x1": box[0],"y1": box[1],"x2": box[2],"y2": box[3]
                    })

                # annotate
                x1,y1,x2,y2=map(int,box)
                color=(0,255,0) if not violation_types else (0,0,255)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                label=f"ID:{tid} {matched_label} {round(speed_kmph,1)}km/h"
                if violation_types: label+=" <- "+",".join(violation_types)
                cv2.putText(frame,label,(x1,max(12,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)

            last_annotated=frame.copy()
        else:
            if last_annotated is not None:
                frame=last_annotated.copy()

        out_vid.write(frame)
        frame_id+=1
        if st_progress: st_progress.progress(min(frame_id/total_frames,1.0))
        if input_is_image: break

    if not input_is_image: cap.release()
    out_vid.release()
    if st_progress: st_progress.empty()

    return pd.DataFrame(violations), pd.DataFrame(detections_summary), out_path

# -------------------------
# Run detection
# -------------------------
if run_button:
    if input_file is None:
        st.error("Please upload a file first.")
        st.stop()

    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(input_file.name)[1])
    tfile.write(input_file.read()); tfile.flush(); tfile.close()
    temp_path = tfile.name

    st.info("Loading YOLO model(s)...")
    try:
        model_obj = load_yolo(model_path)
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        st.stop()

    helmet_obj = None
    if helmet_model_path.strip():
        try:
            helmet_obj = load_helmet(helmet_model_path)
        except Exception:
            st.warning("Helmet model failed; skipping helmet detection.")

    st.info("Running detection (this may take a while)...")
    progress = st.progress(0.0)
    vio_df, det_df, out_vid_path = process_media(
        temp_path, model_obj, helmet_obj, fps_input, meters_per_pixel,
        speed_threshold_kmph, min_conf, iou_tracker_threshold, imgsz, frame_skip,
        st_progress=progress
    )
    progress.empty()

    with open(out_vid_path,"rb") as f:
        out_vid_bytes = f.read()

    # Store in session_state
    st.session_state.processed_vio_df = vio_df
    st.session_state.processed_det_df = det_df
    st.session_state.processed_out_vid_bytes = out_vid_bytes

# -------------------------
# Display results if file processed
# -------------------------
if st.session_state.processed_out_vid_bytes is not None:
    vio_df = st.session_state.processed_vio_df
    det_df = st.session_state.processed_det_df
    out_vid_bytes = st.session_state.processed_out_vid_bytes

    st.markdown("### Violations (sample)")
    if vio_df.empty:
        st.info("No violations detected.")
    else:
        st.dataframe(vio_df.head(200))
        st.download_button(
            "Download Violations CSV",
            data=vio_df.to_csv(index=False).encode('utf-8'),
            file_name="violations.csv",
            mime="text/csv"
        )

    st.markdown("### Annotated Output Video")
    st.video(out_vid_bytes)
    st.download_button(
        "Download Annotated Video",
        data=out_vid_bytes,
        file_name="annotated_output.mp4",
        mime="video/mp4"
    )

    # -------------------------
    # Dashboard charts
    # -------------------------
    st.markdown("### Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        import matplotlib.cm as cm
        import numpy as np

        st.subheader("Violations by Type")
        if not vio_df.empty:
            violation_series = vio_df['violation_type'].dropna()
            violation_series = violation_series[violation_series != ""]

            if not violation_series.empty:
                vc = violation_series.str.split(";").explode().value_counts()
                cmap = cm.get_cmap('tab20')  # 20 distinct colors
                colors = [cmap(i/len(vc)) for i in range(len(vc))]

                fig, ax = plt.subplots(figsize=(6,4))
                ax.pie(
            vc.values, 
            labels=vc.index, 
            autopct=lambda p: f"{p:.1f}% ({int(round(p*sum(vc.values)/100))})",  # show count
            colors=colors,
            textprops={'color':'white'} )
                ax.set_title("Violation Types Distribution", color='white')
                ax.set_aspect('equal')
                st.pyplot(fig)
            else:
                st.info("No violations detected to display.")
        else:
            st.info("No violations detected.")
    with col2:
        st.subheader("Speeds distribution")
        if not det_df.empty:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.hist(det_df["speed_kmph"].dropna(), bins=20)
            ax2.set_xlabel("km/h"); ax2.set_ylabel("Counts")
            st.pyplot(fig2)
        else:
            st.info("No speed data to plot.")

    st.markdown("### Summary Metrics")
    st.metric("Total Violations", len(vio_df))
    st.metric("Total Tracked Vehicles", len(det_df))
else:
    st.info("Upload a video or image and run detection to see results.")
