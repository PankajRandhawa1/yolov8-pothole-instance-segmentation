import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import base64
import os

st.set_page_config(
    page_title="Road Pothole Detection",
    page_icon="🛣️",
    layout="wide"
)


def get_base64(file):
    try:
        with open(file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""


bg = get_base64("road_img.jpg")

bg_css = f'''
background: linear-gradient(rgba(0,0,0,0.78), rgba(0,0,0,0.78)),
url("data:image/jpg;base64,{bg}");
background-size: cover;
background-position: center;
''' if bg else 'background-color: #1e1e1e;'


st.markdown(
    f"""
<style>

.stApp {{
    {bg_css}
}}

.block-container {{
    max-width: 1000px;
    padding-top: 2rem;
}}

h1 {{
    font-size: 44px !important;
    text-align: center;
    color: white;
    margin-bottom: 5px;
}}

.subtitle {{
    text-align: center;
    font-size: 18px;
    color: #ccc;
    margin-bottom: 30px;
}}

[data-testid="stSidebar"] {{
    background: rgba(17,24,39,0.92);
}}

[data-testid="stMetric"] {{
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}}

[data-testid="stMetricLabel"] {{
    color: #d1d5db !important;
}}

[data-testid="stMetricValue"] {{
    color: white !important;
}}

[data-testid="stFileUploader"] {{
    width: 100%;
    display: flex;
    justify-content: center;
}}

[data-testid="stFileUploaderDropzone"] {{
    width: 100% !important;
    min-height: 200px !important;
    padding: 20px !important;
    border-radius: 12px !important;
    border: 2px dashed #22C55E !important;
    background: rgba(255,255,255,0.05) !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
}}

[data-testid="stFileUploaderDropzone"] > div {{
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    text-align: center !important;
}}

[data-testid="stFileUploaderDropzone"] button {{
    margin: 0 auto !important;
}}

div.stButton {{
    display: flex;
    justify-content: center;
    margin: 20px 0;
}}

div.stButton > button {{
    background: linear-gradient(135deg, #22C55E, #16a34a) !important;
    color: white !important;
    font-size: 18px !important;
    padding: 12px 40px !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
}}

.stDownloadButton {{
    display: flex;
    justify-content: center;
}}

.stDownloadButton button {{
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 10px 28px !important;
}}

footer {{
    visibility: hidden;
}}

</style>
""",
    unsafe_allow_html=True
)

with st.sidebar:

    st.title("⚙️ Dashboard")

    st.markdown(
        "Control the model and view results here."
    )

    st.divider()

    st.subheader("🎯 Model Settings")

    conf_threshold = st.slider(
        "Confidence Threshold",
        0.1,
        1.0,
        0.25,
        0.05
    )

    st.subheader("📏 Calibration")

    scale = st.number_input(
        "Pixel to CM Scale",
        value=0.02,
        step=0.01
    )

    st.info(
        f"Current Scale: 1 px = {scale} cm"
    )

    st.divider()

    st.subheader("📌 Supported Formats")

    st.markdown("""
    - JPG
    - PNG
    - JPEG
    - MP4
    """)

    st.divider()

    st.subheader("Developed by")

    st.markdown("""
    * *Pankaj Randhawa*
    * *Rohit Rattu*
    * *Rohit Saroya*
    """)

    st.divider()

    st.subheader("🔗 Project Link")

    st.markdown(
        "[GitHub Repository](https://github.com/PankajRandhawa1/yolov8-pothole-instance-segmentation)"
    )

    st.caption("v1.0.0 | © 2026 RoadSafety AI")


st.markdown(
    "<h1>Road Pothole Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Pothole detection using YOLOv8 and Area calculation</div>",
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    return YOLO("yolov8segm.pt")


model = load_model()


def draw_custom_segmentation(result, image):

    annotated = image.copy()

    blue_color = (255, 0, 0)

    if result.masks is not None:

        masks = result.masks.data.cpu().numpy()

        for idx, mask in enumerate(masks):

            mask = cv2.resize(
                mask,
                (annotated.shape[1], annotated.shape[0])
            )

            binary_mask = (mask > 0.5).astype(np.uint8)

            colored_mask = np.zeros_like(annotated)

            colored_mask[:, :] = blue_color

            annotated = np.where(
                binary_mask[:, :, np.newaxis] == 1,
                cv2.addWeighted(
                    annotated,
                    0.55,
                    colored_mask,
                    0.45,
                    0
                ),
                annotated
            )

    boxes = result.boxes

    if boxes:

        for idx, box in enumerate(boxes.xyxy):

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                blue_color,
                2
            )

            cv2.rectangle(
                annotated,
                (x1, y1 - 30),
                (x1 + 60, y1),
                blue_color,
                -1
            )

            cv2.putText(
                annotated,
                f"P{idx+1}",
                (x1 + 8, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    return annotated


st.markdown(
    """
    <h3 style='text-align:center;color:white;'>
    Upload Image or Video
    </h3>
    """,
    unsafe_allow_html=True
)

_, col2, _ = st.columns([1, 2.5, 1])

with col2:

    file = st.file_uploader(
        "Upload Media",
        type=["jpg", "jpeg", "png", "mp4"],
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <p style='text-align:center;
        color:#ccc;
        font-size:16px;
        margin-top:10px;
        font-weight:500;'>
        Drag & Drop or Browse
        </p>
        """,
        unsafe_allow_html=True
    )

if file:

    _, center_btn, _ = st.columns([1, 1, 1])

    with center_btn:

        run = st.button(
            "Run Detection",
            use_container_width=True
        )

    if run:

        with st.spinner("Processing..."):

            if file.type.startswith("image"):

                file_bytes = np.asarray(
                    bytearray(file.read()),
                    dtype=np.uint8
                )

                img = cv2.imdecode(
                    file_bytes,
                    cv2.IMREAD_COLOR
                )

                results = model(
                    img,
                    conf=conf_threshold
                )

                result_img = draw_custom_segmentation(
                    results[0],
                    img
                )

                boxes = results[0].boxes

                count = len(boxes)

                confidences = (
                    boxes.conf.tolist()
                    if boxes else []
                )

                areas_cm = []

                if boxes:

                    for box in boxes.xyxy:

                        x1, y1, x2, y2 = box

                        area_px = (
                            (x2 - x1) *
                            (y2 - y1)
                        )

                        areas_cm.append(
                            float(
                                area_px *
                                (scale ** 2)
                            )
                        )

                st.success("Detection Completed Successfully!")

                res1, res2 = st.columns(2)

                with res1:

                    st.image(
                        img,
                        caption="Original",
                        use_container_width=True
                    )

                with res2:

                    st.image(
                        result_img,
                        caption="Detected",
                        use_container_width=True
                    )

                st.markdown("### Detection Summary")

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Potholes Found",
                    count
                )

                m2.metric(
                    "Avg Confidence",
                    f"{np.mean(confidences):.2f}"
                    if confidences else "0"
                )

                m3.metric(
                    "Avg Area (cm²)",
                    f"{np.mean(areas_cm):.2f}"
                    if areas_cm else "0"
                )

                if count > 0:

                    with st.expander(
                        "View Detailed Analytics"
                    ):

                        for i in range(count):

                            st.write(
                                f"🔵 Pothole {i+1} → "
                                f"Confidence: {confidences[i]:.2f} | "
                                f"Area: {areas_cm[i]:.2f} cm²"
                            )

                _, buffer = cv2.imencode(
                    ".jpg",
                    result_img
                )

                st.download_button(
                    "⬇️ Download Processed Image",
                    buffer.tobytes(),
                    file_name="detected_output.jpg",
                    use_container_width=True
                )

            else:

                tfile = tempfile.NamedTemporaryFile(
                    delete=False
                )

                tfile.write(file.read())

                cap = cv2.VideoCapture(
                    tfile.name
                )

                output_path = "output.mp4"

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                out = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    20.0,
                    (
                        int(cap.get(3)),
                        int(cap.get(4))
                    )
                )

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                progress_bar = st.progress(0)

                frame_idx = 0

                all_confidences = []

                all_areas_cm = []

                max_potholes = 0

                while cap.isOpened():

                    ret, frame = cap.read()

                    if not ret:
                        break

                    results = model(
                        frame,
                        conf=conf_threshold
                    )

                    annotated_frame = draw_custom_segmentation(
                        results[0],
                        frame
                    )

                    boxes = results[0].boxes

                    if boxes:

                        max_potholes = max(
                            max_potholes,
                            len(boxes)
                        )

                        all_confidences.extend(
                            boxes.conf.tolist()
                        )

                        for box in boxes.xyxy:

                            x1, y1, x2, y2 = box

                            area_px = (
                                (x2 - x1) *
                                (y2 - y1)
                            )

                            all_areas_cm.append(
                                float(
                                    area_px *
                                    (scale ** 2)
                                )
                            )

                    out.write(annotated_frame)

                    frame_idx += 1

                    progress_bar.progress(
                        frame_idx / total_frames
                    )

                cap.release()

                out.release()

                progress_bar.empty()

                st.success(
                    "Video Processing Completed!"
                )

                st.markdown(
                    "### Detection Summary (Video)"
                )

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Max Potholes/Frame",
                    max_potholes
                )

                m2.metric(
                    "Overall Avg Confidence",
                    f"{np.mean(all_confidences):.2f}"
                    if all_confidences else "0"
                )

                m3.metric(
                    "Overall Avg Area (cm²)",
                    f"{np.mean(all_areas_cm):.2f}"
                    if all_areas_cm else "0"
                )

                if max_potholes > 0:

                    with st.expander(
                        "View Detailed Analytics"
                    ):

                        st.write(
                            f"Total Frames Processed: "
                            f"{total_frames}"
                        )

                        st.write(
                            f"Largest Pothole Area Detected: "
                            f"{np.max(all_areas_cm):.2f} cm²"
                        )

                        st.write(
                            f"Highest Confidence Score: "
                            f"{np.max(all_confidences):.2f}"
                        )

                with open(output_path, "rb") as f:

                    st.download_button(
                        "⬇️ Download Output Video",
                        f,
                        file_name="processed_video.mp4",
                        use_container_width=True
                    )

                os.remove(output_path)

st.markdown(
    """
    <hr style='margin-top:40px;margin-bottom:10px;'>

    <div style='text-align:center;color:#9ca3af;font-size:14px;'>

    Built with Streamlit, YOLOv8 and OpenCV

    </div>
    """,
    unsafe_allow_html=True
)