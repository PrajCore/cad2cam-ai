import streamlit as st
import trimesh
import numpy as np
import joblib
from streamlit_stl import stl_from_file
import tempfile
import os

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="CAD2CAM AI", layout="wide")

# ---------------- CLEAN DARK UI ---------------- #
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", "Inter", sans-serif;
}

.main {
    background-color: #0E1117;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 10px;
}

.highlight {
    background-color: #111827;
    padding: 12px;
    border-radius: 8px;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ML ---------------- #
try:
    time_model = joblib.load("time_model.pkl")
    op_model = joblib.load("op_model.pkl")
except:
    time_model = None
    op_model = None

# ---------------- HEADER ---------------- #
st.title("⚙️ CAD2CAM AI")
st.caption("AI-Assisted CAD-to-CAM Machining Planner")

st.divider()

# ---------------- INPUT ---------------- #
uploaded_file = st.file_uploader("Upload STL file", type=["stl"])

material = st.selectbox(
    "Select Material",
    ["Aluminium", "Steel", "Cast Iron", "Plastic"]
)

finish = st.selectbox(
    "Surface Finish Requirement",
    ["Rough", "Medium", "Fine"]
)

# ---------------- FIXED HOLE DETECTION ---------------- #
def detect_hole(mesh):
    bbox_vol = np.prod(mesh.bounding_box.extents)
    mesh_vol = mesh.volume if mesh.volume > 0 else 1e-6

    void_ratio = 1 - (mesh_vol / bbox_vol)
    curvature_hint = np.std(mesh.face_normals)

    return (void_ratio > 0.2) or (curvature_hint > 0.3)

# ---------------- FEATURE EXTRACTION ---------------- #
def extract_features(file):
    mesh_obj = trimesh.load(file, file_type='stl', force='mesh')
    mesh_obj.process()

    volume = mesh_obj.volume or 0
    surface_area = mesh_obj.area or 0

    length, width, height = mesh_obj.bounding_box.extents
    is_cylindrical = abs(length - width) < 0.1 * max(length, width)

    flatness = np.mean(np.abs(mesh_obj.face_normals[:, 2]))
    curvature = np.std(mesh_obj.face_normals)

    # ✅ FIXED LINE HERE
    has_hole = detect_hole(mesh_obj)

    complexity = surface_area / (volume + 1e-6)
    hole_indicator = 1 if has_hole else 0
    curvature_proxy = curvature * flatness

    return {
        "Volume": volume,
        "Surface Area": surface_area,
        "Complexity": complexity,
        "Hole Indicator": hole_indicator,
        "Flatness": flatness,
        "Curvature Proxy": curvature_proxy,
        "Curvature": curvature,
        "Is Cylindrical": is_cylindrical,
        "Has Hole": has_hole
    }

# ---------------- ML ---------------- #
def ml_predict(features):
    if time_model is None or op_model is None:
        return None, None

    X = [[
        features["Volume"],
        features["Surface Area"],
        features["Complexity"],
        features["Hole Indicator"],
        features["Flatness"],
        features["Curvature Proxy"]
    ]]

    return time_model.predict(X)[0], op_model.predict(X)[0]

# ---------------- PROCESS PLAN ---------------- #
def generate_process_plan(features, ml_op, finish):

    steps = []

    if features["Is Cylindrical"]:
        steps += ["Lathe Setup", "Facing", "Turning"]
    else:
        steps += ["Milling Setup", "Facing"]

    if features["Has Hole"]:
        steps.append("Drilling")

    if features["Curvature"] > 0.5:
        steps.append("Rough Machining")

    if ml_op and "thread" in str(ml_op).lower():
        steps.append("Threading")

    if finish == "Fine":
        steps.append("Precision Finishing")

    steps.append("Finishing")
    return steps

# ---------------- TOOL SELECTION ---------------- #
def select_tools(steps):
    tool_map = {
        "Facing": "Face Mill",
        "Turning": "Turning Tool",
        "Drilling": "Drill",
        "Rough Machining": "End Mill",
        "Finishing": "Ball Nose",
        "Precision Finishing": "Fine Ball Nose",
        "Threading": "Thread Tool"
    }

    tools = []
    for s in steps:
        for key in tool_map:
            if key in s:
                tools.append(tool_map[key])

    return list(set(tools))

# ---------------- CNC ---------------- #
def generate_gcode(steps, material, finish):

    feed = 300 if material == "Aluminium" else 200
    if material == "Steel":
        feed = 180

    depth = 1.0 if finish == "Rough" else 0.5 if finish == "Medium" else 0.2

    gcode = ["G21", "G90", "G0 X0 Y0 Z5", f"G1 F{feed}"]

    for s in steps:
        if "Facing" in s:
            gcode.append(f"G1 Z-{depth} X50")
        elif "Turning" in s:
            gcode.append("G1 X0 Z-20")
        elif "Drilling" in s:
            gcode.append("G81 X10 Y10 Z-5")
        elif "Finishing" in s:
            gcode.append(f"G1 Z-{depth/2} X50")

    gcode.append("M30")
    return gcode

# ---------------- 3D VIEW ---------------- #
def show_3d(file):
    stl_from_file(file)

# ---------------- MAIN ---------------- #
if uploaded_file:

    st.success("File uploaded successfully")

    # 3D VIEW
    st.subheader("3D Model")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    show_3d(path)
    os.remove(path)

    # PROCESS
    with st.spinner("Analyzing model..."):
        features = extract_features(uploaded_file)

    st.divider()

    # GEOMETRY
    st.subheader("Geometry")

    col1, col2, col3 = st.columns(3)
    col1.metric("Volume", round(features["Volume"], 2))
    col2.metric("Surface Area", round(features["Surface Area"], 2))
    col3.metric("Complexity", round(features["Complexity"], 2))

    col4, col5, col6 = st.columns(3)
    col4.metric("Flatness", round(features["Flatness"], 4))
    col5.metric("Curvature", round(features["Curvature"], 4))
    col6.metric("Hole", "Yes" if features["Has Hole"] else "No")

    st.divider()

    # ML
    st.subheader("AI Prediction")

    time_pred, op_pred = ml_predict(features)

    if time_pred:
        st.write(f"Estimated Time: **{round(time_pred,2)} min**")
        st.write(f"Operation: **{op_pred}**")

    st.divider()

    # PROCESS PLAN
    st.subheader("Process Plan")

    plan = generate_process_plan(features, op_pred, finish)

    st.markdown(f"<div class='highlight'>{' → '.join(plan)}</div>", unsafe_allow_html=True)

    for i, step in enumerate(plan, 1):
        st.write(f"{i}. {step}")

    st.divider()

    # TOOLS
    st.subheader("Tools")

    tools = select_tools(plan)
    st.write(", ".join(tools))

    st.divider()

    # CNC
    st.subheader("CNC Code")

    gcode = generate_gcode(plan, material, finish)
    gcode_text = "\n".join(gcode)

    st.code(gcode_text, language="gcode")

    st.download_button("Download CNC Code", gcode_text, "gcode.txt")

    st.divider()

    # REPORT
    st.subheader("Machining Report")

    report = f"""
CAD2CAM AI REPORT

Material: {material}
Finish: {finish}

Geometry:
Volume: {features['Volume']}
Surface Area: {features['Surface Area']}
Complexity: {features['Complexity']}

Process Plan:
{chr(10).join(plan)}

Tools:
{', '.join(tools)}

GCODE:
{gcode_text}
"""

    st.text_area("Report Preview", report, height=250)

    st.download_button("Download Report", report, "cad2cam_report.txt")

else:
    st.info("Upload an STL file to begin")
