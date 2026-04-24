import streamlit as st
import trimesh
import numpy as np
import joblib
from streamlit_stl import stl_from_file
import tempfile
import os

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="CAD2CAM AI", layout="wide")

# ---------------- UI ---------------- #
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", "Inter", sans-serif;
}
.main {
    background-color: #0E1117;
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

st.markdown("""
<div style='font-size:16px; color:#9CA3AF; margin-top:-10px; margin-bottom:10px;'>
AI-powered system that converts CAD models into machining insights by analyzing geometry,
predicting operations, and generating material-aware CAM plans with CNC output.
</div>
""", unsafe_allow_html=True)

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

# ---------------- HOLE DETECTION ---------------- #
def detect_hole(mesh):
    bbox_vol = np.prod(mesh.bounding_box.extents)
    mesh_vol = mesh.volume if mesh.volume > 0 else 1e-6
    void_ratio = 1 - (mesh_vol / bbox_vol)
    curvature_hint = np.std(mesh.face_normals)
    return (void_ratio > 0.2) or (curvature_hint > 0.3)

# ---------------- FEATURE EXTRACTION ---------------- #
def extract_features(file):
    mesh = trimesh.load(file, file_type='stl', force='mesh')
    mesh.process()

    volume = mesh.volume or 0
    area = mesh.area or 0

    length, width, height = mesh.bounding_box.extents
    is_cyl = abs(length - width) < 0.1 * max(length, width)

    flatness = np.mean(np.abs(mesh.face_normals[:, 2]))
    curvature = np.std(mesh.face_normals)

    has_hole = detect_hole(mesh)

    complexity = area / (volume + 1e-6)
    hole_indicator = 1 if has_hole else 0
    curvature_proxy = curvature * flatness

    return {
        "Volume": volume,
        "Surface Area": area,
        "Complexity": complexity,
        "Hole Indicator": hole_indicator,
        "Flatness": flatness,
        "Curvature Proxy": curvature_proxy,
        "Curvature": curvature,
        "Is Cylindrical": is_cyl,
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
def generate_plan(features, ml_op, finish):

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
        for k in tool_map:
            if k in s:
                tools.append(tool_map[k])

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

# ---------------- MANUFACTURABILITY ---------------- #
def manufacturability(features, material, finish, time_pred):

    issues = []
    cost = 1.0

    if features["Complexity"] > 5:
        issues.append("High complexity → More machining time")
        cost += 0.3

    if features["Has Hole"] and features["Curvature"] > 0.5:
        issues.append("Complex hole geometry → Difficult machining")
        cost += 0.2

    if material == "Steel":
        issues.append("Hard material → Increased tool wear")
        cost += 0.3

    if finish == "Fine":
        issues.append("Fine finish → Requires precision machining")
        cost += 0.25

    if time_pred and time_pred > 30:
        issues.append("Long machining time → Higher cost")
        cost += 0.2

    return issues, round(cost * 100, 2)

# ---------------- MAIN ---------------- #
if uploaded_file:

    st.success("File uploaded successfully")

    # 3D VIEW
    st.subheader("3D Model")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    stl_from_file(path)
    os.remove(path)

    # PROCESS
    features = extract_features(uploaded_file)

    st.divider()

    # GEOMETRY WITH UNITS
    st.subheader("Geometry")

    c1, c2, c3 = st.columns(3)
    c1.metric("Volume (mm³)", round(features["Volume"], 2))
    c2.metric("Surface Area (mm²)", round(features["Surface Area"], 2))
    c3.metric("Complexity (SA/Vol)", round(features["Complexity"], 2))

    c4, c5, c6 = st.columns(3)
    c4.metric("Flatness (norm)", round(features["Flatness"], 4))
    c5.metric("Curvature (std)", round(features["Curvature"], 4))
    c6.metric("Hole Detected", "Yes" if features["Has Hole"] else "No")

    st.divider()

    # ML
    st.subheader("AI Prediction")
    time_pred, op_pred = ml_predict(features)

    if time_pred:
        st.write(f"Estimated Time: **{round(time_pred,2)} min**")
        st.write(f"Operation: **{op_pred}**")

    st.divider()

    # PLAN
    st.subheader("Process Plan")
    plan = generate_plan(features, op_pred, finish)

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

    st.code(gcode_text)
    st.download_button("Download CNC", gcode_text, "gcode.txt")

    st.divider()

    # MANUFACTURABILITY (NEW)
    st.subheader("Manufacturability Analysis")

    issues, cost_index = manufacturability(features, material, finish, time_pred)

    if issues:
        for i in issues:
            st.warning(i)
    else:
        st.success("No major issues detected")

    st.metric("Estimated Cost Index", cost_index)

    if cost_index < 130:
        st.success("Low Manufacturing Cost")
    elif cost_index < 160:
        st.warning("Moderate Manufacturing Cost")
    else:
        st.error("High Manufacturing Cost")

    st.divider()

    # REPORT
    st.subheader("Machining Report")

    report = f"""
CAD2CAM AI REPORT

Material: {material}
Finish: {finish}

Volume: {features['Volume']}
Surface Area: {features['Surface Area']}
Complexity: {features['Complexity']}

Estimated Time: {time_pred}
Operation: {op_pred}

Process Plan:
{chr(10).join(plan)}

Tools:
{', '.join(tools)}

Manufacturability Issues:
{chr(10).join(issues)}

Cost Index: {cost_index}
"""

    st.text_area("Report Preview", report, height=250)
    st.download_button("Download Report", report, "report.txt")

else:
    st.info("Upload an STL file to begin")

