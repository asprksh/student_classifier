import streamlit as st
import torch
import torch.nn as nn

# ---------------- Model ----------------
class StudentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x

# ---------------- Normalization Values ----------------
# Training data ke min/max
hours_min = 0.5
hours_max = 10.0

attendance_min = 0.0
attendance_max = 100.0

# ---------------- Load Model ----------------
model = StudentClassifier()
model.load_state_dict(torch.load("student_classifier_pass_fail.pth", map_location="cpu"))
model.eval()

# ---------------- UI ----------------
st.title("Student Pass/Fail Predictor")

study_hours = st.slider(
    "Study Hours per Day",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

attendance = st.slider(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

if st.button("Predict"):

    # Normalize exactly same as training
    hours_norm = (study_hours - hours_min) / (hours_max - hours_min)
    attendance_norm = (attendance - attendance_min) / (attendance_max - attendance_min)

    x = torch.tensor(
        [[hours_norm, attendance_norm]],
        dtype=torch.float32
    )

    with torch.no_grad():
        output = model(x)
        probability = torch.sigmoid(output).item()

    if probability >= 0.5:
        st.success(f"PASS")
    else:
        st.error(f"FAIL")

    st.write(f"Pass Probability: {probability * 100 :.4f}")
