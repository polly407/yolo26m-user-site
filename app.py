from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 已訓練 YOLO26m 模型的位置
# 部署時 best.pt 必須與 app.py 放在同一個資料夾
MODEL_PATH = Path(__file__).parent / "best.pt"

st.set_page_config(
    page_title="YOLO26m 物件偵測",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 YOLO26m 物件偵測系統")
st.write("上傳圖片或使用相機拍照，辨識圖片中的物件。")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型檔案：{MODEL_PATH}"
        )
    return YOLO(str(MODEL_PATH))


# 偵測信心分數門檻
confidence = st.slider(
    "偵測信心分數門檻",
    min_value=0.10,
    max_value=0.95,
    value=0.25,
    step=0.05,
)

# 影像來源
source = st.radio(
    "選擇影像來源",
    ["上傳圖片", "相機拍照"],
    horizontal=True,
)

if source == "上傳圖片":
    uploaded_file = st.file_uploader(
        "請選擇圖片",
        type=["jpg", "jpeg", "png"],
    )
    image_file = uploaded_file
else:
    image_file = st.camera_input("請拍攝要辨識的物件")

if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        st.image(
            image,
            caption="原始圖片",
            use_container_width=True,
        )

        if st.button("開始辨識", type="primary"):
            with st.spinner("YOLO26m 模型辨識中，請稍候..."):
                model = load_model()

                results = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=640,
                    device="cpu",
                    verbose=False,
                )

            result = results[0]
            annotated_image = result.plot()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("辨識結果")
                st.image(
                    annotated_image[:, :, ::-1],
                    caption="物件偵測框",
                    use_container_width=True,
                )

            with col2:
                st.subheader("偵測清單")

                if result.boxes is None or len(result.boxes) == 0:
                    st.info(
                        "沒有偵測到符合信心分數門檻的物件。"
                    )
                else:
                    detections = []

                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        score = float(box.conf.item())
                        class_name = result.names[class_id]

                        detections.append({
                            "物件": class_name,
                            "信心分數": f"{score:.1%}",
                        })

                    st.dataframe(
                        detections,
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.metric(
                        "偵測物件總數",
                        len(detections),
                    )

    except Exception as error:
        st.error(f"辨識時發生錯誤：{error}")


st.caption(
    "本系統使用既有 YOLO26m 模型。辨識結果僅供參考，"
    "請勿單憑模型結果判斷物品是否具有危險性。"
)
