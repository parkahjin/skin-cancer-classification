# ═══════════════════════════════════════════════════════════
# 🌐 Streamlit 대시보드 - 피부암 분류
# 양성/악성 병변 예측
# ═══════════════════════════════════════════════════════════

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 페이지 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="피부암 분류 AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 전처리 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preprocess_image(img_path, img_size=224):
    """학습 시와 동일한 전처리"""
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    
    # ImageNet 정규화
    MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
    img = img.astype(np.float32)
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - MEAN[i]) / STD[i]
    
    return img

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource
def load_model():
    """모델 로드 (경로 자동 탐색)"""
    current_dir = os.getcwd()
    
    # 가능한 경로들
    possible_paths = [
        'final_model_resnet50.keras',
        './final_model_resnet50.keras',
        os.path.join(current_dir, 'final_model_resnet50.keras'),
    ]
    
    # 파일 찾기
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        st.error(f"❌ 모델 파일을 찾을 수 없습니다!")
        st.info(f"**현재 디렉토리:** `{current_dir}`")
        st.info("**파일 목록:**")
        st.code("\n".join(os.listdir(current_dir)))
        st.warning("""
        **해결 방법:**
        1. `final_model_resnet50.keras` 파일을 현재 디렉토리에 복사하세요
        2. Streamlit을 재실행하세요: `streamlit run streamlit_app.py`
        """)
        st.stop()
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ 모델 로드 실패: {e}")
        st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 사이드바
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.sidebar.title("🔬 피부암 분류 AI")
st.sidebar.markdown("---")

# 페이지 선택 (드롭다운)
page = st.sidebar.selectbox(
    "📂 메뉴",
    ["📖 서비스 소개", "🩺 AI 예측"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 모델 성능")
st.sidebar.metric("Accuracy", "77.62%")
st.sidebar.metric("Recall", "81.84%", delta="악성 검출")
st.sidebar.metric("Precision", "75.47%")
st.sidebar.metric("AUC", "0.8585")

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ 이 AI는 보조 도구입니다")
st.sidebar.caption("반드시 전문의 진료를 받으세요")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 페이지 1: 서비스 소개
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if page == "📖 서비스 소개":
    
    st.title("🔬 피부암 Binary Classification")
    st.markdown("### ResNet50 기반 피부 병변 양성/악성 분류 AI")
    st.markdown("---")
    
    # 프로젝트 개요
    st.header("📋 프로젝트 개요")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 목적
        
        피부경(dermatoscope)으로 촬영된 **피부 병변 이미지**를 분석하여,
        해당 병변이 **양성(Benign)** 인지 **악성(Malignant)** 인지 분류합니다.
        
        ### 💡 배경
        
        - **흑색종(Melanoma)** 조기 발견 시 5년 생존율 **99%**
        - 늦게 발견 시 생존율 **27%**
        - AI를 활용한 빠른 스크리닝으로 **조기 발견** 가능
        - 의료 접근성이 낮은 지역의 **1차 진단 도구**
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 분류 클래스
        
        **🟢 Benign (양성):**
        - nv: 점, 모반
        - bkl: 지루각화증
        - df: 피부섬유종
        - vasc: 혈관 병변
        
        **🔴 Malignant (악성):**
        - mel: 흑색종 ⚠️ 치명적
        - bcc: 기저세포암
        - akiec: 광선각화증
        """)
    
    st.markdown("---")
    
    # 모델 정보
    st.header("🤖 모델 정보")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📐 아키텍처
        
        **ResNet50 Transfer Learning**
        
        ```
        Input (224×224×3)
            ↓
        ResNet50 Base (ImageNet)
            ↓
        GlobalAveragePooling2D
            ↓
        Dense(256) + Dropout(0.5)
            ↓
        Dense(128) + Dropout(0.3)
            ↓
        Dense(1) + Sigmoid
            ↓
        Output (0~1)
        ```
        """)
    
    with col2:
        st.markdown("""
        ### ⚙️ 학습 설정
        
        - **Framework:** TensorFlow 2.15
        - **Learning Rate:** 0.001
        - **Batch Size:** 32
        - **Epochs:** 10
        - **Optimizer:** Adam
        - **Loss:** Binary Crossentropy
        - **Data:** HAM10000 (3,908장)
        """)
    
    st.markdown("---")
    
    # 성능 지표
    st.header("📊 성능 지표 (Validation Set)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Accuracy",
            value="77.62%",
            delta="목표 70% 달성"
        )
    
    with col2:
        st.metric(
            label="Recall",
            value="81.84%",
            delta="악성 검출력 높음"
        )
    
    with col3:
        st.metric(
            label="Precision",
            value="75.47%",
            delta="양성 정확도"
        )
    
    with col4:
        st.metric(
            label="AUC",
            value="0.8585",
            delta="우수한 분류 성능"
        )
    
    st.markdown("---")
    
    # 데이터셋
    st.header("📚 데이터셋")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### HAM10000
        
        - **출처:** Human Against Machine with 10000 training images
        - **총 이미지:** 10,015장
        - **클래스:** 7개 (양성 4개, 악성 3개)
        - **촬영 방법:** 피부경(dermatoscope)
        """)
    
    with col2:
        st.markdown("""
        ### 사용 데이터
        
        - **학습 데이터:** 3,126장 (80%)
        - **검증 데이터:** 782장 (20%)
        - **균형 샘플링:** Benign 1,954 vs Malignant 1,954
        - **총 샘플:** 3,908장
        """)
    
    st.markdown("---")
    
    # 주의사항
    st.header("⚠️ 중요 안내")
    
    st.warning("""
    ### 🚨 의료 면책 조항
    
    **이 AI는 보조 도구일 뿐입니다:**
    
    - ❌ **의사를 대체할 수 없습니다**
    - ❌ **최종 진단은 전문의가 수행해야 합니다**
    - ❌ **조직검사 없이 치료 결정 불가**
    - ✅ **의심 병변 스크리닝 용도로만 사용**
    
    **반드시 전문의 진료를 받으세요!**
    """)
    
    st.markdown("---")
    
    # 기술 스택
    st.header("🛠️ 기술 스택")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Deep Learning**
        - TensorFlow 2.15
        - Keras
        - ResNet50
        """)
    
    with col2:
        st.markdown("""
        **Data Processing**
        - OpenCV
        - NumPy
        - Pandas
        """)
    
    with col3:
        st.markdown("""
        **Dashboard**
        - Streamlit
        - Pillow
        - Matplotlib
        """)
    
    st.markdown("---")
    st.success("💡 **다음 단계:** 좌측 메뉴에서 '🩺 AI 예측'을 선택하여 실제 예측을 시도해보세요!")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 페이지 2: AI 예측
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "🩺 AI 예측":
    
    st.title("🩺 AI 예측")
    st.markdown("### 피부 병변 이미지를 업로드하여 양성/악성을 예측합니다")
    st.markdown("---")
    
    # 모델 로드
    try:
        model = load_model()
        st.success("✅ 모델 로드 완료!")
    except Exception as e:
        st.error(f"❌ 모델 로드 실패")
        st.stop()
    
    st.markdown("---")
    
    # 사용 안내
    st.info("""
    **💡 사용 방법:**
    1. 샘플 이미지 4장을 업로드하세요 (양성 2장, 악성 2장)
    2. AI가 각 이미지를 분석합니다
    3. 예측 결과와 정답을 비교 확인하세요
    
    **📌 샘플 이미지:** `streamlit_samples/` 폴더에 4장이 준비되어 있습니다
    """)
    
    # 이미지 업로드
    st.header("📸 이미지 업로드")
    
    uploaded_files = st.file_uploader(
        "이미지를 선택하세요 (최대 4장)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) > 0:
        
        st.markdown("---")
        st.header("🔍 예측 결과")
        
        # 4개씩 표시
        cols = st.columns(4)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[idx % 4]
            
            with col:
                # 원본 이미지 표시
                from PIL import Image
                image = Image.open(uploaded_file)
                st.image(image, caption=f"샘플 {idx+1}", use_container_width=True)
                
                # 임시 파일로 저장
                temp_path = f"temp_{idx}.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 전처리
                preprocessed = preprocess_image(temp_path)
                
                # 예측
                prediction = model.predict(
                    np.expand_dims(preprocessed, axis=0),
                    verbose=0
                )[0][0]
                
                # 결과 표시
                if prediction > 0.5:
                    label = "Malignant"
                    color = "🔴"
                    confidence = float(prediction * 100)  # float 변환!
                else:
                    label = "Benign"
                    color = "🟢"
                    confidence = float((1 - prediction) * 100)  # float 변환!
                
                st.markdown(f"### {color} {label}")
                st.metric("확률", f"{confidence:.1f}%")
                st.progress(confidence / 100)  # 이제 작동!
                
                # 파일명에서 정답 추출
                filename = uploaded_file.name.lower()
                if 'benign' in filename:
                    ground_truth = "Benign"
                elif 'malignant' in filename:
                    ground_truth = "Malignant"
                else:
                    ground_truth = "Unknown"
                
                # 정답 비교
                if ground_truth != "Unknown":
                    if label == ground_truth:
                        st.success(f"✅ 정답: {ground_truth}")
                    else:
                        st.error(f"❌ 정답: {ground_truth}")
        
        st.markdown("---")
        
        # 통계
        st.header("📊 전체 통계")
        
        correct = 0
        total = len(uploaded_files)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            temp_path = f"temp_{idx}.jpg"
            preprocessed = preprocess_image(temp_path)
            prediction = model.predict(
                np.expand_dims(preprocessed, axis=0),
                verbose=0
            )[0][0]
            
            pred_label = "Malignant" if prediction > 0.5 else "Benign"
            
            filename = uploaded_file.name.lower()
            if 'benign' in filename:
                gt_label = "Benign"
            elif 'malignant' in filename:
                gt_label = "Malignant"
            else:
                continue
            
            if pred_label == gt_label:
                correct += 1
        
        accuracy = float((correct / total) * 100) if total > 0 else 0.0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 이미지", total)
        with col2:
            st.metric("정답 개수", correct)
        with col3:
            st.metric("정확도", f"{accuracy:.1f}%")
        
        if accuracy >= 75:
            st.success("🎉 우수한 성능입니다!")
        elif accuracy >= 50:
            st.info("👍 괜찮은 성능입니다!")
        else:
            st.warning("⚠️ 더 많은 테스트가 필요합니다")
    
        # CSV 다운로드
        st.markdown("---")
        st.header("💾 결과 다운로드")
        
        # 결과를 DataFrame으로 변환
        results_data = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            temp_path = f"temp_{idx}.jpg"
            preprocessed = preprocess_image(temp_path)
            prediction = model.predict(
                np.expand_dims(preprocessed, axis=0),
                verbose=0
            )[0][0]
            
            pred_label = "Malignant" if prediction > 0.5 else "Benign"
            confidence = float(prediction * 100 if prediction > 0.5 else (1 - prediction) * 100)
            
            filename = uploaded_file.name.lower()
            if 'benign' in filename:
                gt_label = "Benign"
            elif 'malignant' in filename:
                gt_label = "Malignant"
            else:
                gt_label = "Unknown"
            
            results_data.append({
                '파일명': uploaded_file.name,
                '예측 결과': pred_label,
                '신뢰도(%)': f"{confidence:.2f}",
                '정답': gt_label,
                '정확성': '✅' if pred_label == gt_label else '❌'
            })
        
        # DataFrame 생성
        df_results = pd.DataFrame(results_data)
        
        # 미리보기
        st.dataframe(df_results, use_container_width=True)
        
        # CSV 다운로드 버튼
        csv = df_results.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 결과 CSV 다운로드",
            data=csv,
            file_name="skin_cancer_prediction_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    else:
        st.info("👆 위에서 이미지를 업로드하세요!")