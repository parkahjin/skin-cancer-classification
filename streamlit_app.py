# ═══════════════════════════════════════════════════════════
# 🌐 피부암 분류 AI - Google Drive 최종 버전
# ═══════════════════════════════════════════════════════════

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import os
import gdown
from pathlib import Path

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
# Google Drive에서 모델 다운로드 (수정됨)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource
def download_model_from_gdrive():
    """Google Drive에서 모델 다운로드 (최초 1회만)"""
    
    # 절대 경로 사용
    model_path = Path('/tmp/final_model_resnet50.keras')
    
    # 이미 존재하면 스킵
    if model_path.exists():
        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
        if file_size > 10:  # 10MB 이상이면 유효
            st.info(f'✅ 캐시된 모델 사용: {file_size:.1f} MB')
            return str(model_path)
        else:
            # 잘못된 파일 삭제
            model_path.unlink()
    
    # Google Drive 전체 URL 사용 (fuzzy=True)
    gdrive_url = 'https://drive.google.com/file/d/13RsivlToes33FwGINH-CATCPT9lUbudL/view?usp=sharing'
    
    # 다운로드
    with st.spinner('🔄 AI 모델 다운로드 중... (최초 1회, 약 2-3분 소요)'):
        try:
            # fuzzy=True: Google Drive 공유 링크 자동 처리
            st.info('📥 다운로드 시작... (큰 파일이라 시간이 걸립니다)')
            
            output = gdown.download(
                gdrive_url, 
                str(model_path), 
                quiet=False,
                fuzzy=True  # ⭐ 중요!
            )
            
            # 다운로드 확인
            if not model_path.exists():
                raise FileNotFoundError(f"다운로드 실패: 파일이 생성되지 않음")
            
            file_size = model_path.stat().st_size / (1024 * 1024)  # MB
            
            # 파일 크기 검증
            if file_size < 10:
                raise ValueError(f"파일 크기 이상: {file_size:.1f} MB (예상: 80-100 MB)")
            
            st.success(f'✅ 모델 다운로드 완료! ({file_size:.1f} MB)')
            
        except Exception as e:
            st.error(f'❌ 다운로드 실패: {e}')
            
            # 상세 정보
            if model_path.exists():
                file_size = model_path.stat().st_size / (1024 * 1024)
                st.error(f"다운로드된 파일 크기: {file_size:.1f} MB")
                
                # HTML 에러 페이지인지 확인
                with open(model_path, 'rb') as f:
                    first_bytes = f.read(100)
                    if b'<!DOCTYPE' in first_bytes or b'<html' in first_bytes:
                        st.error("❌ HTML 페이지를 다운받았습니다 (실제 파일 아님)")
            
            st.info("""
            **문제 해결:**
            
            1. **Google Drive 링크 재확인:**
               - https://drive.google.com/file/d/13RsivlToes33FwGINH-CATCPT9lUbudL/view
               - 이 링크를 브라우저에서 열어보세요
               - 로그인 없이 다운로드되나요?
            
            2. **공유 설정 확인:**
               - Google Drive에서 파일 우클릭
               - "공유" → "링크가 있는 모든 사용자"
               - 권한: "뷰어"
            
            3. **다른 방법:**
               - Streamlit Cloud 로그 확인
               - 5분 후 페이지 새로고침
            """)
            st.stop()
    
    return str(model_path)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource
def load_model():
    """모델 로드"""
    model_path = download_model_from_gdrive()
    
    try:
        st.info(f'📂 모델 로드 중...')
        model = tf.keras.models.load_model(model_path)
        st.success('✅ 모델 로드 성공!')
        return model
    except Exception as e:
        st.error(f"❌ 모델 로드 실패: {e}")
        st.error(f"경로: {model_path}")
        
        # 파일 정보 출력
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            st.error(f"파일 크기: {size / (1024*1024):.1f} MB")
        
        st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 전처리 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preprocess_image(img_path, img_size=224):
    """학습 시와 동일한 전처리"""
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    
    MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
    img = img.astype(np.float32)
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - MEAN[i]) / STD[i]
    
    return img

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 사이드바
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.sidebar.title("🔬 피부암 분류 AI")
st.sidebar.markdown("---")

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
        st.metric("Accuracy", "77.62%", delta="목표 70% 달성")
    
    with col2:
        st.metric("Recall", "81.84%", delta="악성 검출력 높음")
    
    with col3:
        st.metric("Precision", "75.47%", delta="양성 정확도")
    
    with col4:
        st.metric("AUC", "0.8585", delta="우수한 분류 성능")
    
    st.markdown("---")
    
    # 주의사항
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
    except Exception as e:
        st.error(f"❌ 모델 로드 중 오류 발생")
        st.stop()
    
    st.markdown("---")
    
    # 사용 안내
    st.info("""
    **💡 사용 방법:**
    1. 피부 병변 이미지를 업로드하세요 (최대 4장)
    2. AI가 각 이미지를 분석합니다
    3. 예측 결과를 확인하세요
    4. CSV 파일로 결과를 다운로드할 수 있습니다
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
                temp_path = f"/tmp/temp_{idx}.jpg"
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
                    confidence = float(prediction * 100)
                else:
                    label = "Benign"
                    color = "🟢"
                    confidence = float((1 - prediction) * 100)
                
                st.markdown(f"### {color} {label}")
                st.metric("확률", f"{confidence:.1f}%")
                st.progress(confidence / 100)
                
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
            temp_path = f"/tmp/temp_{idx}.jpg"
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
            temp_path = f"/tmp/temp_{idx}.jpg"
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
