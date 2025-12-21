
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import textwrap
from pathlib import Path
import sys

# Config
st.set_page_config(page_title="Marketing Simulator", page_icon="🎮", layout="wide")

# Setup Paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
model_dir = project_root / "03_trained_model"
sys.path.append(str(model_dir))

try:
    from model_inference import ModelInference
except ImportError:
    st.error("ModelInference module not found.")
    st.stop()

# --- Shared Logic (Duplicate for page independence or use common module) ---
@st.cache_data
def load_and_score():
    try:
        data_path = project_root / "data/processed/kkbox_train_feature_v4.parquet"
        if not data_path.exists(): return None
        df = pd.read_parquet(data_path).sample(n=2000, random_state=42)
        inf_v4 = ModelInference(model_dir=str(model_dir), model_version='v4')
        inf_v5 = ModelInference(model_dir=str(model_dir), model_version='v5.2')
        df['score_v4'] = inf_v4.predict(df)
        df['score_v5'] = inf_v5.predict(df)
        
        # Max Risk for Targeting
        df['max_risk'] = df[['score_v4', 'score_v5']].max(axis=1)
        
        def assign_segment(row):
            v4, v5 = row['score_v4'], row['score_v5']
            if v4 < 0.5 and v5 < 0.5: return '안전 지대'
            elif v4 < 0.5 and v5 >= 0.5: return '주의 지대'
            elif v4 >= 0.5 and v5 < 0.5: return '경보 지대'
            else: return '위험 지대'
        df['segment'] = df.apply(assign_segment, axis=1)
        df['user_id'] = [f"U_{20000+i}" for i in range(len(df))]
        return df
    except Exception as e:
        return None

def main():
    st.title("🎮 마케팅 시뮬레이터 (Marketing Simulator)")
    st.markdown("**마케팅 범위를 직접 설정하고 그에 따른 타겟 위치와 전략을 도출합니다.**")
    
    df = load_and_score()
    if df is None: st.stop()
    
    st.divider()
    
    # 3.1 Targeting Slider
    st.subheader("3.1 타겟 범위 설정 (Targeting Slider)")
    top_n = st.slider("이탈 위험군 상위 N% 설정", 1, 100, 20, help="위험도가 높은 유저부터 순차적으로 포함합니다.")
    
    threshold_val = np.percentile(df['max_risk'], 100 - top_n)
    df['is_target'] = df['max_risk'] >= threshold_val
    target_df = df[df['is_target']]
    
    # Real-time metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("선택된 타겟 유저 수", f"{len(target_df):,}명")
    
    seg_counts = target_df['segment'].value_counts()
    primary_seg = seg_counts.index[0] if not seg_counts.empty else "None"
    c2.metric("주요 분포 구역", primary_seg)
    
    avg_risk = target_df['max_risk'].mean()
    c3.metric("타겟 평균 위험도", f"{avg_risk*100:.1f}%")

    st.divider()

    # 3.2 Action Plan & Visualization
    col_plot, col_action = st.columns([1.5, 1])
    
    with col_plot:
        st.subheader("타겟 위치 시각화")
        
        # Plot all but highlight targets
        fig = px.scatter(
            df, x='score_v5', y='score_v4',
            color='is_target',
            color_discrete_map={True: '#FF4B4B', False: '#D3D3D3'},
            opacity=0.6,
            labels={'score_v5': '행동 위험도', 'score_v4': '결제 위험도'},
            title=f"상위 {top_n}% 유저 분포 (Red: Target)"
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.3)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.3)
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col_action:
        st.subheader("3.2 맞춤형 마케팅 처방")
        
        if primary_seg == "위험 지대":
            st.error("🚨 **위태로운 상태 (Danger Focus)**")
            st.markdown(textwrap.dedent("""
                이미 이탈이 거의 확실시되는 그룹이 다수입니다.
                - **권장 전략**: 1개월 무료 쿠폰 발송, 파격적 Win-back 프로모션.
                - **메시지**: "당신을 위한 마지막 혜택, 다시 돌아오세요!"
            """))
        elif primary_seg == "주의 지대":
            st.warning("🟡 **권태기 유저 (Watch-out Focus)**")
            st.markdown(textwrap.dedent("""
                활동이 급격히 줄어든 그룹입니다. (가성비 중심 마케팅 보단 가치 전달 마케팅)
                - **권장 전략**: 신곡 추천 푸시, 플레이리스트 큐레이션.
                - **메시지**: "요즘 유행하는 이 노래, 들어보셨나요?"
            """))
        elif primary_seg == "경보 지대":
            st.info("🟠 **환경 불안 유저 (Warning Focus)**")
            st.markdown(textwrap.dedent("""
                결제 실패나 해지가 우려되는 그룹입니다.
                - **권장 전략**: 결제 수단 자동 갱신 유도, 소액 리워드.
                - **메시지**: "구독이 곧 만료됩니다. 중단 없는 음악 감상을 위해 결제 정보를 확인해 주세요."
            """))
        else:
            st.success("🟢 **안정적 유저 (Safe Focus)**")
            st.markdown("현재 매우 안정적인 유저들입니다. 추가 혜택보다는 현재의 만족도를 유지하는 것이 중요합니다.")

        st.divider()
        st.subheader("📄 데이터 추출")
        csv = target_df[['user_id', 'score_v4', 'score_v5', 'segment']].to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="타겟 유저 리스트(CSV) 다운로드",
            data=csv,
            file_name=f"churn_target_top_{top_n}.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
