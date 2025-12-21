
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import textwrap
from pathlib import Path
import sys

# Config
st.set_page_config(page_title="Risk Matrix Dashboard", page_icon="📈", layout="wide")

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

# --- Shared Logic ---
@st.cache_data
def load_and_score():
    """Load data and predict with both models to create the matrix"""
    try:
        data_path = project_root / "data/processed/kkbox_train_feature_v4.parquet"
        if not data_path.exists(): return None
        
        # Load sample
        df = pd.read_parquet(data_path).sample(n=2000, random_state=42)
        
        # Load models
        inf_v4 = ModelInference(model_dir=str(model_dir), model_version='v4')
        inf_v5 = ModelInference(model_dir=str(model_dir), model_version='v5.2')
        
        # Predict
        df['score_v4'] = inf_v4.predict(df)
        df['score_v5'] = inf_v5.predict(df)
        
        # Define Segments
        def assign_segment(row):
            v4, v5 = row['score_v4'], row['score_v5']
            if v4 < 0.5 and v5 < 0.5: return '1. 안전 지대 (Safe)'
            elif v4 < 0.5 and v5 >= 0.5: return '2. 주의 지대 (Watch-out)'
            elif v4 >= 0.5 and v5 < 0.5: return '3. 경보 지대 (Warning)'
            else: return '4. 위험 지대 (Danger)'
            
        df['segment'] = df.apply(assign_segment, axis=1)
        return df
    except Exception as e:
        st.error(f"Data scoring error: {e}")
        return None

def main():
    st.title("📈 고객 위험도 매트릭스 (Risk Matrix)")
    st.markdown("**행동 변화(V5.2)와 과거 이력(V4)을 결합하여 유저의 현재 위치를 진단합니다.**")
    
    df = load_and_score()
    if df is None: st.stop()
    
    st.divider()
    
    # 2.1 4-Quadrant Analysis
    col_plot, col_info = st.columns([2, 1])
    
    with col_plot:
        st.subheader("2.1 4분면 위험도 분석 (4-Quadrant Matrix)")
        
        fig = px.scatter(
            df, x='score_v5', y='score_v4',
            color='segment',
            color_discrete_map={
                '1. 안전 지대 (Safe)': '#E8F5E9',   # Green
                '2. 주의 지대 (Watch-out)': '#FFFDE7', # Yellow
                '3. 경보 지대 (Warning)': '#FFF3E0',   # Orange
                '4. 위험 지대 (Danger)': '#FFEBEE'     # Red
            },
            hover_data=['score_v4', 'score_v5'],
            labels={'score_v5': '행동 위험도 (V5.2)', 'score_v4': '결제/이력 위험도 (V4)'},
            category_orders={'segment': ['1. 안전 지대 (Safe)', '2. 주의 지대 (Watch-out)', '3. 경보 지대 (Warning)', '4. 위험 지대 (Danger)']}
        )
        
        # Add Quadrant Lines
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add Labels to Quadrants
        fig.add_annotation(x=0.25, y=0.25, text="Safe", showarrow=False, font=dict(color="green", size=15))
        fig.add_annotation(x=0.75, y=0.25, text="Watch-out", showarrow=False, font=dict(color="orange", size=15))
        fig.add_annotation(x=0.25, y=0.75, text="Warning", showarrow=False, font=dict(color="orange", size=15))
        fig.add_annotation(x=0.75, y=0.75, text="Danger", showarrow=False, font=dict(color="red", size=20, weight="bold"))
        
        fig.update_layout(height=600, showlegend=True, legend_title_text='고객 그룹')
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.subheader("2.2 그룹 세그먼트 정의")
        
        def segment_card(color, title, emoji, detail, strategy):
            html = textwrap.dedent(f"""
                <div style="background-color: {color}; padding: 12px; border-radius: 10px; margin-bottom: 10px; border: 1px solid rgba(0,0,0,0.05);">
                    <h5 style="margin: 0; color: #333;">{emoji} {title}</h5>
                    <p style="font-size: 0.85rem; margin: 5px 0; color: #555;"><strong>상태</strong>: {detail}</p>
                    <p style="font-size: 0.85rem; margin: 0; color: #222;"><strong>전략</strong>: {strategy}</p>
                </div>
            """)
            return html

        st.markdown(segment_card("#E8F5E9", "안전 지대 (Safe)", "✅", "활동 왕성, 결제 안정", "유지(Keep) 및 팬덤 관리"), unsafe_allow_html=True)
        st.markdown(segment_card("#FFFDE7", "주의 지대 (Watch-out)", "🟡", "결제 유지 중이나 활동 급감", "콘텐츠 푸시 (권태기 유저)"), unsafe_allow_html=True)
        st.markdown(segment_card("#FFF3E0", "경보 지대 (Warning)", "🟠", "활동은 있으나 결제 이력 불안", "결제 수단 업데이트 유도"), unsafe_allow_html=True)
        st.markdown(segment_card("#FFEBEE", "위험 지대 (Danger)", "🚨", "활동 전무, 이탈 징후 뚜렷", "강력한 프로모션 (이별 직전)"), unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 💡 분석 요약")
        counts = df['segment'].value_counts()
        total = len(df)
        st.write(f"- 전체 유저 수: {total:,}명")
        st.write(f"- 고위험군(Danger) 비중: {counts.get('4. 위험 지대 (Danger)', 0)/total*100:.1f}%")
        st.write(f"- 관리 필요군(Watch+Warning) 비중: {(counts.get('2. 주의 지대 (Watch-out)', 0) + counts.get('3. 경보 지대 (Warning)', 0))/total*100:.1f}%")

if __name__ == "__main__":
    main()
