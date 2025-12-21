
import streamlit as st
import pandas as pd
import textwrap

st.set_page_config(
    page_title="KKBox Churn Management Center",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 KKBox 구독 이탈 관리 시스템")
    st.subheader("데이터 기반의 분석 체계 및 도메인 개요")
    
    st.divider()
    
    # 1.1 Data Boundary
    st.markdown("### 1.1 데이터 바운더리 (Data Boundary)")
    html_boundary = textwrap.dedent("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 15px; border-left: 5px solid #4a90e2;">
            <ul style="list-style-type: none; padding-left: 0; margin-bottom: 0;">
                <li><strong>📅 기준 시점(T)</strong>: 2017-04-01</li>
                <li><strong>🎧 행동 로그(User Logs)</strong>: 2017-03-01 ~ 2017-03-31 (T 기준 과거 30일 단기 집중 분석)</li>
                <li><strong>💳 거래 이력(Transactions)</strong>: 가입 시점부터 T까지의 전체 이력 (Payment Context 확보)</li>
                <li><strong>🚨 이탈 정의</strong>: 구독 만료 후 30일 이내에 재결제가 발생하지 않은 상태</li>
            </ul>
        </div>
    """)
    st.markdown(html_boundary, unsafe_allow_html=True)
    
    st.divider()
    
    # 1.2 Top-7 Feature Specification
    st.markdown("### 1.2 핵심 변수 명세 (Top-7 Feature Importance)")
    st.markdown("각 모델의 예측에 가장 큰 영향을 미치는 상위 7개 변수입니다.")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 🚑 Track 1: 응급실 모델 (V4)")
        st.caption("금융/결제 상태 중심 - 즉각적인 이탈 징후 포착")
        v4_features = [
            {"지표명": "days_since_last_payment", "비즈니스 의미": "마지막 결제 후 경과일 (가장 강력한 이탈 신호)"},
            {"지표명": "reg_days", "비즈니스 의미": "서비스 가입 기간 (장기 유저일수록 유지 경향)"},
            {"지표명": "is_auto_renew_last", "비즈니스 의미": "마지막 결제의 자동 갱신 여부 (Off일 경우 위험)"},
            {"지표명": "last_payment_method", "비즈니스 의미": "최종 결제 수단 (수단별 이탈률 차이 존재)"},
            {"지표명": "days_since_last_cancel", "비즈니스 의미": "마지막 해지 후 경과일 (최근 해지 이력이 있을 시 위험)"},
            {"지표명": "subscription_months_est", "비즈니스 의미": "실질적 서비스 충성도 기간"},
            {"지표명": "payment_count_last_30d", "비즈니스 의미": "최근 30일 내 결제 성공 횟수"}
        ]
        st.table(pd.DataFrame(v4_features))
        
    with c2:
        st.markdown("#### 🩺 Track 2.5: 건강검진 모델 (V5.2)")
        st.caption("행동 이력 중심 - 잠재적인 권태기/심리적 변화 포착")
        v5_features = [
            {"지표명": "reg_days", "비즈니스 의미": "서비스 가입 기간 (역사적 충성도)"},
            {"지표명": "avg_amount_per_payment", "비즈니스 의미": "1회 평균 결제 금액 (가격 민감도)"},
            {"지표명": "has_ever_cancelled", "비즈니스 의미": "과거 해지 이력 유무 (이탈 경험 데이터)"},
            {"지표명": "subscription_months_est", "비즈니스 의미": "누적 구독 기간"},
            {"지표명": "last_payment_method", "비즈니스 의미": "주요 결제 수단 환경"},
            {"지표명": "total_amount_paid", "비즈니스 의미": "서비스에 지출한 총 누적액 (LTV)"},
            {"지표명": "registered_via", "비즈니스 의미": "최초 가입 경로 (가입 매체별 지속성 차이)"}
        ]
        st.table(pd.DataFrame(v5_features))

    st.divider()
    st.markdown("#### 👈 사이드바를 통해 분석 대시보드로 이동하세요.")

if __name__ == "__main__":
    main()
