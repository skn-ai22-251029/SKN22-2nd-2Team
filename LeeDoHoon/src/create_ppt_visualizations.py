"""
PPT용 시각화 자료 생성 스크립트
- churn 위험 패턴별 이탈율 분석
- 결제 패턴별 이탈율 분석
- 파생 컬럼 및 feature dropping 실험 결과 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/kkbox_train_feature_v4.parquet"
OUTPUT_DIR = PROJECT_ROOT / "LeeDoHoon/outputs/ppt_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """데이터 로드"""
    data_path = DATA_PATH
    print(f"Loading data from {data_path}...")
    
    if not data_path.exists():
        # 대체 경로 시도
        alt_path = PROJECT_ROOT / "LeeDoHoon" / "kkbox_train_feature_v4.parquet"
        if alt_path.exists():
            print(f"  Using alternative path: {alt_path}")
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"  Parquet read error: {e}")
        print("  Trying with engine='fastparquet'...")
        try:
            df = pd.read_parquet(data_path, engine='fastparquet')
        except Exception as e2:
            print(f"  Fastparquet also failed: {e2}")
            print("  Trying to read with pyarrow and ignore errors...")
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(data_path, use_pandas_metadata=True)
                df = table.to_pandas()
            except Exception as e3:
                print(f"  PyArrow also failed: {e3}")
                raise
    
    # 샘플링 (메모리 절약 및 빠른 처리)
    if len(df) > 100000:
        print(f"  Sampling from {len(df)} to 100000 rows...")
        df = df.sample(n=100000, random_state=719)
    
    print(f"Data loaded: {df.shape}")
    return df

def calculate_churn_rate(df, group_col, value_col=None):
    """그룹별 이탈율 계산"""
    if value_col is None:
        result = df.groupby(group_col).agg({
            'is_churn': ['sum', 'count', 'mean']
        }).reset_index()
        result.columns = [group_col, 'churn_count', 'total_count', 'churn_rate']
    else:
        result = df.groupby([group_col, value_col]).agg({
            'is_churn': ['sum', 'count', 'mean']
        }).reset_index()
        result.columns = [group_col, value_col, 'churn_count', 'total_count', 'churn_rate']
    return result

# ============================================
# 1. 위험 패턴별 평균 대비 이탈률 증가
# ============================================
def visualize_risk_pattern_churn_increase(df):
    """위험 패턴별 평균 대비 이탈률 증가 시각화"""
    print("\n[1] 위험 패턴별 평균 대비 이탈률 증가 시각화 생성 중...")
    
    overall_churn_rate = df['is_churn'].mean()
    
    # 위험 패턴 정의
    risk_patterns = []
    
    # 1. 활동 급감 패턴 (active_decay_rate)
    df['risk_active_decay'] = pd.cut(
        df['active_decay_rate'], 
        bins=[0, 0.3, 0.6, 1.0, float('inf')],
        labels=['매우 위험', '위험', '주의', '정상']
    )
    pattern1 = calculate_churn_rate(df, 'risk_active_decay')
    pattern1['pattern_name'] = '활동 급감율'
    pattern1['pattern_value'] = pattern1['risk_active_decay']
    risk_patterns.append(pattern1[['pattern_name', 'pattern_value', 'churn_rate']])
    
    # 2. 청취 시간 감소 패턴 (listening_time_velocity)
    df['risk_listening_velocity'] = pd.cut(
        df['listening_time_velocity'],
        bins=[float('-inf'), -5000, -1000, 0, float('inf')],
        labels=['매우 위험', '위험', '주의', '정상']
    )
    pattern2 = calculate_churn_rate(df, 'risk_listening_velocity')
    pattern2['pattern_name'] = '청취 시간 감소'
    pattern2['pattern_value'] = pattern2['risk_listening_velocity']
    risk_patterns.append(pattern2[['pattern_name', 'pattern_value', 'churn_rate']])
    
    # 3. 스킵 비율 증가 패턴 (skip_passion_index)
    df['risk_skip'] = pd.cut(
        df['skip_passion_index'],
        bins=[0, 0.3, 0.6, 1.0, float('inf')],
        labels=['정상', '주의', '위험', '매우 위험']
    )
    pattern3 = calculate_churn_rate(df, 'risk_skip')
    pattern3['pattern_name'] = '스킵 비율'
    pattern3['pattern_value'] = pattern3['risk_skip']
    risk_patterns.append(pattern3[['pattern_name', 'pattern_value', 'churn_rate']])
    
    # 4. 결제 수단 위험도
    if 'last_payment_method' in df.columns:
        pattern4 = calculate_churn_rate(df, 'last_payment_method')
        pattern4['pattern_name'] = '결제 수단'
        pattern4['pattern_value'] = pattern4['last_payment_method']
        risk_patterns.append(pattern4[['pattern_name', 'pattern_value', 'churn_rate']])
    
    # 통합 데이터프레임
    all_patterns = pd.concat(risk_patterns, ignore_index=True)
    all_patterns['increase_rate'] = (all_patterns['churn_rate'] / overall_churn_rate - 1) * 100
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('위험 패턴별 평균 대비 이탈률 증가', fontsize=18, fontweight='bold', y=0.995)
    
    patterns = all_patterns['pattern_name'].unique()
    colors = ['#FF6B6B', '#FFA07A', '#FFD700', '#98D8C8']
    
    for idx, pattern in enumerate(patterns):
        ax = axes[idx // 2, idx % 2]
        pattern_data = all_patterns[all_patterns['pattern_name'] == pattern].sort_values('churn_rate', ascending=False)
        
        bars = ax.barh(
            pattern_data['pattern_value'].astype(str),
            pattern_data['increase_rate'],
            color=colors[idx % len(colors)],
            alpha=0.8,
            edgecolor='white',
            linewidth=2
        )
        
        # 평균선 표시
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='전체 평균')
        
        # 값 표시
        for i, (bar, rate) in enumerate(zip(bars, pattern_data['increase_rate'])):
            ax.text(
                rate + (1 if rate >= 0 else -1),
                i,
                f'+{rate:.1f}%' if rate >= 0 else f'{rate:.1f}%',
                va='center',
                fontsize=11,
                fontweight='bold'
            )
        
        ax.set_xlabel('평균 대비 이탈률 증가율 (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{pattern}', fontsize=14, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.legend()
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "01_risk_pattern_churn_increase.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] 저장: {output_path}")
    plt.close()

# ============================================
# 2. 위험 패턴별 이탈율과 전체 평균 비교
# ============================================
def visualize_risk_pattern_vs_average(df):
    """위험 패턴별 이탈율과 전체 평균 비교"""
    print("\n[2] 위험 패턴별 이탈율과 전체 평균 비교 시각화 생성 중...")
    
    overall_churn_rate = df['is_churn'].mean()
    
    # 주요 위험 패턴 정의
    patterns_data = []
    
    # 활동 급감율
    df['active_decay_cat'] = pd.cut(
        df['active_decay_rate'],
        bins=[0, 0.3, 0.6, 1.0, float('inf')],
        labels=['매우 위험\n(<0.3)', '위험\n(0.3-0.6)', '주의\n(0.6-1.0)', '정상\n(>1.0)']
    )
    active_decay = calculate_churn_rate(df, 'active_decay_cat')
    active_decay['pattern'] = '활동 급감율'
    patterns_data.append(active_decay)
    
    # 청취 시간 감소
    df['velocity_cat'] = pd.cut(
        df['listening_time_velocity'],
        bins=[float('-inf'), -5000, -1000, 0, float('inf')],
        labels=['매우 위험\n(<-5000)', '위험\n(-5000~-1000)', '주의\n(-1000~0)', '정상\n(>0)']
    )
    velocity = calculate_churn_rate(df, 'velocity_cat')
    velocity['pattern'] = '청취 시간 감소'
    patterns_data.append(velocity)
    
    # 스킵 비율
    df['skip_cat'] = pd.cut(
        df['skip_passion_index'],
        bins=[0, 0.3, 0.6, 1.0, float('inf')],
        labels=['정상\n(<0.3)', '주의\n(0.3-0.6)', '위험\n(0.6-1.0)', '매우 위험\n(>1.0)']
    )
    skip = calculate_churn_rate(df, 'skip_cat')
    skip['pattern'] = '스킵 비율'
    patterns_data.append(skip)
    
    # 통합
    all_data = pd.concat(patterns_data, ignore_index=True)
    
    # 시각화
    fig, ax = plt.subplots(figsize=(14, 8))
    
    patterns = all_data['pattern'].unique()
    x_pos = np.arange(len(patterns))
    width = 0.2
    
    for i, pattern in enumerate(patterns):
        pattern_subset = all_data[all_data['pattern'] == pattern].sort_values('churn_rate', ascending=False)
        categories = pattern_subset.iloc[:, 0].astype(str).values
        churn_rates = pattern_subset['churn_rate'].values * 100
        
        x = x_pos[i] + np.arange(len(categories)) * width - width * (len(categories) - 1) / 2
        
        bars = ax.bar(x, churn_rates, width, label=pattern, alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # 값 표시
        for bar, rate in zip(bars, churn_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 전체 평균선
    ax.axhline(y=overall_churn_rate * 100, color='red', linestyle='--', linewidth=2.5, 
              label=f'전체 평균 ({overall_churn_rate*100:.1f}%)', zorder=0)
    ax.text(len(patterns) - 0.5, overall_churn_rate * 100 + 1, 
           f'전체 평균: {overall_churn_rate*100:.1f}%',
           fontsize=11, fontweight='bold', color='red')
    
    ax.set_xlabel('위험 패턴', fontsize=13, fontweight='bold')
    ax.set_ylabel('이탈율 (%)', fontsize=13, fontweight='bold')
    ax.set_title('위험 패턴별 이탈율과 전체 평균 비교', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(patterns)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "02_risk_pattern_vs_average.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] 저장: {output_path}")
    plt.close()

# ============================================
# 3. 사용 강도별 이탈율 변화
# ============================================
def visualize_usage_intensity_churn(df):
    """사용 강도별 이탈율 변화 시각화"""
    print("\n[3] 사용 강도별 이탈율 변화 시각화 생성 중...")
    
    # 사용 강도 지표들
    usage_metrics = {
        '활동 일수 (W7)': 'num_days_active_w7',
        '일평균 청취 시간 (W7)': 'avg_secs_per_day_w7',
        '일평균 재생 곡 수 (W7)': 'avg_songs_per_day_w7',
        '총 청취 시간 (W7)': 'total_secs_w7'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('사용 강도별 이탈율 변화', fontsize=18, fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    for idx, (metric_name, metric_col) in enumerate(usage_metrics.items()):
        if metric_col not in df.columns:
            continue
            
        ax = axes[idx]
        
        # 5분위수로 구분
        try:
            df['usage_quintile'] = pd.qcut(
                df[metric_col],
                q=5,
                labels=['매우 낮음', '낮음', '보통', '높음', '매우 높음'],
                duplicates='drop'
            )
        except ValueError:
            # 중복값이 많아 bins가 적게 생성된 경우, labels 수를 조정
            df['usage_quintile'] = pd.qcut(
                df[metric_col],
                q=5,
                duplicates='drop'
            )
            # labels를 동적으로 생성
            unique_bins = df['usage_quintile'].cat.categories
            labels_map = ['매우 낮음', '낮음', '보통', '높음', '매우 높음']
            df['usage_quintile'] = df['usage_quintile'].cat.rename_categories(
                labels_map[:len(unique_bins)]
            )
        
        usage_churn = calculate_churn_rate(df, 'usage_quintile')
        usage_churn = usage_churn.sort_values('churn_rate')
        
        # 막대 그래프
        colors = ['#FF6B6B', '#FFA07A', '#FFD700', '#98D8C8', '#6BCB77']
        bars = ax.barh(
            usage_churn['usage_quintile'].astype(str),
            usage_churn['churn_rate'] * 100,
            color=colors,
            alpha=0.8,
            edgecolor='white',
            linewidth=2
        )
        
        # 값 표시
        for bar, rate in zip(bars, usage_churn['churn_rate'] * 100):
            ax.text(
                rate + 0.5,
                bar.get_y() + bar.get_height()/2,
                f'{rate:.1f}%',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_xlabel('이탈율 (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "03_usage_intensity_churn.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] 저장: {output_path}")
    plt.close()

# ============================================
# 4. 실제 결제금액의 차이에 따른 이탈율
# ============================================
def visualize_payment_amount_churn(df):
    """실제 결제금액의 차이에 따른 이탈율 시각화"""
    print("\n[4] 실제 결제금액의 차이에 따른 이탈율 시각화 생성 중...")
    
    if 'total_amount_paid' not in df.columns or 'avg_amount_per_payment' not in df.columns:
        print("  [WARNING] 결제금액 컬럼이 없어 건너뜁니다.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('결제금액별 이탈율 분석', fontsize=18, fontweight='bold', y=1.02)
    
    # 1. 총 결제금액별
    ax1 = axes[0]
    try:
        df['total_amount_cat'] = pd.qcut(
            df['total_amount_paid'],
            q=5,
            duplicates='drop'
        )
    except ValueError:
        df['total_amount_cat'] = pd.qcut(
            df['total_amount_paid'],
            q=5,
            duplicates='drop'
        )
    
    # 구간별 실제 금액 범위 추출
    total_amount_churn = calculate_churn_rate(df, 'total_amount_cat')
    
    # Interval을 문자열로 변환 (금액 범위 표시)
    max_total = df['total_amount_paid'].max()
    def format_interval(interval):
        if pd.isna(interval):
            return "N/A"
        try:
            left = float(interval.left)
            right = float(interval.right)
        except:
            # Interval 객체가 아닌 경우
            return str(interval)
        
        # 소수점 제거 및 천 단위 구분
        if left < 1:
            return f"0원\n이하"
        elif right == float('inf') or right > max_total * 0.99:
            return f"{int(left):,}원\n이상"
        else:
            return f"{int(left):,}원\n~ {int(right):,}원"
    
    total_amount_churn['amount_range'] = total_amount_churn['total_amount_cat'].apply(format_interval)
    # 금액 기준 오름차순 정렬
    total_amount_churn['interval_left'] = total_amount_churn['total_amount_cat'].apply(
        lambda x: float(x.left) if pd.notna(x) else 0
    )
    total_amount_churn = total_amount_churn.sort_values('interval_left', ascending=True)
    
    bars1 = ax1.bar(
        range(len(total_amount_churn)),
        total_amount_churn['churn_rate'] * 100,
        color=['#FF6B6B', '#FFA07A', '#FFD700', '#98D8C8', '#6BCB77'][:len(total_amount_churn)],
        alpha=0.8,
        edgecolor='white',
        linewidth=2
    )
    
    for bar, rate in zip(bars1, total_amount_churn['churn_rate'] * 100):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_xticks(range(len(total_amount_churn)))
    ax1.set_xticklabels(total_amount_churn['amount_range'], rotation=45, ha='right', fontsize=10)
    ax1.set_xlabel('총 결제금액 구간', fontsize=12, fontweight='bold')
    ax1.set_ylabel('이탈율 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('총 결제금액별 이탈율', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 2. 평균 결제금액별
    ax2 = axes[1]
    try:
        df['avg_amount_cat'] = pd.qcut(
            df['avg_amount_per_payment'],
            q=5,
            duplicates='drop'
        )
    except ValueError:
        df['avg_amount_cat'] = pd.qcut(
            df['avg_amount_per_payment'],
            q=5,
            duplicates='drop'
        )
    
    # 구간별 실제 금액 범위 추출
    avg_amount_churn = calculate_churn_rate(df, 'avg_amount_cat')
    
    # Interval을 문자열로 변환 (금액 범위 표시)
    max_avg = df['avg_amount_per_payment'].max()
    def format_interval_avg(interval):
        if pd.isna(interval):
            return "N/A"
        try:
            left = float(interval.left)
            right = float(interval.right)
        except:
            # Interval 객체가 아닌 경우
            return str(interval)
        
        # 소수점 제거 및 천 단위 구분
        if left < 1:
            return f"0원\n이하"
        elif right == float('inf') or right > max_avg * 0.99:
            return f"{int(left):,}원\n이상"
        else:
            return f"{int(left):,}원\n~ {int(right):,}원"
    
    avg_amount_churn['amount_range'] = avg_amount_churn['avg_amount_cat'].apply(format_interval_avg)
    # 금액 기준 오름차순 정렬
    avg_amount_churn['interval_left'] = avg_amount_churn['avg_amount_cat'].apply(
        lambda x: float(x.left) if pd.notna(x) else 0
    )
    avg_amount_churn = avg_amount_churn.sort_values('interval_left', ascending=True)
    
    bars2 = ax2.bar(
        range(len(avg_amount_churn)),
        avg_amount_churn['churn_rate'] * 100,
        color=['#FF6B6B', '#FFA07A', '#FFD700', '#98D8C8', '#6BCB77'][:len(avg_amount_churn)],
        alpha=0.8,
        edgecolor='white',
        linewidth=2
    )
    
    for bar, rate in zip(bars2, avg_amount_churn['churn_rate'] * 100):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xticks(range(len(avg_amount_churn)))
    ax2.set_xticklabels(avg_amount_churn['amount_range'], rotation=45, ha='right', fontsize=10)
    ax2.set_xlabel('평균 결제금액 구간', fontsize=12, fontweight='bold')
    ax2.set_ylabel('이탈율 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('평균 결제금액별 이탈율', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "04_payment_amount_churn.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] 저장: {output_path}")
    plt.close()

# ============================================
# 5. 파생 컬럼 생성 효과 시각화
# ============================================
def visualize_derived_features_impact(df):
    """파생 컬럼 생성 효과 시각화"""
    print("\n[5] 파생 컬럼 생성 효과 시각화 생성 중...")
    
    derived_features = {
        'active_decay_rate': '활동 급감율',
        'listening_time_velocity': '청취 시간 가속도',
        'skip_passion_index': '스킵 열정도',
        'discovery_index': '탐색 지수',
        'last_active_gap': '마지막 활동 간격'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('파생 컬럼별 이탈 위험도 분포', fontsize=18, fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    for idx, (feat_col, feat_name) in enumerate(derived_features.items()):
        if feat_col not in df.columns:
            axes[idx].axis('off')
            continue
            
        ax = axes[idx]
        
        # 이탈/유지 그룹별 분포
        churn_data = df[df['is_churn'] == 1][feat_col].dropna()
        retain_data = df[df['is_churn'] == 0][feat_col].dropna()
        
        # 이상치 제거 (상위 1%)
        if len(churn_data) > 0:
            churn_upper = churn_data.quantile(0.99)
            churn_lower = churn_data.quantile(0.01)
            churn_data = churn_data[(churn_data >= churn_lower) & (churn_data <= churn_upper)]
        
        if len(retain_data) > 0:
            retain_upper = retain_data.quantile(0.99)
            retain_lower = retain_data.quantile(0.01)
            retain_data = retain_data[(retain_data >= retain_lower) & (retain_data <= retain_upper)]
        
        ax.hist(retain_data, bins=30, alpha=0.6, label='유지', color='#6BCB77', edgecolor='white')
        ax.hist(churn_data, bins=30, alpha=0.6, label='이탈', color='#FF6B6B', edgecolor='white')
        
        ax.set_xlabel(feat_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('빈도', fontsize=11, fontweight='bold')
        ax.set_title(f'{feat_name}', fontsize=12, fontweight='bold', pad=10)
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')
    
    # 마지막 subplot 제거
    axes[-1].axis('off')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "05_derived_features_impact.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] 저장: {output_path}")
    plt.close()

# ============================================
# 6. Feature Dropping 실험 결과 시각화
# ============================================
def visualize_feature_dropping_experiment():
    """Feature Dropping 실험 결과 시각화"""
    print("\n[6] Feature Dropping 실험 결과 시각화 생성 중...")
    
    # V4 vs V5.2 비교 (실제 모델 메타데이터 기반)
    # V5.2는 Data Leakage 피처를 제거했지만, 다른 피처들이 포함되어 있음
    experiments = {
        'V4 (Baseline)': {
            'features': 74,
            'dropped': 0,
            'description': '전체 피처 사용\n(Data Leakage 포함)'
        },
        'V5.2 (Safe)': {
            'features': 84,
            'dropped': 6,
            'description': 'Data Leakage 피처 제거\n(실전 적용 가능)'
        }
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Dropping 실험: 모델 신뢰도 향상', fontsize=18, fontweight='bold', y=1.02)
    
    # 1. 피처 수 비교
    ax1 = axes[0]
    models = list(experiments.keys())
    feature_counts = [experiments[m]['features'] for m in models]
    dropped_counts = [experiments[m]['dropped'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, feature_counts, width, label='사용 피처', color='#6BCB77', alpha=0.8)
    bars2 = ax1.bar(x + width/2, dropped_counts, width, label='제거 피처', color='#FF6B6B', alpha=0.8)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('모델 버전', fontsize=12, fontweight='bold')
    ax1.set_ylabel('피처 수', fontsize=12, fontweight='bold')
    ax1.set_title('피처 수 비교', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 2. 제거된 피처 설명
    ax2 = axes[1]
    ax2.axis('off')
    
    # 텍스트를 여러 줄로 나누어 표시
    text_lines = [
        "제거된 피처 (Data Leakage 방지):",
        "",
        "• days_since_last_payment",
        "  → 직접적인 이탈 상태 정보 포함",
        "",
        "• days_since_last_cancel",
        "  → 이탈 의도 직접 반영",
        "",
        "• is_auto_renew_last",
        "  → 현재 구독 상태 정보",
        "",
        "• last_plan_days",
        "  → 결제 타이밍 정보",
        "",
        "• payment_count_last_30d/90d",
        "  → 최근 결제 상태 지표",
        "",
        "→ 모델의 실전 적용 가능성 향상",
        "→ 신뢰도 있는 예측 결과 도출"
    ]
    
    # 텍스트를 y 위치에 맞춰 배치
    y_start = 0.95
    y_step = 0.045
    
    for i, line in enumerate(text_lines):
        y_pos = y_start - (i * y_step)
        if line.startswith("제거된"):
            ax2.text(0.05, y_pos, line, fontsize=14, fontweight='bold',
                    transform=ax2.transAxes, verticalalignment='top')
        elif line.startswith("•"):
            ax2.text(0.08, y_pos, line, fontsize=11, fontweight='bold',
                    transform=ax2.transAxes, verticalalignment='top')
        elif line.startswith("  →") or line.startswith("→"):
            ax2.text(0.12, y_pos, line, fontsize=10,
                    transform=ax2.transAxes, verticalalignment='top',
                    style='italic', color='#555555')
        else:
            ax2.text(0.05, y_pos, line, fontsize=10,
                    transform=ax2.transAxes, verticalalignment='top')
    
    # 배경 박스 추가
    from matplotlib.patches import FancyBboxPatch
    fancy_box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                               boxstyle="round,pad=0.01",
                               transform=ax2.transAxes,
                               facecolor='#FFF9E6', edgecolor='#D4A574',
                               linewidth=2, alpha=0.8, zorder=0)
    ax2.add_patch(fancy_box)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "06_feature_dropping_experiment.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] 저장: {output_path}")
    plt.close()

# ============================================
# 7. 종합 위험도 매트릭스
# ============================================
def visualize_comprehensive_risk_matrix(df):
    """종합 위험도 매트릭스 시각화"""
    print("\n[7] 종합 위험도 매트릭스 시각화 생성 중...")
    
    # 위험도 점수 계산 (간단한 버전)
    df['risk_score'] = 0
    
    # 활동 급감
    df.loc[df['active_decay_rate'] < 0.3, 'risk_score'] += 3
    df.loc[(df['active_decay_rate'] >= 0.3) & (df['active_decay_rate'] < 0.6), 'risk_score'] += 2
    df.loc[(df['active_decay_rate'] >= 0.6) & (df['active_decay_rate'] < 1.0), 'risk_score'] += 1
    
    # 청취 시간 감소
    df.loc[df['listening_time_velocity'] < -5000, 'risk_score'] += 3
    df.loc[(df['listening_time_velocity'] >= -5000) & (df['listening_time_velocity'] < -1000), 'risk_score'] += 2
    df.loc[(df['listening_time_velocity'] >= -1000) & (df['listening_time_velocity'] < 0), 'risk_score'] += 1
    
    # 스킵 비율
    df.loc[df['skip_passion_index'] > 1.0, 'risk_score'] += 3
    df.loc[(df['skip_passion_index'] > 0.6) & (df['skip_passion_index'] <= 1.0), 'risk_score'] += 2
    df.loc[(df['skip_passion_index'] > 0.3) & (df['skip_passion_index'] <= 0.6), 'risk_score'] += 1
    
    # 위험도 그룹
    df['risk_group'] = pd.cut(
        df['risk_score'],
        bins=[0, 2, 4, 6, 10],
        labels=['낮음', '보통', '높음', '매우 높음']
    )
    
    # 그룹별 이탈율
    risk_churn = calculate_churn_rate(df, 'risk_group')
    risk_churn = risk_churn.sort_values('churn_rate', ascending=False)
    
    # 시각화
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#6BCB77', '#FFD700', '#FFA07A', '#FF6B6B']
    bars = ax.barh(
        risk_churn['risk_group'].astype(str),
        risk_churn['churn_rate'] * 100,
        color=colors,
        alpha=0.8,
        edgecolor='white',
        linewidth=2.5
    )
    
    # 값 표시
    for bar, rate, count in zip(bars, risk_churn['churn_rate'] * 100, risk_churn['total_count']):
        ax.text(
            rate + 0.5,
            bar.get_y() + bar.get_height()/2,
            f'{rate:.1f}% (n={count:,})',
            va='center',
            fontsize=12,
            fontweight='bold'
        )
    
    # 전체 평균선
    overall_churn = df['is_churn'].mean() * 100
    ax.axvline(x=overall_churn, color='black', linestyle='--', linewidth=2.5, 
              label=f'전체 평균 ({overall_churn:.1f}%)', zorder=0)
    
    ax.set_xlabel('이탈율 (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('위험도 그룹', fontsize=14, fontweight='bold')
    ax.set_title('종합 위험도 매트릭스별 이탈율', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "07_comprehensive_risk_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] 저장: {output_path}")
    plt.close()

# ============================================
# 메인 실행
# ============================================
def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("PPT용 시각화 자료 생성 시작")
    print("=" * 60)
    
    # 데이터 로드
    df = load_data()
    
    # 시각화 생성
    try:
        visualize_risk_pattern_churn_increase(df)
        visualize_risk_pattern_vs_average(df)
        visualize_usage_intensity_churn(df)
        visualize_payment_amount_churn(df)
        visualize_derived_features_impact(df)
        visualize_feature_dropping_experiment()
        visualize_comprehensive_risk_matrix(df)
        
        print("\n" + "=" * 60)
        print("모든 시각화 자료 생성 완료!")
        print(f"출력 디렉토리: {OUTPUT_DIR}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

