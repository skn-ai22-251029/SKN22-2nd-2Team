# -*- coding: utf-8 -*-
"""
KKBox Churn Prediction - Inference Pipeline
작성자: 이도훈 (LDH)
작성일: 2025-12-19

전처리부터 추론까지 일관된 inference 코드
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import json
import warnings
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# ============================================
# 설정
# ============================================
MODEL_DIR = Path(__file__).parent
MODEL_FILE = MODEL_DIR / "catboost_recall_selected.cbm"
FEATURE_COLS_FILE = MODEL_DIR / "feature_cols.json"
RESULTS_FILE = MODEL_DIR / "recall_selected_results.json"

# 최적 Threshold (recall_selected_results.json에서 로드)
OPTIMAL_THRESHOLD = 0.58  # 기본값, results 파일에서 로드 시 업데이트

# 중복 피처 제거 목록 (학습 시 제거된 피처)
DUPLICATE_FEATURES = [
    "recency_secs_ratio",   # = secs_trend_w7_w30
    "recency_songs_ratio",  # = songs_trend_w7_w30
]


# ============================================
# 모델 로드
# ============================================
class ChurnPredictor:
    """이탈 예측 모델 클래스"""
    
    def __init__(self, model_path: Optional[Path] = None, 
                 feature_cols_path: Optional[Path] = None,
                 results_path: Optional[Path] = None):
        """
        Args:
            model_path: 모델 파일 경로 (.cbm)
            feature_cols_path: 피처 컬럼 목록 파일 경로 (.json)
            results_path: 결과 파일 경로 (.json) - threshold 정보 포함
        """
        self.model_path = model_path or MODEL_FILE
        self.feature_cols_path = feature_cols_path or FEATURE_COLS_FILE
        self.results_path = results_path or RESULTS_FILE
        
        # 모델 및 메타데이터 로드
        self.model = None
        self.feature_cols = None
        self.threshold = OPTIMAL_THRESHOLD
        self.model_metadata = {}
        
        self._load_model()
        self._load_metadata()
    
    def _load_model(self):
        """모델 파일 로드"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        print(f"모델 로드 중: {self.model_path}")
        self.model = CatBoostClassifier()
        self.model.load_model(str(self.model_path))
        print("✓ 모델 로드 완료")
    
    def _load_metadata(self):
        """피처 목록 및 메타데이터 로드"""
        # 피처 컬럼 로드
        if self.feature_cols_path.exists():
            with open(self.feature_cols_path, 'r') as f:
                self.feature_cols = json.load(f)
            print(f"✓ 피처 목록 로드 완료 ({len(self.feature_cols)}개)")
        else:
            raise FileNotFoundError(f"피처 목록 파일을 찾을 수 없습니다: {self.feature_cols_path}")
        
        # 결과 파일에서 threshold 로드
        if self.results_path.exists():
            with open(self.results_path, 'r') as f:
                results = json.load(f)
            self.threshold = results.get('optimal_threshold', OPTIMAL_THRESHOLD)
            self.model_metadata = results
            print(f"✓ 최적 Threshold: {self.threshold:.4f}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        입력 데이터를 모델에 맞는 피처 형식으로 변환
        
        Args:
            df: 입력 데이터프레임 (msno 포함 가능)
        
        Returns:
            모델 입력용 데이터프레임
        """
        # msno, is_churn 제외
        exclude_cols = ["msno", "is_churn"]
        available_cols = [c for c in df.columns if c not in exclude_cols]
        
        # 중복 피처 제거 (학습 시 제거된 피처)
        feature_cols_to_use = [c for c in self.feature_cols if c in available_cols]
        missing_cols = [c for c in self.feature_cols if c not in available_cols]
        
        if missing_cols:
            print(f"⚠ 경고: 다음 피처가 입력 데이터에 없습니다 ({len(missing_cols)}개):")
            for col in missing_cols[:10]:  # 최대 10개만 출력
                print(f"  - {col}")
            if len(missing_cols) > 10:
                print(f"  ... 외 {len(missing_cols) - 10}개")
        
        # 피처 추출
        X = df[feature_cols_to_use].copy()
        
        # 누락된 피처는 0으로 채움 (또는 적절한 기본값)
        for col in missing_cols:
            X[col] = 0
        
        # 피처 순서 정렬 (모델 학습 시 순서와 동일하게)
        X = X[self.feature_cols]
        
        return X
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        이탈 확률 예측
        
        Args:
            df: 입력 데이터프레임
        
        Returns:
            이탈 확률 배열 (shape: [n_samples, 2])
        """
        X = self.prepare_features(df)
        proba = self.model.predict_proba(X)
        return proba
    
    def predict(self, df: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        이탈 여부 예측 (이진 분류)
        
        Args:
            df: 입력 데이터프레임
            threshold: 예측 threshold (None이면 최적 threshold 사용)
        
        Returns:
            이탈 예측 결과 (0: 유지, 1: 이탈)
        """
        if threshold is None:
            threshold = self.threshold
        
        proba = self.predict_proba(df)
        churn_proba = proba[:, 1]
        predictions = (churn_proba >= threshold).astype(int)
        
        return predictions
    
    def predict_with_metadata(self, df: pd.DataFrame, 
                              threshold: Optional[float] = None) -> pd.DataFrame:
        """
        예측 결과와 메타데이터를 함께 반환
        
        Args:
            df: 입력 데이터프레임
            threshold: 예측 threshold
        
        Returns:
            예측 결과 데이터프레임 (msno, churn_proba, churn_pred 포함)
        """
        if threshold is None:
            threshold = self.threshold
        
        # msno 보존
        msno = df['msno'].copy() if 'msno' in df.columns else pd.Series(range(len(df)))
        
        # 예측
        proba = self.predict_proba(df)
        churn_proba = proba[:, 1]
        churn_pred = (churn_proba >= threshold).astype(int)
        
        # 결과 데이터프레임 생성
        results = pd.DataFrame({
            'msno': msno,
            'churn_proba': churn_proba,
            'churn_pred': churn_pred,
        })
        
        return results


# ============================================
# 편의 함수
# ============================================
def load_predictor(model_dir: Optional[Path] = None) -> ChurnPredictor:
    """
    ChurnPredictor 인스턴스 생성 (편의 함수)
    
    Args:
        model_dir: 모델 디렉토리 경로 (None이면 현재 디렉토리)
    
    Returns:
        ChurnPredictor 인스턴스
    """
    if model_dir is None:
        model_dir = MODEL_DIR
    
    return ChurnPredictor(
        model_path=model_dir / "catboost_recall_selected.cbm",
        feature_cols_path=model_dir / "feature_cols.json",
        results_path=model_dir / "recall_selected_results.json"
    )


def predict_batch(df: pd.DataFrame, 
                  model_dir: Optional[Path] = None,
                  threshold: Optional[float] = None) -> pd.DataFrame:
    """
    배치 예측 (편의 함수)
    
    Args:
        df: 입력 데이터프레임
        model_dir: 모델 디렉토리 경로
        threshold: 예측 threshold
    
    Returns:
        예측 결과 데이터프레임
    """
    predictor = load_predictor(model_dir)
    return predictor.predict_with_metadata(df, threshold=threshold)


# ============================================
# 예제 사용법
# ============================================
if __name__ == "__main__":
    # 예제 1: 기본 사용법
    print("=" * 60)
    print("KKBox Churn Prediction - Inference 예제")
    print("=" * 60)
    
    # 모델 로드
    predictor = load_predictor()
    
    # 예제 데이터 (실제로는 전처리된 데이터를 사용)
    # 예: df = pd.read_parquet("data/kkbox_train_feature_v3.parquet")
    # 예제를 위해 더미 데이터 생성
    print("\n예제: 더미 데이터로 예측")
    print("-" * 60)
    
    # 실제 사용 시:
    # df = pd.read_parquet("data/kkbox_train_feature_v3.parquet")
    # results = predictor.predict_with_metadata(df)
    # print(results.head())
    
    print("\n사용법:")
    print("1. 전처리된 데이터 로드")
    print("   df = pd.read_parquet('data/kkbox_train_feature_v3.parquet')")
    print("\n2. 예측 실행")
    print("   predictor = load_predictor()")
    print("   results = predictor.predict_with_metadata(df)")
    print("\n3. 결과 확인")
    print("   print(results.head())")
    print("\n✓ Inference 파이프라인 준비 완료!")

