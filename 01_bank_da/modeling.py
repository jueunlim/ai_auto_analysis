"""
모델링 스크립트 - Random Forest와 Gradient Boosting 비교
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """데이터 로드 및 전처리"""
    df = pd.read_csv('bank-additional-full.csv', sep=';')
    
    # 타겟 변수 인코딩
    le_target = LabelEncoder()
    df['y'] = le_target.fit_transform(df['y'])
    
    # 범주형 변수 인코딩
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'y' in categorical_cols:
        categorical_cols.remove('y')
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # 특성과 타겟 분리
    X = df.drop('y', axis=1)
    y = df['y']
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def train_models(X_train, X_test, y_train, y_test):
    """Random Forest와 Gradient Boosting 모델 학습"""
    models = {}
    predictions = {}
    probabilities = {}
    
    # Random Forest
    print("Random Forest 모델 학습 중...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    predictions['Random Forest'] = rf.predict(X_test)
    probabilities['Random Forest'] = rf.predict_proba(X_test)[:, 1]
    
    # Gradient Boosting
    print("Gradient Boosting 모델 학습 중...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    predictions['Gradient Boosting'] = gb.predict(X_test)
    probabilities['Gradient Boosting'] = gb.predict_proba(X_test)[:, 1]
    
    return models, predictions, probabilities

def calculate_metrics(y_test, predictions, probabilities):
    """모델 성능 지표 계산"""
    results = {}
    
    for model_name in predictions.keys():
        auc = roc_auc_score(y_test, probabilities[model_name])
        results[model_name] = {
            'AUC-ROC': auc,
            'predictions': predictions[model_name],
            'probabilities': probabilities[model_name]
        }
    
    return results

def plot_roc_curves(y_test, probabilities, output_dir='plots'):
    """ROC 곡선 시각화"""
    Path(output_dir).mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    for model_name, prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC 곡선 비교', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC 곡선이 {output_dir}/roc_curves.png에 저장되었습니다.")

def plot_feature_importance(models, feature_names, output_dir='plots', top_n=15):
    """특성 중요도 시각화"""
    Path(output_dir).mkdir(exist_ok=True)
    
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 8))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, model) in enumerate(models.items()):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        axes[idx].barh(range(top_n), importances[indices])
        axes[idx].set_yticks(range(top_n))
        axes[idx].set_yticklabels([feature_names[i] for i in indices])
        axes[idx].set_xlabel('특성 중요도', fontsize=12)
        axes[idx].set_title(f'{model_name} - 상위 {top_n}개 특성', fontsize=13, fontweight='bold')
        axes[idx].invert_yaxis()
        axes[idx].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"특성 중요도가 {output_dir}/feature_importance.png에 저장되었습니다.")
    
    return {name: model.feature_importances_ for name, model in models.items()}

def generate_modeling_report(results, feature_importances, feature_names, output_file='result.md'):
    """모델링 결과를 마크다운으로 저장"""
    # 기존 파일 읽기 (있다면)
    existing_content = ""
    if Path(output_file).exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_content = f.read()
    
    report = []
    report.append("\n\n# 모델링 결과\n")
    
    report.append("## 1. 모델 성능 비교\n")
    report.append("| 모델 | AUC-ROC |\n")
    report.append("|------|----------|\n")
    for model_name, metrics in results.items():
        auc = metrics['AUC-ROC']
        report.append(f"| {model_name} | {auc:.4f} |\n")
    
    report.append("\n## 2. 특성 중요도 (상위 10개)\n")
    
    for model_name, importances in feature_importances.items():
        report.append(f"\n### {model_name}\n")
        indices = np.argsort(importances)[::-1][:10]
        report.append("| 순위 | 특성명 | 중요도 |\n")
        report.append("|------|--------|--------|\n")
        for rank, idx in enumerate(indices, 1):
            report.append(f"| {rank} | {feature_names[idx]} | {importances[idx]:.4f} |\n")
    
    report.append("\n## 3. 시각화\n")
    report.append("- `plots/roc_curves.png`: ROC 곡선 비교\n")
    report.append("- `plots/feature_importance.png`: 특성 중요도 비교\n")
    
    # 기존 내용에 추가
    final_content = existing_content + "\n".join(report)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print(f"모델링 결과가 {output_file}에 추가되었습니다.")

if __name__ == '__main__':
    print("데이터 로딩 및 전처리 중...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    print("모델 학습 중...")
    models, predictions, probabilities = train_models(X_train, X_test, y_train, y_test)
    
    print("성능 지표 계산 중...")
    results = calculate_metrics(y_test, predictions, probabilities)
    
    print("ROC 곡선 생성 중...")
    plot_roc_curves(y_test, probabilities)
    
    print("특성 중요도 계산 및 시각화 중...")
    feature_importances = plot_feature_importance(models, feature_names)
    
    print("결과 리포트 생성 중...")
    generate_modeling_report(results, feature_importances, feature_names)
    
    print("모델링 완료!")


