"""
탐색적 데이터 분석 (EDA) 스크립트
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """CSV 파일 로드"""
    df = pd.read_csv('bank-additional-full.csv', sep=';')
    return df

def basic_info(df):
    """기본 정보 수집"""
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'target_distribution': df['y'].value_counts().to_dict() if 'y' in df.columns else None
    }
    return info

def create_plots(df, output_dir='plots'):
    """EDA 플롯 생성"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 타겟 변수 분포
    if 'y' in df.columns:
        plt.figure(figsize=(8, 6))
        df['y'].value_counts().plot(kind='bar')
        plt.title('타겟 변수 분포')
        plt.xlabel('y')
        plt.ylabel('빈도')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 수치형 변수 분포
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'y' in numeric_cols:
        numeric_cols.remove('y')
    
    if numeric_cols:
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(numeric_cols[:n_rows*n_cols]):
            axes[idx].hist(df[col].dropna(), bins=50, edgecolor='black')
            axes[idx].set_title(f'{col} 분포')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('빈도')
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/numeric_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 범주형 변수 분포 (상위 몇 개만)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        n_plots = min(6, len(categorical_cols))
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(categorical_cols[:n_plots]):
            value_counts = df[col].value_counts().head(10)
            axes[idx].barh(range(len(value_counts)), value_counts.values)
            axes[idx].set_yticks(range(len(value_counts)))
            axes[idx].set_yticklabels(value_counts.index)
            axes[idx].set_title(f'{col} (상위 10개)')
            axes[idx].set_xlabel('빈도')
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 상관관계 히트맵 (수치형 변수만)
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('수치형 변수 상관관계 히트맵')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_eda_report(df, info, output_file='result.md'):
    """EDA 결과를 마크다운으로 저장"""
    report = []
    report.append("# 탐색적 데이터 분석 (EDA) 결과\n")
    
    report.append("## 1. 데이터 기본 정보\n")
    report.append(f"- **데이터 크기**: {info['shape'][0]}행 × {info['shape'][1]}열\n")
    report.append(f"- **중복 행 수**: {info['duplicates']}개\n")
    
    report.append("\n## 2. 컬럼 정보\n")
    report.append(f"- **총 컬럼 수**: {len(info['columns'])}개\n")
    report.append("\n### 컬럼 목록:\n")
    for col in info['columns']:
        dtype = info['dtypes'].get(col, 'unknown')
        missing = info['missing_values'].get(col, 0)
        report.append(f"- `{col}` ({dtype}) - 결측치: {missing}개\n")
    
    report.append("\n## 3. 결측치 현황\n")
    missing_data = {k: v for k, v in info['missing_values'].items() if v > 0}
    if missing_data:
        for col, count in missing_data.items():
            pct = (count / info['shape'][0]) * 100
            report.append(f"- `{col}`: {count}개 ({pct:.2f}%)\n")
    else:
        report.append("- 결측치가 없습니다.\n")
    
    report.append("\n## 4. 타겟 변수 분포\n")
    if info['target_distribution']:
        for key, value in info['target_distribution'].items():
            pct = (value / info['shape'][0]) * 100
            report.append(f"- `{key}`: {value}개 ({pct:.2f}%)\n")
    
    report.append("\n## 5. 수치형 변수 통계\n")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'y' in numeric_cols:
        numeric_cols.remove('y')
    if numeric_cols:
        stats = df[numeric_cols].describe()
        report.append("\n| 변수 | 평균 | 표준편차 | 최소값 | 25% | 50% | 75% | 최대값 |\n")
        report.append("|------|------|----------|--------|-----|-----|-----|--------|\n")
        for col in numeric_cols:
            mean_val = stats.loc['mean', col]
            std_val = stats.loc['std', col]
            min_val = stats.loc['min', col]
            q25 = stats.loc['25%', col]
            q50 = stats.loc['50%', col]
            q75 = stats.loc['75%', col]
            max_val = stats.loc['max', col]
            report.append(f"| {col} | {mean_val:.2f} | {std_val:.2f} | {min_val:.2f} | {q25:.2f} | {q50:.2f} | {q75:.2f} | {max_val:.2f} |\n")
    
    report.append("\n## 6. 생성된 시각화\n")
    report.append("- `plots/target_distribution.png`: 타겟 변수 분포\n")
    report.append("- `plots/numeric_distributions.png`: 수치형 변수 분포\n")
    report.append("- `plots/categorical_distributions.png`: 범주형 변수 분포\n")
    report.append("- `plots/correlation_heatmap.png`: 상관관계 히트맵\n")
    
    # 기존 파일이 있으면 덮어쓰기
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"EDA 결과가 {output_file}에 저장되었습니다.")

if __name__ == '__main__':
    print("데이터 로딩 중...")
    df = load_data()
    
    print("기본 정보 수집 중...")
    info = basic_info(df)
    
    print("시각화 생성 중...")
    create_plots(df)
    
    print("EDA 리포트 생성 중...")
    generate_eda_report(df, info)
    
    print("EDA 완료!")


