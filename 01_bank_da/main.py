"""
메인 실행 스크립트 - EDA와 모델링을 순차적으로 실행
"""
import subprocess
import sys

def run_script(script_name):
    """Python 스크립트 실행"""
    print(f"\n{'='*60}")
    print(f"{script_name} 실행 중...")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    if result.returncode != 0:
        print(f"오류: {script_name} 실행 중 문제가 발생했습니다.")
        sys.exit(1)
    return result

if __name__ == '__main__':
    print("="*60)
    print("은행 마케팅 데이터 분석 시작")
    print("="*60)
    
    # 1. EDA 실행
    run_script('eda.py')
    
    # 2. 모델링 실행
    run_script('modeling.py')
    
    print("\n" + "="*60)
    print("모든 분석이 완료되었습니다!")
    print("결과는 result.md 파일에서 확인할 수 있습니다.")
    print("="*60)


