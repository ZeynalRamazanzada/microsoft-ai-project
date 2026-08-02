"""
BDFS — Proje Çalıştırma Yöneticisi (run_all.py)
==============================================
Bu script, BDFS pipeline'ındaki tüm adımları sırasıyla çalıştırır.
Herhangi bir adımda hata oluşursa, işlemi durdurur ve hatayı raporlar.
"""

import os
import subprocess
import sys

def run_script(script_path):
    """Verilen Python dosyasını çalıştırır ve hata varsa durdurur."""
    print("=" * 70)
    print(f"BAŞLATILIYOR: {script_path}")
    print("=" * 70)
    
    try:
        # Scripti çalıştır
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True
        )
        print(f"✓ BAŞARILI: {script_path}\n")
    except subprocess.CalledProcessError as e:
        print("\n" + "!" * 70)
        print(f"HATA: {script_path} çalıştırılırken bir hata oluştu!")
        print(f"Çıkış kodu (Exit code): {e.returncode}")
        print("!" * 70)
        sys.exit(1)

def main():
    # Çalıştırılacak scriptlerin sıralı listesi
    pipeline_scripts = [
        "src/data_generator.py",
        "src/preprocessor.py",
        "src/feature_engineer.py",
        "src/train_step5.py",
        "src/evaluate_step6.py",
        "src/ablation_step7.py",
        "src/shap_step8.py"
    ]
    
    # Proje ana dizinini doğrula
    if not os.path.exists("src/data_generator.py"):
        print("HATA: run_all.py dosyası BDFS_Project ana dizininden çalıştırılmalıdır.")
        print("Örnek: python run_all.py")
        sys.exit(1)
        
    print("BDFS PIPELINE BAŞLATILIYOR...")
    print(f"Toplam {len(pipeline_scripts)} adım çalıştırılacak.\n")
    
    for script in pipeline_scripts:
        run_script(script)
        
    print("=" * 70)
    print("🎉 TEBRİKLER: Tüm BDFS Pipeline'ı Başarıyla Tamamlandı!")
    print("=" * 70)

if __name__ == "__main__":
    main()
